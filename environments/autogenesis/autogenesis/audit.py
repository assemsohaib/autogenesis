"""The audit: score one builder reply without any model calls.

Pipeline (each stage can end the audit with score 0):

  extract   find the python code block in the reply
  policy    static checks on the code (`policy.py`)
  generate  run generate()/solve() on hidden random seeds in the sandbox
  verify    score reference answers (twice, for stability) and every probe,
            in one shuffled batch so a stateful verify() cannot use call order

The audit score is a product, so no single term can be maximized alone:

  score = valid * determinism * solvability * hack_resistance * (1 - leak)
          * ((1 - w) + w * diversity)

The model panel (`env.py`) multiplies this by a discrimination term.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field
from pydantic_config import BaseConfig

from autogenesis.policy import check_code
from autogenesis.probes import PROBE_NAMES, build_probes
from autogenesis.sandbox import Executor

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
_SPACE = re.compile(r"\s+")


class AuditConfig(BaseConfig):
    num_tasks: int = Field(24, ge=4, le=256)
    """Task instances generated from hidden random seeds."""
    min_valid_fraction: float = Field(0.9, ge=0.0, le=1.0)
    """Below this fraction of working seeds, the audit stops at `generate`."""
    pass_threshold: float = Field(0.5, ge=0.0, le=1.0)
    """A verify() score at or above this counts as a pass (probes and solvers)."""
    solved_threshold: float = Field(0.99, ge=0.0, le=1.0)
    """A reference answer must score at least this to count as solved."""
    diversity_weight: float = Field(0.5, ge=0.0, le=1.0)
    leak_min_chars: int = Field(4, ge=1)
    """Answers shorter than this are not checked for leaks (a "3" is in many prompts)."""
    call_timeout: float = Field(2.0, gt=0)
    """Seconds for each generate/solve/verify call."""
    sandbox_timeout: float = Field(120.0, gt=0)
    """Seconds for one runner process (one operation over the whole batch)."""
    memory_mb: int = Field(1024, ge=128)
    cpu_seconds: int = Field(60, ge=1)
    python: str = "python3"
    """Interpreter inside the sandbox runtime."""
    seed: int | None = None
    """Fix the hidden seeds (tests, debugging). None draws fresh seeds per episode."""


@dataclass
class AuditReport:
    stage: str
    """The last stage reached: extract, policy, generate, verify, or complete."""
    error: str | None = None
    code: str = ""
    violations: list[str] = field(default_factory=list)
    valid: float = 0.0
    determinism: float = 0.0
    diversity: float = 0.0
    leak: float = 0.0
    solvability: float = 0.0
    reference_stability: float = 0.0
    hack_resistance: float = 0.0
    probe_pass_rates: dict[str, float] = field(default_factory=dict)
    worst_probe: str | None = None
    score: float = 0.0
    tasks: list[dict[str, Any]] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    solved: list[bool] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.stage == "complete"

    def metrics(self) -> dict[str, float]:
        stages = ("extract", "policy", "generate", "verify", "complete")
        return {
            "audit_score": self.score,
            "audit_complete": float(self.complete),
            "audit_stage": float(stages.index(self.stage)),
            "valid": self.valid,
            "determinism": self.determinism,
            "diversity": self.diversity,
            "leak": self.leak,
            "solvability": self.solvability,
            "reference_stability": self.reference_stability,
            "hack_resistance": self.hack_resistance,
        }

    def summary(self) -> dict[str, Any]:
        """A small JSON record for `trace.info`, for people reading traces."""
        return {
            "stage": self.stage,
            "error": self.error,
            "violations": self.violations[:10],
            "worst_probe": self.worst_probe,
            "probe_pass_rates": self.probe_pass_rates,
            "sample_prompts": [t.get("prompt", "")[:500] for t in self.tasks[:3]],
            "sample_answers": [a[:200] for a in self.answers[:3]],
        }


def extract_code(reply: str) -> str | None:
    """The last ```python block in `reply`, else the last fenced block of any language."""
    blocks = _FENCE.findall(reply or "")
    if not blocks:
        return None
    python = [body for lang, body in blocks if lang.lower() in {"python", "py", "python3"}]
    chosen = python[-1] if python else blocks[-1][1]
    return chosen.strip() or None


def _normalize(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


def _passes(entry: list, threshold: float) -> bool:
    score, _error = entry
    return score is not None and score >= threshold


def leak_rate(prompts: list[str], answers: list[str], min_chars: int) -> float:
    """Share of tasks whose reference answer is printed in their own prompt.

    An answer that is also in most *other* prompts is template text, not a leak: for
    example "valid" in "Is the argument valid or invalid?". Guessing such answers is
    the job of the `mode_answer` probe.
    """
    normalized = [_normalize(p) for p in prompts]
    leaked = 0
    for i, answer in enumerate(answers):
        norm = _normalize(answer)
        if len(norm) < min_chars or norm not in normalized[i]:
            continue
        others = [p for j, p in enumerate(normalized) if j != i]
        if sum(norm in p for p in others) < 0.5 * len(others):
            leaked += 1
    return leaked / len(answers)


async def run_audit(
    reply: str,
    executor: Executor,
    config: AuditConfig,
    rng: random.Random | None = None,
) -> AuditReport:
    rng = rng or (random.Random(config.seed) if config.seed is not None else random.SystemRandom())

    code = extract_code(reply)
    if code is None:
        return AuditReport(stage="extract", error="no fenced code block in the reply")

    policy = check_code(code)
    if not policy.ok:
        return AuditReport(
            stage="policy", code=code, violations=policy.violations, error=policy.violations[0]
        )

    seeds = rng.sample(range(1, 2**31 - 1), config.num_tasks)
    generated = await executor("generate", code=code, seeds=seeds)
    if "fatal" in generated:
        return AuditReport(stage="generate", code=code, error=generated["fatal"])

    records = generated["records"]
    good = [r for r in records if "error" not in r]
    valid = len(good) / len(records)
    if valid < config.min_valid_fraction or len(good) < 2:
        first_error = next((r["error"] for r in records if "error" in r), "too few tasks")
        return AuditReport(stage="generate", code=code, valid=valid, error=first_error)

    tasks = [r["task"] for r in good]
    answers = [r["answer"] for r in good]
    prompts = [t["prompt"] for t in tasks]
    determinism = sum(bool(r["deterministic"]) for r in good) / len(good)
    diversity = len(set(prompts)) / len(prompts)
    leak = leak_rate(prompts, answers, config.leak_min_chars)

    # One shuffled batch: two reference checks per task, then every probe.
    labels: list[tuple[str, int]] = []
    items: list[list] = []
    for i, answer in enumerate(answers):
        for copy_name in ("reference_a", "reference_b"):
            labels.append((copy_name, i))
            items.append([i, answer])
    for name, i, response in build_probes(prompts, answers):
        labels.append((name, i))
        items.append([i, response])
    order = list(range(len(items)))
    rng.shuffle(order)
    verified = await executor("verify", code=code, tasks=tasks, items=[items[j] for j in order])
    report = AuditReport(
        stage="verify",
        code=code,
        valid=valid,
        determinism=determinism,
        diversity=diversity,
        leak=leak,
        tasks=tasks,
        answers=answers,
    )
    if "fatal" in verified:
        report.error = verified["fatal"]
        return report

    scores = verified.get("scores")
    if not isinstance(scores, list) or len(scores) != len(items):
        report.error = "sandbox returned the wrong number of scores"
        return report
    results: dict[tuple[str, int], list] = {}
    for position, j in enumerate(order):
        results[labels[j]] = scores[position]

    solved = []
    stable = []
    for i in range(len(tasks)):
        a, b = results[("reference_a", i)], results[("reference_b", i)]
        solved.append(_passes(a, config.solved_threshold) and _passes(b, config.solved_threshold))
        stable.append(a[0] == b[0])
    report.solved = solved
    report.solvability = sum(solved) / len(solved)
    report.reference_stability = sum(stable) / len(stable)

    passes: dict[str, list[bool]] = defaultdict(list)
    for (name, _i), entry in results.items():
        if not name.startswith("reference_"):
            passes[name].append(_passes(entry, config.pass_threshold))
    report.probe_pass_rates = {
        name: sum(passes[name]) / len(passes[name]) for name in PROBE_NAMES if passes[name]
    }
    report.worst_probe, worst_rate = max(report.probe_pass_rates.items(), key=lambda kv: kv[1])
    report.hack_resistance = 1.0 - worst_rate

    w = config.diversity_weight
    report.score = (
        valid
        * determinism
        * report.solvability
        * report.hack_resistance
        * (1.0 - leak)
        * ((1.0 - w) + w * diversity)
    )
    report.stage = "complete"
    if report.solvability == 0.0:
        first = next((results[("reference_a", i)][1] for i in range(len(tasks))), None)
        report.error = f"verify() rejects every reference answer ({first or 'score too low'})"
    return report
