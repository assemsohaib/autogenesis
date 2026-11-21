"""Autogenesis: the builder writes an RL environment; the env audits it and tests it
on a weak and a strong solver.

One episode:

  1. builder   (trainable) gets a domain brief and writes generate/solve/verify.
  2. audit     runs the code in a sandbox runtime on hidden seeds and attacks the
               verifier with the probe suite (`audit.py`). No model calls.
  3. panel     if the audit completes, a weak and a strong solver (both fixed, not
               trained) attempt a sample of the generated tasks. The builder's
               verify() grades them in the sandbox.
  4. reward    autogenesis = audit_score * (floor + (1 - floor) * discrimination),
               discrimination = max(0, strong_accuracy - weak_accuracy).

Only the builder's trace carries a reward. The solver traces are untrainable and
carry only a `solved` metric, so they do not dilute episode scores.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass

import verifiers.v1 as vf
from pydantic import Field
from pydantic_config import BaseConfig
from verifiers.v1.configs.runtime import DockerConfig, RuntimeConfig, SubprocessConfig
from verifiers.v1.harnesses.null import NullHarnessConfig
from verifiers.v1.runtimes import provision_runtime

from autogenesis.audit import AuditConfig, AuditReport, run_audit
from autogenesis.prompts import SOLVER_SYSTEM_PROMPT
from autogenesis.sandbox import Executor, SandboxExecutor

logger = logging.getLogger(__name__)

DEFAULT_PANEL_INSTANCES = 6


def _chat_role() -> vf.AgentConfig:
    """A tool-less chat seat. It runs no model-written code, so the local
    subprocess runtime is safe for it."""
    return vf.AgentConfig(harness=NullHarnessConfig(id="null"), runtime=SubprocessConfig())


class PanelConfig(BaseConfig):
    instances: int | None = Field(None, ge=0, le=64)
    """Generated tasks each solver attempts. 0 turns the panel off (audit-only reward).
    None (default): 6 when --env.weak.model and --env.strong.model are both set,
    otherwise 0 (audit-only, with a warning)."""
    floor: float = Field(0.25, ge=0.0, le=1.0)
    """Share of the audit score paid even with zero discrimination, so early training
    still gets signal from the audit alone."""


class AutogenesisEnvConfig(vf.EnvConfig):
    builder: vf.AgentConfig = _chat_role()
    """The policy: writes the environment. Unpinned model = the run's model."""
    weak: vf.AgentConfig = _chat_role()
    """Fixed weak solver. Pin it: --env.weak.model <id>."""
    strong: vf.AgentConfig = _chat_role()
    """Fixed strong solver. Pin it: --env.strong.model <id>."""
    audit: AuditConfig = AuditConfig()
    panel: PanelConfig = PanelConfig()
    sandbox: RuntimeConfig = DockerConfig(allow=[], memory=1.0, cpu=1.0)
    """Where builder code runs. Default: a Docker container with no network, 1 GB and
    1 CPU. `prime` runs it in a Prime sandbox. `subprocess` runs untrusted model code
    on this machine: use it only for development (--env.sandbox.type subprocess)."""


@dataclass
class PanelResult:
    weak_accuracy: float = 0.0
    strong_accuracy: float = 0.0
    error: str | None = None
    pairs: int = 0
    """Generated tasks where both solvers answered. Accuracies count only these."""
    infra_failure: bool = False
    """The panel could not measure anything for reasons outside the builder's code
    (solver API errors). The builder's trace is then dropped, not scored 0."""

    @property
    def discrimination(self) -> float:
        return max(0.0, self.strong_accuracy - self.weak_accuracy)


def combine(report: AuditReport, panel: PanelResult | None, floor: float) -> float:
    """The builder's reward. Without a panel, the audit score alone."""
    if panel is None:
        return report.score
    return report.score * (floor + (1.0 - floor) * panel.discrimination)


class AutogenesisEnv(vf.Env[AutogenesisEnvConfig]):
    def __init__(self, config: AutogenesisEnvConfig) -> None:
        super().__init__(config)
        weak, strong = config.weak.model, config.strong.model
        if config.panel.instances is None:
            if weak is None or strong is None:
                logger.warning(
                    "autogenesis: solver panel off (audit-only reward). Set "
                    "--env.weak.model and --env.strong.model to turn it on."
                )
            pinned = weak is not None and strong is not None
            config.panel.instances = DEFAULT_PANEL_INSTANCES if pinned else 0
        if isinstance(config.sandbox, SubprocessConfig):
            logger.warning(
                "autogenesis: sandbox type 'subprocess' runs model-written code on this "
                "machine. Use it only for development."
            )
        if config.panel.instances > 0 and (weak is None or strong is None or weak == strong):
            raise ValueError(
                "the solver panel needs two different pinned models: set "
                "--env.weak.model and --env.strong.model, or turn the panel off "
                "with --env.panel.instances 0"
            )

    async def setup(self, agents: vf.Agents) -> None:
        # The solvers measure the builder's environment; they are never trained.
        agents.weak.trainable = False
        agents.strong.trainable = False

    def _executor(self, runtime: vf.Runtime) -> Executor:
        audit = self.config.audit
        return SandboxExecutor(
            runtime,
            python=audit.python,
            call_timeout=audit.call_timeout,
            sandbox_timeout=audit.sandbox_timeout,
            memory_mb=audit.memory_mb,
            cpu_seconds=audit.cpu_seconds,
        )

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        builder = await agents.builder.run(task)
        if not builder.ok:
            return  # a failed builder run already fails the episode

        audit_config = self.config.audit
        rng = (
            random.Random(audit_config.seed)
            if audit_config.seed is not None
            else random.SystemRandom()
        )
        panel: PanelResult | None = None
        async with provision_runtime(self.config.sandbox) as runtime:
            executor = self._executor(runtime)
            report = await run_audit(builder.last_reply, executor, audit_config, rng)
            if self.config.panel.instances > 0:
                panel = PanelResult(error="audit incomplete")
                if report.complete and report.score > 0:
                    panel = await self._run_panel(report, agents, executor)

        builder.record_metrics(report.metrics())
        if panel is not None:
            builder.record_metrics(
                {
                    "weak_accuracy": panel.weak_accuracy,
                    "strong_accuracy": panel.strong_accuracy,
                    "discrimination": panel.discrimination,
                }
            )
        builder.record_reward("autogenesis", combine(report, panel, self.config.panel.floor))
        builder.info["autogenesis"] = {
            **report.summary(),
            "panel_error": panel.error if panel else None,
            "panel_pairs": panel.pairs if panel else 0,
        }
        if panel is not None and panel.infra_failure:
            builder.drop(f"solver panel failed: {panel.error}")

    def complete(self, episode: vf.Episode) -> bool:
        # A solver API error fails only that solver's trace (it is dropped and the
        # panel uses the other tasks), so the episode is valid when the builder is.
        builders = [t for t in episode.traces if t.agent.name == "builder"]
        return bool(builders) and all(t.ok for t in builders)

    async def _run_panel(
        self, report: AuditReport, agents: vf.Agents, executor: Executor
    ) -> PanelResult:
        # Only tasks the reference solves: an unsolved task says nothing about solvers.
        indices = [i for i, ok in enumerate(report.solved) if ok][: self.config.panel.instances]
        tasks = [report.tasks[i] for i in indices]
        solver_tasks = [
            vf.Task(
                vf.TaskData(
                    name=f"generated-{n}",
                    prompt=generated["prompt"],
                    system_prompt=SOLVER_SYSTEM_PROMPT,
                )
            )
            for n, generated in enumerate(tasks)
        ]
        weak_traces, strong_traces = await asyncio.gather(
            asyncio.gather(*(agents.weak.run(t) for t in solver_tasks)),
            asyncio.gather(*(agents.strong.run(t) for t in solver_tasks)),
        )
        traces = [*weak_traces, *strong_traces]
        count = len(tasks)
        for trace in traces:
            if not trace.ok:
                trace.drop("solver call failed")
        # Pair by task: a task counts only when both solvers answered it, so one
        # solver's outage cannot look like discrimination.
        pairs = [n for n in range(count) if weak_traces[n].ok and strong_traces[n].ok]
        if not pairs:
            return PanelResult(error="no task got answers from both solvers", infra_failure=True)
        items = [
            [n, trace.last_reply] for n in pairs for trace in (weak_traces[n], strong_traces[n])
        ]
        verified = await executor("verify", code=report.code, tasks=tasks, items=items)
        if "fatal" in verified:
            return PanelResult(error=verified["fatal"], pairs=len(pairs))

        threshold = self.config.audit.pass_threshold
        scores = verified.get("scores")
        if not isinstance(scores, list) or len(scores) != len(items):
            return PanelResult(error="sandbox returned a wrong score count", pairs=len(pairs))
        solved = [score is not None and score >= threshold for score, _error in scores]
        weak_solved, strong_solved = solved[0::2], solved[1::2]
        for n, weak_ok, strong_ok in zip(pairs, weak_solved, strong_solved, strict=True):
            weak_traces[n].record_metrics({"solved": float(weak_ok)})
            strong_traces[n].record_metrics({"solved": float(strong_ok)})
        lost = count - len(pairs)
        return PanelResult(
            weak_accuracy=sum(weak_solved) / len(pairs),
            strong_accuracy=sum(strong_solved) / len(pairs),
            pairs=len(pairs),
            error=f"{lost} task(s) lost to solver errors" if lost else None,
        )
