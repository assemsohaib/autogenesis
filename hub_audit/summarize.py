"""Make a markdown summary of Hub audit reports.

Usage: python hub_audit/summarize.py runs/hub_audit > report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SEVERE = 0.5  # a probe that passes on at least half of the tasks
MIN_TASKS = 10
# Hand-check results. They replace the automatic verdict, with the reason.
_MANUAL_PATH = Path(__file__).with_name("manual_checks.json")
MANUAL: dict = json.loads(_MANUAL_PATH.read_text()) if _MANUAL_PATH.exists() else {}
MARGIN = 0.10  # a probe must beat the best constant answer by this much
# Each candidate in its own block: a first- or last-match parser passes about 1/N by luck.
POSITION = {"shotgun_each", "shotgun_each_reversed"}
# Responses with no task content. Their mean reward must stay far below the reference.
JUNK = {"empty", "zero", "one", "yes", "true", "none", "json_claim", "injection",
        "long_junk", "number_spray", "raw_empty", "echo_prompt", "prompt_numbers"}  # fmt: skip


def row(report: dict) -> tuple[str, list[str]]:
    env = report.get("env", "?")
    if report.get("stage") != "complete":
        err = (report.get("error") or "").splitlines()
        reason = err[-1][:120] if err else ""
        return f"| {env} | failed at `{report.get('stage')}` | | | | | {reason} |", []
    probes = report.get("probes") or {}
    base = report.get("constant_baseline") or {}
    base_pass, base_mean = base.get("pass_rate", 0.0), base.get("mean_reward", 0.0)
    hits = sorted(
        (
            (name, p["pass_rate"])
            for name, p in probes.items()
            if p.get("n") and p["pass_rate"] > 0 and not name.startswith("const_")
        ),
        key=lambda x: -x[1],
    )
    worst = ", ".join(f"{name} {rate:.0%}" for name, rate in hits[:5]) or "none"
    # A probe counts only if it beats both luck by position and the best constant answer.
    real = [
        (n, r) for n, r in hits if (n not in POSITION or r >= SEVERE) and r > base_pass + MARGIN
    ]
    ref_mean = report.get("reference_mean") or 0.0
    junk_mean = max((p.get("mean_reward", 0.0) for n, p in probes.items() if n in JUNK), default=0)
    junk_wins = junk_mean > base_mean + MARGIN * max(ref_mean, 1e-9) and junk_mean >= ref_mean
    if (report.get("tasks") or 0) < MIN_TASKS:
        verdict = f"too few tasks (<{MIN_TASKS})"
    elif not report.get("tasks_with_answer") or report.get("reference_mean") is None:
        verdict = "no reference: check by hand"
    elif (real and real[0][1] >= SEVERE) or (junk_mean > 0 and junk_wins):
        verdict = "**severe**"
    elif real:
        verdict = "mild"
    elif hits:
        verdict = "no gain over luck"
    else:
        verdict = "no probe passed"
    ref = f"{report.get('reference_format')} ({ref_mean:.2f})"
    baseline = f"{base_pass:.0%}" if base else "n/a"
    notes = []
    if ref_mean < SEVERE:
        notes.append("reference format scored low: pass rule unreliable, check by hand")
    if junk_mean > 0:
        notes.append(f"best junk mean reward {junk_mean:.2f} vs reference {ref_mean:.2f}")
    if report.get("undeclared_deps"):
        notes.append("undeclared deps: " + ", ".join(report["undeclared_deps"]))
    if not report.get("tasks_with_answer"):
        notes.append("no answer field: answer probes not run")
    manual = report.get("manual_check") or MANUAL.get(env)
    if manual:
        verdict = manual["verdict"]
        notes.insert(0, manual["note"])
    line = (
        f"| {env} | {verdict} | {report.get('mode')} | {report.get('tasks', '')} "
        f"| {ref} | {baseline} | {worst}{'; ' + '; '.join(notes) if notes else ''} |"
    )
    examples = [
        f"- `{env}` / `{name}`: {probes[name].get('example', {})}"
        for name, rate in real[:3]
        if rate >= SEVERE and verdict == "**severe**"
    ]
    return line, examples


def main() -> None:
    root = Path(sys.argv[1])
    reports = [json.loads(p.read_text()) for p in sorted(root.glob("*/report.json"))]
    print(
        "| Environment | Result | Mode | Tasks | Reference format (reward) "
        "| Best constant answer | Passing probes |"
    )
    print("|---|---|---|---|---|---|---|")
    examples: list[str] = []
    for report in reports:
        line, ex = row(report)
        print(line)
        examples += ex
    if examples:
        print("\n## Severe examples\n")
        print("\n".join(e[:600] for e in examples))


if __name__ == "__main__":
    main()
