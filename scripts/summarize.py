"""Summarize autogenesis runs: python scripts/summarize.py runs/baseline/*/traces.jsonl

Prints, per builder model: how far audits got, mean audit metrics, panel results,
which probes beat the verifiers, and the most common errors.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from statistics import mean

STAGES = ["extract", "policy", "generate", "verify", "complete"]
KEYS = ["audit_score", "valid", "determinism", "diversity", "leak", "solvability",
        "hack_resistance", "weak_accuracy", "strong_accuracy", "discrimination"]  # fmt: skip


def load(paths: list[str]) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        with open(path) as f:
            lines = f.read().splitlines()
        for line in lines:
            episode = json.loads(line)
            builder = episode["traces"][0]
            # run name: autogenesis--<org>--<model>--<harness>--<id>
            model = "/".join(episode["run"]["name"].split("--")[1:-2])
            by_model[model].append(
                {
                    "brief": episode["task"]["data"].get("id"),
                    "family": episode["task"]["data"].get("family"),
                    "reward": builder["rewards"].get("autogenesis", {}).get("score", 0.0),
                    "metrics": builder["metrics"],
                    "info": builder["info"].get("autogenesis", {}),
                    "ok": episode["ok"],
                    "dropped": builder.get("dropped"),
                }
            )
    return by_model


def report(model: str, rows: list[dict]) -> None:
    print(f"\n## {model}  ({len(rows)} episodes, {sum(r['ok'] for r in rows)} ok)")
    stages = Counter(r["info"].get("stage", "none") for r in rows)
    print("stage reached:", {s: stages.get(s, 0) for s in STAGES})
    scored = [r for r in rows if not r["dropped"]]
    print(f"mean builder reward: {mean(r['reward'] for r in scored):.3f} "
          f"({len(rows) - len(scored)} builder traces dropped)")  # fmt: skip
    complete = [r for r in rows if r["info"].get("stage") == "complete"]
    if complete:
        print("means over completed audits:")
        for key in KEYS:
            values = [r["metrics"][key] for r in complete if key in r["metrics"]]
            if values:
                print(f"  {key:16s} {mean(values):.3f}")
        hr = [r["metrics"].get("hack_resistance", 1.0) for r in complete]
        severe = sum(h < 0.5 for h in hr)
        mild = sum(0.5 <= h < 1 for h in hr)
        print(f"hack_resistance: severe (<0.5) {severe}, mild (0.5-1) {mild}, "
              f"none hacked {len(hr) - severe - mild}  (of {len(hr)} completed)")  # fmt: skip
        hacked = [r for r in complete if r["metrics"].get("hack_resistance", 1) < 1]
        solvable = [r for r in complete if r["metrics"].get("solvability", 0) == 1]
        hacked_solvable = [r for r in solvable if r["metrics"].get("hack_resistance", 1) < 1]
        print(
            f"hackable verifiers: {len(hacked)}/{len(complete)} completed audits; "
            f"{len(hacked_solvable)}/{len(solvable)} of those whose reference passes"
        )
        worst = Counter(r["info"]["worst_probe"] for r in hacked)
        print("worst probe on hackable verifiers:", dict(worst.most_common()))
        beaten = Counter(
            name
            for r in complete
            for name, rate in r["info"].get("probe_pass_rates", {}).items()
            if rate > 0
        )
        print("probes that pass at least once, per env:", dict(beaten.most_common()))
    errors = Counter((r["info"].get("error") or "")[:90] for r in rows if r["info"].get("error"))
    if errors:
        print("errors:")
        for text, n in errors.most_common(8):
            print(f"  {n}x {text}")
    severe_list = [
        (r["brief"], r["info"].get("worst_probe"), round(r["metrics"]["hack_resistance"], 2))
        for r in complete
        if r["metrics"].get("hack_resistance", 1) < 0.5
    ]
    print("severely hackable:", severe_list)
    best = sorted(rows, key=lambda r: -r["reward"])[:5]
    print("top briefs:", [(r["brief"], round(r["reward"], 2)) for r in best])


if __name__ == "__main__":
    for model, rows in load(sys.argv[1:]).items():
        report(model, rows)
