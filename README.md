# Autogenesis

Train a model that turns a capability target (a description plus a few eval items)
into an RL environment that trains that target. The environment must be correct, hard
to hack, and discriminative, and training on it must transfer to the target.

This repository holds the first stage: the `autogenesis` environment for the
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 API. See
[`environments/autogenesis/README.md`](environments/autogenesis/README.md) for the
environment.

```
environments/autogenesis/   the environment package (taskset id: autogenesis)
tests/                      unit, sandbox, and end-to-end tests (fake model server)
scripts/summarize.py        per-model summary of vf-run traces
```

## Baseline (stage 1)

Three open builder models, all 35 briefs, one build per brief, 24 hidden seeds per
audit. Weak solver `mistralai/mistral-nemo`, strong solver
`deepseek/deepseek-v4.1-flash-1017`. Total inference cost: $0.77.

| Builder | Audit complete | Reference passes | Hackable (any probe) | Severe (hack_resistance < 0.5) | Discrimination |
|---|---|---|---|---|---|
| google/gemma-4-31b-it | 29/35 | 29 | 29/29 | 2 | 0.25 |
| openai/gpt-oss-120b | 28/35 | 27 | 27/27 | 5 | 0.24 |
| qwen/qwen3.6-35b-a3b | 27/35 | 24 | 21/24 | 4 | 0.15 |
| **total** | **84/105** | **80** | **77/80 (96%)** | **11/84 (13%)** | |

- Model-written verifiers almost never enforce "exactly one answer", even when the
  prompt asks for it. Most accept a response with several candidate answers on some
  tasks. Some accept it on every task.
- Other severe holes: a copy of the prompt passes (base conversion), and answer
  distributions are so unbalanced that the most common answer passes on about half of
  the tasks.
- Discrimination is low: the 12B weak solver solves 34–52% of the generated tasks.
- About 20% of builds fail as code (syntax errors, banned imports or names, timeouts).

Treat these numbers as rough: one build per brief and one seed set.

## Roadmap

1. Audit existing Environments Hub environments with the same probes.
2. Target-driven builds: the input is a description plus a few items of an eval, and
   the reward includes transfer (training a student on the environment must improve
   held-out items of the target).
3. A learned Hacker in place of the static probes (self-play).
4. Agentic environments with tools and hidden state.

## Development

```bash
uv sync                      # Python 3.11–3.14 (verifiers does not support 3.15)
uv run pytest                # 49 tests, about 2 min, no network or keys needed
uv run pytest -m "not integration"   # skip the full vf-run episode
uv run ruff check . && uv run ruff format --check .
```

## Run against real models

```bash
export PRIME_API_KEY=...     # or: prime login
uv run vf-run autogenesis -m <builder> -n 4 \
  --env.weak.model <small> --env.strong.model <large> --env.audit.seed 0
```
