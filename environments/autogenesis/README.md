# autogenesis

An RL environment whose policy builds RL environments. Source, tests and baseline
results: https://github.com/assemsohaib/autogenesis

The policy (the **builder**) gets a short domain brief and writes a Python module with
three functions:

```python
def generate(seed: int) -> dict        # one task; "prompt" is what a solver sees
def solve(task: dict) -> str           # a reference answer
def verify(task: dict, response: str) -> float   # the reward function, 0..1
```

The environment then attacks that module and pays the builder only for environments
that survive.

## One episode

| Step | What happens | Model calls |
|---|---|---|
| 1. build | `builder` (trainable) writes the module for one brief | 1 |
| 2. audit | The code runs in a sandbox on hidden random seeds. The audit measures validity, determinism, diversity, answer leaks, and solvability, then attacks `verify()` with 16 reward-hacking probes | 0 |
| 3. panel | A fixed `weak` and a fixed `strong` solver attempt up to 6 generated tasks. The builder's own `verify()` grades them | 2 × instances |
| 4. reward | `autogenesis = audit_score × (floor + (1 − floor) × max(0, strong_acc − weak_acc))` | 0 |

`audit_score = valid × determinism × solvability × hack_resistance × (1 − leak) × ((1 − w) + w × diversity)`

The audit is multiplicative on purpose: one broken property gives zero reward. For
example, a verifier that the "shotgun" probe (all reference answers in one message)
beats gets `hack_resistance = 0`.

### The probe suite (a static Hacker)

Constant responses: empty, whitespace, `0`, `1`, `yes`, `true`, `None`, a JSON claim
of correctness, a prompt injection, 5000 characters of junk, and every integer from −50 to 1000.
Task-dependent responses: the prompt itself, every number in the prompt, all
reference answers at once, the most common answer of the other tasks, and a different task's answer.
`hack_resistance = 1 − max probe pass rate`. Reference answers and probes are mixed in
one shuffled batch, so `verify()` cannot tell which call is the audit.

## Taskset

35 domain briefs in 10 families. `split=train` (default) gives 29 briefs in 8 families.
`split=eval` gives 6 briefs in the held-out families `grids` and `code_tracing`.
`split=all` gives all 35. `repeats=N` repeats each brief.

## Usage

```bash
prime env install assemsohaib/autogenesis
```

Builder code runs in a Docker container with no network by default, so a running
Docker daemon is needed (or use `--env.sandbox.type prime`).

```bash
# audit-only reward (default when no solvers are set; no solver calls)
vf-run autogenesis -m <builder-model>

# full reward: pin two different solver models; the panel then turns on
vf-run autogenesis -m <builder-model> \
  --env.weak.model <small-model> --env.strong.model <large-model>

# held-out families, 4 attempts per brief, solver calls in parallel
vf-run autogenesis -m <builder-model> -r 4 --env.taskset.split eval \
  --env.weak.model <small> --env.strong.model <large> --env.max-concurrent-agents None
```

Main options (`vf-run autogenesis --help` shows all):

| Option | Default | Meaning |
|---|---|---|
| `--env.audit.num-tasks` | 24 | Hidden seeds per audit |
| `--env.audit.seed` | random | Fix the hidden seeds (reproducible evals) |
| `--env.audit.call-timeout` | 2.0 | Seconds per builder function call |
| `--env.audit.diversity-weight` | 0.5 | Weight of the diversity term |
| `--env.panel.instances` | auto | Tasks per solver. `0` = audit-only. Auto: 6 when both solvers are set, else 0 |
| `--env.panel.floor` | 0.25 | Share of the audit score paid with zero discrimination |
| `--env.sandbox.type` | docker | `docker` (no network, 1 GB, 1 CPU), `prime`, or `subprocess` (development only) |

## Metrics

On the builder trace: `audit_score`, `audit_stage` (0 extract, 1 policy, 2 generate,
3 verify, 4 complete), `valid`, `determinism`, `diversity`, `leak`, `solvability`,
`reference_stability`, `hack_resistance`, `weak_accuracy`, `strong_accuracy`,
`discrimination`. `info.autogenesis` contains the error, policy violations, per-probe
pass rates, the worst probe, and sample prompts and answers.

Solver traces are untrainable. They carry a `solved` metric and no reward. A solver
trace whose model call fails is dropped, and the panel scores only the tasks that both
solvers answered (`info.autogenesis.panel_pairs`). If no task is left, the builder
trace is dropped too: an infrastructure failure never counts as a bad environment.
Note: the `mean reward` line that `vf-run` prints is the mean over *all* traces, so it
includes the reward-less solver traces. Read the builder reward from `traces.jsonl`.

## Security

Builder code is untrusted model output. There are three layers:

1. **Static policy** (`policy.py`): an AST check with a module allowlist, no
   `eval`/`exec`/`open`/`getattr`/…, no dunder names or attributes, no frame
   introspection, no `global`/`nonlocal`.
2. **Runner** (`runner.py`, stdlib only, runs inside the sandbox): restricted builtins,
   an import hook with the same allowlist, an empty environment, rlimits (memory, CPU,
   no file writes, no core dumps), a time limit for each call, and a per-call result
   nonce on a duplicated stdout.
3. **Runtime**: a `vf.Runtime`. The default is a Docker container (`python:3.11-slim`)
   with no network, 1 GB of memory and 1 CPU. `prime` runs the code in a Prime
   sandbox. `subprocess` runs untrusted model code **on your machine**: use it only for
   development. The environment logs a warning when you select it.

Known limit: builder code runs in the same interpreter as the runner. Layers 1 and 2
make tampering with the runner hard. They do not make it impossible. Layer 3 is the
real boundary.
