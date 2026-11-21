# autogenesis

An RL environment whose policy builds RL environments.

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
# full reward (panel on): pin two different solver models
vf-run autogenesis -m <builder-model> \
  --env.weak.model <small-model> --env.strong.model <large-model>

# audit-only reward (no solver calls)
vf-run autogenesis -m <builder-model> --env.panel.instances 0

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
| `--env.panel.instances` | 6 | Tasks per solver. `0` = audit-only |
| `--env.panel.floor` | 0.25 | Share of the audit score paid with zero discrimination |
| `--env.sandbox.type` | subprocess | `subprocess`, `docker`, or `prime` |

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
3. **Runtime**: a `vf.Runtime`. The default `subprocess` runtime runs on the host and is
   for **development only**. For training, use `--env.sandbox.type docker` or `prime`.

Known limit: builder code runs in the same interpreter as the runner. Layers 1 and 2
make tampering with the runner hard. They do not make it impossible. Layer 3 is the
real boundary.
