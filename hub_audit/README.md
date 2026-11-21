# Hub reward audit

Run Hub packages only in the audit Docker image. Do not install third-party Hub
code on the host. Prime CLI resolves packages; each container installs dependencies
at the package's publish date. Dependency resolution uses `--upgrade` with the
publication cutoff so cached installations do not retain a newer verifier. Model
provider credentials are not mounted in the container.

```sh
docker build -t autogenesis-hubaudit hub_audit
python3 hub_audit/run.py --out runs/hub_v1 --examples 40 \
  primeintellect/math500 primeintellect/gsm8k
python3 hub_audit/run.py --out runs/hub_multi --examples 20 \
  --env-args '{"mode":"multi"}' om-shastri/instruction-adherence
python3 hub_audit/run.py --out runs/hub_judge --judge puiji/flawed-judge
python3 hub_audit/summarize.py runs/hub_v1
```

Use a new output directory for a new experiment. An existing report is reused,
including a failed report. `--resume` reuses completed phases of a failed run;
it rejects changed plans, package/verifier versions, sample counts or configuration.
Complete cached reports also check configuration and publication time. `--jobs 1`
and `--timeout 5400` are the defaults. `--image`
can select a local audit image that already has dependency downloads in its cache.
`--taskset-config FILE.json` passes configuration to a v1 taskset. `--env-args JSON`
passes arguments to a v0 environment (for example, select its multi-turn mode).
The report records configuration overrides.

## Scoring paths

- **v0:** run the package's `vf-eval` against a fake OpenAI server. In a multi-turn
  rollout, repeat the selected probe after every environment response. This audits
  static responses, not a learned tool-using policy. Modern legacy harnesses
  reject literal empty replies before scoring; those payloads are recorded as
  skipped, with per-probe task counts. Legacy worker executors are capped at eight
  threads so the framework's 512-thread default cannot exhaust the container PID
  cap. The server uses the original
  prompt and previous assistant response to keep concurrent rollouts separate.
- **v1:** load the taskset inside Docker. Build a trace, run setup, append one sampled
  assistant reply, run finalize, and call the task's own score method. Each probe
  gets a fresh task, trace and local runtime workspace. Two probes run at a time.
  This path supports static reward checks; it does not simulate arbitrary v1 agent
  or tool interactions. The local runtime uses the image's installed `uv` instead
  of downloading it again. The task's reward scripts and dependencies still run
  through verifiers' own runtime methods.

Discovery, reference-format checks and probes have separate `plan.json`,
`rows.jsonl` and `requests.jsonl` artifacts. v0 rows include the conversation;
v1 rows include independent metrics, such as `true_correct` beside judge reward.
Scoring errors, non-finite rewards, failed evaluations and missing probe results
stop the audit. No answer field means no calibrated reference and no automatic
robustness verdict. Fewer than ten distinct prompts means no verdict. Environments
that reuse a prompt for different hidden answers need another task identity adapter.

## Live judges through Prime CLI

`--judge` uses the host's existing `prime login` or project `PRIME_API_KEY`. It
requires explicit authorization for billable model calls. The model is Mistral
Nemo, replacing the environment's default judge; results apply to this model.
The original judge prompt and scoring code are retained. Both `task.judge` and
native `task.judges` lists are supported; question fields, answer fields and
verdict labels are preserved from the validated task configuration. For
`flawed-judge`, both
its no-reference reward and reference-judge metric run, beside the code check.

The host relay calls `prime inference chat`; the container reaches it through a
mounted Unix socket and a local HTTP forwarder. A random relay token gives access
only to that relay, not to the Prime account. The relay opens no host TCP port and passes no Prime
account credentials to the container. A Docker test with networking disabled checked this
path with a live Prime call.

Limits are 600 calls, 12,000 UTF-8 input bytes and 32 output tokens per call, with
one user prompt and optional system prompt. The relay accepts plain, non-streaming text chat and rejects tool and structured-output requests.
Calls are reserved in `judge-usage.jsonl` before execution; restarting the relay
with the same log retains the limit. An exclusive file lock prevents two relay
processes from spending the same budget. Failed calls also consume the limit.
`--judge` defaults to ten examples and rejects larger samples. Ten tasks with the
five standard formats and the two `flawed-judge` calls per response fit the cap.
Extra formats or other judges can reach the cap; that is an incomplete audit, not
an environment verdict. Check the model price before another billable run. At the
2026-11-21 prices ($0.165/$0.17 per million input/output tokens), the full cap costs
less than $1.21 even at one input token per byte, with room for chat framing.

## Interpretation

Compare probes with the five most common constant answers. A passing shotgun
response can contain the correct answer by chance. A judge reward is not proof
that a response is correct: inspect independent correctness metrics and check
examples by hand. Do not infer general robustness from fixed probes or a small
sample. Keep unconfirmed findings in the draft report. Do not send bug reports or
publish results without the owner's instruction.
