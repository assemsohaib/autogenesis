"""Score v1 taskset probes in-process, without a rollout model or remote runtime.

Runs inside the audit container (the taskset is third-party code). For each task, it
builds a Trace with the task prompt and one sampled assistant reply per planned probe,
then calls `task.score(trace, runtime)`. The runtime is verifiers' local
`SubprocessRuntime`, so a reward that runs a script (for example `run_uv_script`) runs it
in this same container, the audit sandbox. Optional LLM judges use the host relay;
there is no rollout model, remote runtime, or upload.

Usage: python v1_score.py TASKSET_ID PLAN_JSON ROWS_JSONL LOG_JSONL --num-tasks N --rollouts R
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from probe_server import content_text, prompt_key

SCORE_TIMEOUT = 120.0  # the first call may install a uv script's dependencies
SETUP_TIMEOUT = 300.0


def use_installed_uv() -> None:
    """Avoid a redundant network install in the local runtime's script bootstrap.

    The Docker image supplies uv. Keep verifiers' original installer as a fallback;
    script dependency resolution and the reward scripts themselves are unchanged.
    """
    from verifiers.v1.runtimes import base

    if hasattr(base, "_ENSURE_UV") and not base._ENSURE_UV.startswith("command -v uv"):
        base._ENSURE_UV = "command -v uv >/dev/null 2>&1 || { " + base._ENSURE_UV + "; }"


def configure_judges(config, override: dict):
    """Change judge transport while keeping the task's grading prompt and fields."""
    task = config.task
    updates = {}
    count = 0

    def configure(judge):
        nonlocal count
        if not all(hasattr(judge, field) for field in ("model", "base_url", "api_key_var")):
            return judge
        count += 1
        return type(judge).model_validate({**judge.model_dump(), **override})

    if hasattr(task, "judge"):
        updates["judge"] = configure(task.judge)
    if getattr(task, "judges", None):
        updates["judges"] = [configure(judge) for judge in task.judges]
    if not count:
        raise ValueError("no compatible model judge found in task.judge or task.judges")
    return config.model_copy(update={"task": task.model_copy(update=updates)})


def _messages(data) -> list[dict]:
    """The task's prompt messages as plain dicts: system (if any), then the prompt."""
    out: list[dict] = []
    system = getattr(data, "system_prompt", None)
    if system:
        out.append({"role": "system", "content": content_text(system) or str(system)})
    prompt = getattr(data, "prompt", None)
    if isinstance(prompt, str):
        out.append({"role": "user", "content": prompt})
    elif isinstance(prompt, list):
        for m in prompt:
            m = m.model_dump() if hasattr(m, "model_dump") else dict(m)
            out.append({"role": m.get("role", "user"), "content": content_text(m.get("content"))})
    return out


def _trace(vf, task, msgs: list[dict], reply: str | None):
    """A trace with the prompt messages and, if `reply` is given, one sampled reply."""
    from verifiers.v1.state import state_cls
    from verifiers.v1.trace import Trace, TraceTask

    kinds = {"system": vf.SystemMessage, "user": vf.UserMessage, "assistant": vf.AssistantMessage}
    trace = Trace(
        task=TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
        state=state_cls(type(task))(),
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="probe", trainable=False),
    )
    nodes = [kinds.get(m["role"], vf.UserMessage)(content=m["content"]) for m in msgs]
    if reply is not None:
        nodes.append(vf.AssistantMessage(content=reply))
    for i, message in enumerate(nodes):
        sampled = reply is not None and i == len(nodes) - 1
        trace.nodes.append(
            vf.MessageNode(parent=i - 1 if i else None, message=message, sampled=sampled)
        )
    if reply is not None:
        trace.is_completed = True
        trace.ok = True
    return trace


async def _setup(task, trace, runtime) -> None:
    """Run the task's setup hook. verifiers 0.3.1 calls `setup(trace, runtime)`; some
    tasksets override it as `setup(runtime)`."""
    if "trace" in inspect.signature(task.setup).parameters:
        await task.setup(trace, runtime)
    else:
        await task.setup(runtime)


async def _score(
    vf,
    task,
    msgs,
    reply: str,
    runtime,
    *,
    details: dict | None = None,
) -> tuple[float, str | None, list]:
    task = copy.deepcopy(task)
    trace = _trace(vf, task, msgs, None)
    try:
        await asyncio.wait_for(_setup(task, trace, runtime), SETUP_TIMEOUT)
        trace.nodes.append(
            vf.MessageNode(
                parent=len(trace.nodes) - 1 if trace.nodes else None,
                message=vf.AssistantMessage(content=reply),
                sampled=True,
            )
        )
        trace.is_completed = True
        trace.ok = True
        if hasattr(task, "finalize"):
            await asyncio.wait_for(task.finalize(trace, runtime), SETUP_TIMEOUT)
        await asyncio.wait_for(task.score(trace, runtime), SCORE_TIMEOUT)
        if not any(value is not None for value in trace.rewards.values()):
            raise RuntimeError("task scored no reward")
        if details is not None:
            details["metrics"] = dict(trace.metrics)
    except Exception as exc:  # a scoring failure is a result, not a crash
        return 0.0, f"{type(exc).__name__}: {exc}"[:300], []
    return float(trace.reward), None, sorted(trace.rewards)


async def _score_replies(vf, task, msgs, replies):
    try:
        from verifiers.v1.configs.runtime import SubprocessConfig
    except ImportError:  # verifiers 0.3.x and 0.4.0 keep configs beside the runtime
        from verifiers.v1.runtimes.subprocess import SubprocessConfig
    from verifiers.v1.runtimes import provision_runtime

    async def one(reply):
        details = {}
        # Each probe gets a fresh workspace as well as a fresh task and trace.
        async with provision_runtime(SubprocessConfig()) as runtime:
            result = await _score(vf, task, msgs, reply, runtime, details=details)
        return reply, result, details

    records = []
    for start in range(0, len(replies), 2):
        batch = await asyncio.gather(*(one(reply) for reply in replies[start : start + 2]))
        records.extend(batch)
        if any(result[1] for _reply, result, _details in batch):
            break
    return records


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("taskset")
    parser.add_argument("plan")
    parser.add_argument("rows")
    parser.add_argument("log")
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--rollouts", type=int, required=True)
    parser.add_argument("--taskset-config")
    args = parser.parse_args()

    import verifiers.v1 as vf
    from verifiers.v1.utils.loaders import taskset_class, taskset_config_type

    use_installed_uv()

    settings = json.loads(Path(args.taskset_config).read_text()) if args.taskset_config else {}
    judge_override = settings.pop("_audit_judge", None)
    config = taskset_config_type(args.taskset).model_validate({**settings, "id": args.taskset})
    if judge_override:
        config = configure_judges(config, judge_override)
    taskset = taskset_class(args.taskset)(config)
    plan = json.loads(Path(args.plan).read_text())
    default = plan.get("default", "I am not sure.")
    by_key: dict = plan.get("by_key", {})

    scored: set[str] = set()
    rows, log = [], []
    n = 0
    for task in taskset:
        if n >= args.num_tasks:
            break
        n += 1
        msgs = _messages(task.data)
        user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        key = prompt_key(user)
        planned = [text for _, text in by_key.get(key, [])]
        replies = (planned + [default] * args.rollouts)[: args.rollouts]
        data = task.data.model_dump()
        records = await _score_replies(vf, task, msgs, replies)
        for reply, (reward, error, names), detail in records:
            scored.update(names)
            log.append({"key": key, "system": system, "user": user})
            rows.append({"key": key, "user": user, "answer": data.get("answer"),
                         "info": data.get("info"), "completion": reply,
                         "reward": reward, "error": error, **detail})  # fmt: skip
        if any(result[1] for _reply, result, _detail in records):
            break
    Path(args.rows).write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    Path(args.log).write_text("".join(json.dumps(r) + "\n" for r in log))
    if rows and not scored:
        raise RuntimeError("no reward was scored: " + str(rows[0]["error"]))
    print(json.dumps({"tasks": n, "scored_rewards": sorted(scored)}))


if __name__ == "__main__":
    asyncio.run(main())
