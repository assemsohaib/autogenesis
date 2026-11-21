"""Sandbox runner for builder code. Standard library only; Python >= 3.10.

The host writes this file and a request file into the sandbox runtime, then runs
`python3 -I runner.py <request.json>`. The runner deletes the request file, limits its
own resources, loads the builder module with a restricted set of builtins, runs one
operation, and prints a single result line prefixed with the request's nonce.

The runner is deliberately dumb: it only executes `generate` / `solve` / `verify` and
reports raw values. Every scoring decision happens on the host (see `audit.py`).

Operations:
  generate  {"seeds": [int, ...]}
            -> {"records": [{"seed", "task", "answer", "deterministic"} | {"seed", "error"}]}
  verify    {"tasks": [dict, ...], "items": [[task_index, response], ...]}
            -> {"scores": [[score | None, error | None], ...]}
A load failure of the module itself returns {"fatal": "..."}.
"""

import builtins
import json
import math
import os
import signal
import sys

ALLOWED = frozenset(
    {
        "__future__",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "textwrap",
        "typing",
        "unicodedata",
    }
)
BANNED = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "open",
        "quit",
        "setattr",
        "vars",
    }
)
MAX_TASK_CHARS = 20_000
MAX_ANSWER_CHARS = 4_000
MAX_ERROR_CHARS = 300

_real_import = builtins.__import__


class CallTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise CallTimeout("call exceeded its time limit")


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED:
        raise ImportError(f"import of {name!r} is not allowed")
    return _real_import(name, globals, locals, fromlist, level)


def _limit_resources(memory_mb, cpu_seconds):
    try:
        import resource
    except ImportError:
        return
    limits = [
        (resource.RLIMIT_AS, memory_mb * 2**20),
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_FSIZE, 0),
        (resource.RLIMIT_CORE, 0),
    ]
    for kind, value in limits:
        try:
            resource.setrlimit(kind, (value, value))
        except (ValueError, OSError):
            pass


def _describe(error):
    text = f"{type(error).__name__}: {error}"
    return text[:MAX_ERROR_CHARS]


def _call(fn, args, timeout):
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _load(code, timeout):
    safe_builtins = {k: v for k, v in vars(builtins).items() if k not in BANNED}
    safe_builtins["__import__"] = _safe_import
    namespace = {"__name__": "builder_env", "__builtins__": safe_builtins}
    _call(exec, (compile(code, "builder_env.py", "exec"), namespace), timeout)
    functions = {}
    for name in ("generate", "solve", "verify"):
        fn = namespace.get(name)
        if not callable(fn):
            raise TypeError(f"{name}() is not defined")
        functions[name] = fn
    return functions


def _op_generate(fns, request, timeout):
    records = []
    for seed in request["seeds"]:
        try:
            first = _call(fns["generate"], (seed,), timeout)
            second = _call(fns["generate"], (seed,), timeout)
            if not isinstance(first, dict):
                raise TypeError("generate() must return a dict")
            prompt = first.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise TypeError("generate() must return a non-empty str 'prompt'")
            encoded = json.dumps(first, sort_keys=True)
            if len(encoded) > MAX_TASK_CHARS:
                raise ValueError(f"task is over {MAX_TASK_CHARS} chars as JSON")
            try:
                deterministic = encoded == json.dumps(second, sort_keys=True)
            except (TypeError, ValueError):
                deterministic = False
            answer = _call(fns["solve"], (json.loads(encoded),), timeout)
            if not isinstance(answer, str):
                raise TypeError("solve() must return a str")
            if not answer.strip():
                raise ValueError("solve() returned an empty answer")
            if len(answer) > MAX_ANSWER_CHARS:
                raise ValueError(f"solve() answer is over {MAX_ANSWER_CHARS} chars")
            records.append(
                {
                    "seed": seed,
                    "task": json.loads(encoded),
                    "answer": answer,
                    "deterministic": deterministic,
                }
            )
        except BaseException as e:  # builder code may raise anything
            if isinstance(e, KeyboardInterrupt):
                raise
            records.append({"seed": seed, "error": _describe(e)})
    return {"records": records}


def _coerce_score(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0:
        return float(value)
    raise ValueError(f"verify() must return a float in [0, 1], got {value!r}"[:MAX_ERROR_CHARS])


def _op_verify(fns, request, timeout):
    encoded = [json.dumps(task, sort_keys=True) for task in request["tasks"]]
    scores = []
    for index, response in request["items"]:
        try:
            value = _call(fns["verify"], (json.loads(encoded[index]), response), timeout)
            scores.append([_coerce_score(value), None])
        except BaseException as e:  # builder code may raise anything
            if isinstance(e, KeyboardInterrupt):
                raise
            scores.append([None, _describe(e)])
    return {"scores": scores}


OPS = {"generate": _op_generate, "verify": _op_verify}


def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        request = json.loads(f.read())
    os.remove(path)
    nonce = request.pop("nonce")
    limits = request.get("limits", {})
    timeout = float(limits.get("call_timeout", 2.0))

    out = os.fdopen(os.dup(1), "w")
    sys.stdout = sys.stderr = open(os.devnull, "w")  # noqa: SIM115 - lives for the process
    os.environ.clear()
    signal.signal(signal.SIGALRM, _on_alarm)
    _limit_resources(int(limits.get("memory_mb", 1024)), int(limits.get("cpu_seconds", 60)))
    sys.setrecursionlimit(5000)

    try:
        functions = _load(request["code"], timeout)
    except BaseException as e:  # builder code may raise anything
        result = {"fatal": "module load failed: " + _describe(e)}
    else:
        result = OPS[request["op"]](functions, request, timeout)
    out.write(nonce + json.dumps(result) + "\n")
    out.flush()


if __name__ == "__main__":
    main()
