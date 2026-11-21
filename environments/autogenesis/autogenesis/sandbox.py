"""Host side of the sandbox: ship builder code to a `vf.Runtime` and read raw results.

The runtime type is a config choice (`--env.sandbox.type subprocess|docker|prime`).
`subprocess` is for development only: the runner limits its own resources and the
builder code passes the static policy, but it still runs on the host. Use `docker` or
`prime` for training runs.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Protocol

from verifiers.v1.runtimes import Runtime

RUNNER_SOURCE = (Path(__file__).parent / "runner.py").read_bytes()
RUNNER_PATH = "autogenesis_runner.py"


class Executor(Protocol):
    """Runs one runner operation on builder code and returns the runner's result
    dict, or `{"fatal": "..."}` when the sandbox itself failed."""

    def __call__(self, op: str, *, code: str, **payload: Any) -> Awaitable[dict]: ...


class SandboxExecutor:
    def __init__(
        self,
        runtime: Runtime,
        *,
        python: str = "python3",
        call_timeout: float = 2.0,
        sandbox_timeout: float = 120.0,
        memory_mb: int = 1024,
        cpu_seconds: int = 60,
    ) -> None:
        self.runtime = runtime
        self.python = python
        self.sandbox_timeout = sandbox_timeout
        self.limits = {
            "call_timeout": call_timeout,
            "memory_mb": memory_mb,
            "cpu_seconds": cpu_seconds,
        }
        self._ready = False

    async def _setup(self) -> None:
        if not self._ready:
            await self.runtime.write(RUNNER_PATH, RUNNER_SOURCE)
            self._ready = True

    async def __call__(self, op: str, *, code: str, **payload: Any) -> dict:
        await self._setup()
        nonce = secrets.token_hex(16)
        request_path = f"autogenesis_request_{nonce[:12]}.json"
        request = {"nonce": nonce, "op": op, "code": code, "limits": self.limits, **payload}
        await self.runtime.write(request_path, json.dumps(request).encode())
        try:
            result = await asyncio.wait_for(
                self.runtime.run([self.python, "-I", RUNNER_PATH, request_path], env={}),
                timeout=self.sandbox_timeout,
            )
        except TimeoutError:
            return {"fatal": f"sandbox exceeded {self.sandbox_timeout:g}s"}
        except Exception as exc:  # a dead or misconfigured runtime is a sandbox failure
            return {"fatal": f"sandbox failed to start: {type(exc).__name__}: {exc}"[:400]}
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(nonce):
                try:
                    return json.loads(line[len(nonce) :])
                except json.JSONDecodeError:
                    return {"fatal": "sandbox returned malformed result"}
        detail = (result.stderr or result.stdout).strip()[-300:]
        return {
            "fatal": f"sandbox returned no result (exit code {result.exit_code})"
            + (f": {detail}" if detail else "")
        }
