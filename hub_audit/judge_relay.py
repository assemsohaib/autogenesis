"""Host-only OpenAI chat relay to Prime CLI. Never mount Prime credentials in Docker.

Fixed transport limits bound spend: at most 600 calls, 12,000 input bytes and 32
output tokens per call. At Mistral Nemo's checked prices ($0.165/$0.17 per million
input/output tokens, 2026-11-21), even one input token per byte costs under $1.21.
The model and token limit cannot be selected by untrusted environment code.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import secrets
import socketserver
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = "mistralai/mistral-nemo"
MAX_CALLS = 600
MAX_INPUT_BYTES = 12_000
MAX_OUTPUT_TOKENS = 32


class PrimeJudgeRelay:
    def __init__(self, log: Path, *, executable: str = "prime", max_calls: int = MAX_CALLS):
        if not 0 < max_calls <= MAX_CALLS:
            raise ValueError(f"max_calls must be between 1 and {MAX_CALLS}")
        self.log = log
        self.executable = executable
        self.max_calls = max_calls
        self._budget_lock = log.with_suffix(log.suffix + ".lock").open("a")
        try:
            fcntl.flock(self._budget_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._budget_lock.close()
            raise RuntimeError("judge budget log is already in use") from None
        self.calls = (
            max(
                (json.loads(line)["call"] for line in log.read_text().splitlines()),
                default=0,
            )
            if log.exists()
            else 0
        )
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(2)
        self.token = secrets.token_urlsafe(32)

    def close(self):
        self._budget_lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def complete(self, body: dict) -> dict:
        messages = body.get("messages", [])
        # Prime chat takes one user prompt and an optional system prompt. Reject
        # unsupported histories rather than change what the environment's judge sees.
        roles = [m.get("role") for m in messages]
        if roles not in (["user"], ["system", "user"]):
            raise ValueError("judge relay requires one user and optional system message")
        if any(not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("judge relay requires text messages")
        if sum(len(m["content"].encode()) for m in messages) > MAX_INPUT_BYTES:
            raise ValueError("judge request exceeds input size limit")
        if body.get("stream") or body.get("tools") or body.get("response_format"):
            raise ValueError("judge relay supports plain non-streaming chat only")
        with self.lock:
            if self.calls >= self.max_calls:
                raise RuntimeError("judge call limit reached")
            self.calls += 1  # reserve before the call, including failed requests
            with self.log.open("a") as stream:
                stream.write(json.dumps({"call": self.calls, "event": "reserved"}) + "\n")
            call = self.calls
        with self.slots:
            return self._complete(messages, roles, call)

    def _complete(self, messages: list[dict], roles: list[str], call: int) -> dict:
        cmd = [
            self.executable,
            "inference",
            "chat",
            MODEL,
            "--max-tokens",
            str(MAX_OUTPUT_TOKENS),
            "--temperature",
            "0",
            "--output",
            "json",
            "--plain",
        ]
        if roles[0] == "system":
            cmd += ["--system=" + messages[0]["content"]]
        done = subprocess.run(
            cmd,
            input=messages[-1]["content"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        record = {"call": call, "model": MODEL, "exit_code": done.returncode}
        try:
            if done.returncode:
                raise RuntimeError(f"Prime CLI failed (exit {done.returncode})")
            # Some CLI versions print a progress line before the JSON result.
            result = json.loads(done.stdout[done.stdout.index("{") :])
            text = result["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise RuntimeError("Prime CLI returned no judge text")
            record["usage"] = result.get("usage")
            return result
        finally:
            with self.log.open("a") as stream:
                stream.write(json.dumps(record) + "\n")


@contextlib.contextmanager
def serve(relay: PrimeJudgeRelay, *, socket_path: Path | None = None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            status = 200
            try:
                if self.path != "/v1/chat/completions":
                    raise ValueError("unknown endpoint")
                if self.headers.get("Authorization") != f"Bearer {relay.token}":
                    self.send_error(403)
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 100_000:
                    raise ValueError("request body size limit")
                payload = relay.complete(json.loads(self.rfile.read(size)))
            except Exception as exc:
                status = 502
                payload = {"error": {"message": str(exc), "type": "judge_relay_error"}}
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    if socket_path is None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        url = f"http://127.0.0.1:{server.server_port}/v1"
    else:

        class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

        server = UnixServer(str(socket_path), Handler)
        socket_path.chmod(0o666)  # the non-root container user connects via its bind mount
        url = "http://127.0.0.1/v1"  # replaced with the container-local forwarder's port
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        if socket_path is not None:
            socket_path.unlink(missing_ok=True)
