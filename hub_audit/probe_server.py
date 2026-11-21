"""An OpenAI-compatible chat server that answers with planned probe responses.

Runs inside the audit container. The plan is a JSON file:
    {"default": "<text>", "by_key": {"<key>": [["<probe name>", "<text>"], ...]}}
The key of a request is `prompt_key(last user message)`. The i-th request with a key
gets the i-th planned response (cycling). Every answer is logged as one JSON line
`{"key", "probe", "text"}`, so results can be joined back to probes.

Run: python probe_server.py <port> <plan.json> <log.jsonl>
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

from aiohttp import web


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return content_text(message.get("content"))
    return ""


def prompt_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


class Planner:
    def __init__(self, plan: dict, log_path: str) -> None:
        self.default = plan.get("default", "I am not sure.")
        self.by_key: dict[str, list[list[str]]] = plan.get("by_key", {})
        self.counts: dict[str, int] = {}
        self.log = open(log_path, "a")  # noqa: SIM115

    def next(self, messages: list[dict]) -> str:
        user = last_user_text(messages)
        key = prompt_key(user)
        # Match the original task, even after the environment adds feedback. The
        # previous sampled response identifies this rollout without shared counters
        # for its later turns (other rollouts may run concurrently).
        start = len(messages)
        for j, message in enumerate(messages):
            if message.get("role") == "user":
                candidate = content_text(message.get("content"))
                if prompt_key(candidate) in self.by_key:
                    user, key, start = candidate, prompt_key(candidate), j + 1
                    break
        planned = self.by_key.get(key)
        if planned:
            previous = next(
                (
                    content_text(m.get("content"))
                    for m in messages[start:]
                    if m.get("role") == "assistant"
                ),
                None,
            )
            selected = next(((n, t) for n, t in planned if t == previous), None)
            if previous is not None and selected is None:
                raise ValueError("continued rollout has no matching planned response")
            if selected is None:
                i = self.counts.get(key, 0)
                self.counts[key] = i + 1
                selected = planned[i % len(planned)]
            name, text = selected
        else:
            name, text = "default", self.default
        turn = sum(m.get("role") == "assistant" for m in messages[start:]) + 1
        record = {"key": key, "probe": name, "text": text, "turn": turn}
        if not planned:
            record["user"] = user
            record["system"] = next(
                (content_text(m.get("content")) for m in messages if m.get("role") == "system"),
                "",
            )
        self.log.write(json.dumps(record) + "\n")
        self.log.flush()
        return text


def _usage(text: str) -> dict:
    n = max(1, len(text) // 4)
    return {"prompt_tokens": 10, "completion_tokens": n, "total_tokens": 10 + n}


def make_app(planner: Planner) -> web.Application:
    async def chat(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        text = planner.next(body.get("messages", []))
        base = {"id": "chatcmpl-probe", "created": int(time.time()), "model": body.get("model")}
        if not body.get("stream"):
            choice = {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
            return web.json_response(
                {**base, "object": "chat.completion", "choices": [choice], "usage": _usage(text)}
            )
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for choice in (
            {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None},
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        ):
            event = {**base, "object": "chat.completion.chunk", "choices": [choice]}
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
        final = {**base, "object": "chat.completion.chunk", "choices": [], "usage": _usage(text)}
        await response.write(f"data: {json.dumps(final)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def models(_request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": "probe", "object": "model"}]})

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/models", models)
    return app


if __name__ == "__main__":
    port, plan_path, log_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    with open(plan_path) as f:
        plan = json.load(f)
    web.run_app(make_app(Planner(plan, log_path)), host="127.0.0.1", port=port, print=None)
