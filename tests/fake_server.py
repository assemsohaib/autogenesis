"""A tiny OpenAI-compatible chat server for offline end-to-end runs.

Routes by model id:
  fake-builder  replies with the good weighted-sum environment (fixtures.GOOD_REPLY)
  fake-strong   parses the weighted sum from the prompt and answers correctly
  fake-weak     always answers `ANSWER: 0`
  fake-broken   always fails with HTTP 404 (like a model with no provider)
  fake-flaky    answers like fake-weak, but fails every second request with HTTP 404

Run: python tests/fake_server.py <port>
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent))
from fixtures import GOOD_REPLY

_TERM = re.compile(r"(\d+)\*\((-?\d+)\)")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def answer(model: str, messages: list[dict]) -> str:
    prompt = _text(messages[-1].get("content")) if messages else ""
    if model == "fake-builder":
        return GOOD_REPLY
    if model == "fake-strong":
        total = sum(int(w) * int(n) for w, n in _TERM.findall(prompt))
        return f"Adding the weighted terms gives {total}.\nANSWER: {total}"
    return "ANSWER: 0"


def _usage(text: str) -> dict:
    return {"prompt_tokens": 10, "completion_tokens": max(1, len(text) // 4), "total_tokens": 10}


_calls: dict[str, int] = {}


def _not_found(model: str) -> web.Response:
    error = {"message": f"no provider for {model}", "type": "not_found_error", "code": "not_found"}
    return web.json_response({"error": error}, status=404)


async def chat(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    model = body.get("model", "")
    _calls[model] = _calls.get(model, 0) + 1
    if model == "fake-broken" or (model == "fake-flaky" and _calls[model] % 2 == 0):
        return _not_found(model)
    text = answer(model, body.get("messages", []))
    base = {"id": "chatcmpl-fake", "created": int(time.time()), "model": model}
    if not body.get("stream"):
        return web.json_response(
            {
                **base,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(text),
            }
        )
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    chunks = [
        {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None},
        {"index": 0, "delta": {}, "finish_reason": "stop"},
    ]
    for choice in chunks:
        event = {**base, "object": "chat.completion.chunk", "choices": [choice]}
        await response.write(f"data: {json.dumps(event)}\n\n".encode())
    final = {**base, "object": "chat.completion.chunk", "choices": [], "usage": _usage(text)}
    await response.write(f"data: {json.dumps(final)}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


async def models(_request: web.Request) -> web.Response:
    ids = ["fake-builder", "fake-weak", "fake-strong"]
    data = [{"id": i, "object": "model"} for i in ids]
    return web.json_response({"object": "list", "data": data})


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="127.0.0.1", port=int(sys.argv[1]), print=None)
