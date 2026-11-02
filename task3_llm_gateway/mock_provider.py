"""Fake OpenAI-style provider that streams chat.completion.chunk events over SSE.

Behaviour is driven by app.state.script so tests can control chunking, delays and failures:
  deltas   list of content strings (one choice)
  choices  list of delta-lists (several choices, interleaved)
  delay    seconds between content frames
  split    if set, write each SSE frame to the socket in pieces of this many bytes
  status   if set, return this error status with a fake stack trace in the body
"""
from __future__ import annotations

import asyncio
import json

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

DEFAULT_DELTAS = [
    "Sure! ", "You can reach ", "Dana at ", "dana.", "ross", "@exam", "ple.com", ". Her SSN is ", "123-4", "5-6789",
    " and her card ", "4111 ", "1111 ", "1111 ", "1111", ". Anything else?",
]


def _chunk(idx: int, delta: dict, finish: str | None = None) -> dict:
    return {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 1, "model": "mock-1",
            "choices": [{"index": idx, "delta": delta, "finish_reason": finish}]}


def _frame(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"


async def completions(request: Request):
    st = request.app.state
    body = await request.json()
    st.requests.append({"body": body, "headers": dict(request.headers)})
    sc = st.script

    if sc.get("status"):
        return JSONResponse(
            {"error": {"message": "Traceback (most recent call last): File '/srv/provider/internal.py', line 88"}},
            status_code=sc["status"],
        )

    choices = sc.get("choices") or [sc.get("deltas", DEFAULT_DELTAS)]

    if not body.get("stream"):
        return JSONResponse({
            "id": "chatcmpl-mock", "object": "chat.completion", "model": "mock-1",
            "choices": [{"index": i, "message": {"role": "assistant", "content": "".join(d)}, "finish_reason": "stop"}
                        for i, d in enumerate(choices)],
        })

    delay, split = sc.get("delay", 0), sc.get("split")

    async def gen():
        for i in range(len(choices)):
            yield _frame(_chunk(i, {"role": "assistant", "content": ""}))
        for step in range(max(len(d) for d in choices)):
            for i, ds in enumerate(choices):
                if step < len(ds):
                    data = _frame(_chunk(i, {"content": ds[step]}))
                    if split:
                        for k in range(0, len(data), split):
                            yield data[k:k + split]
                            await asyncio.sleep(0)
                    else:
                        yield data
            if delay:
                await asyncio.sleep(delay)
        for i in range(len(choices)):
            yield _frame(_chunk(i, {}, "stop"))
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def create_mock_app() -> Starlette:
    app = Starlette(routes=[Route("/v1/chat/completions", completions, methods=["POST"])])
    app.state.script = {}
    app.state.requests = []
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_mock_app(), host="127.0.0.1", port=9100, log_level="info")
