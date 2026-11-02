"""Task 3 - LLM gateway with a streaming PII guardrail.

POST /v1/chat/completions (OpenAI-compatible) -> upstream provider -> response streamed back
with emails / SSNs / card numbers replaced by [REDACTED].

Why it stays fast and small:
  * the upstream body is read line by line and every event is forwarded as soon as it is
    redacted, nothing is collected into a big string
  * each choice has a StreamRedactor that holds back at most a few hundred chars
  * the generator only pulls the next upstream line when the client has taken the last one,
    so a slow client slows the upstream read instead of filling memory

Run: UPSTREAM_URL=http://127.0.0.1:9100/v1/chat/completions python gateway.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from redactor import StreamRedactor, redact_text

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s [llm-gw] %(message)s")
log = logging.getLogger("llm-gw")

MAX_BODY_BYTES = 1_000_000


def gw_error(status: int, type_: str, message: str) -> JSONResponse:
    """Standard gateway error body. Never contains anything from the upstream."""
    return JSONResponse({"error": {"type": type_, "message": message}}, status_code=status)


class BadEvent(Exception):
    """Upstream sent something we can't safely inspect."""


class EventProcessor:
    """Turns upstream SSE events into redacted SSE events (pure, no I/O, easy to test)."""

    def __init__(self) -> None:
        self.reds: dict[int, StreamRedactor] = {}
        self.meta: dict[str, Any] = {}

    def _flush_chunks(self) -> list[bytes]:
        outs = []
        for idx, red in self.reds.items():
            tail = red.flush()
            if tail:
                outs.append(self._tail_frame({idx: tail}))
        return outs

    def _tail_frame(self, tails: dict[int, str]) -> bytes:
        choices = [{"index": i, "delta": {"content": t}, "finish_reason": None} for i, t in tails.items()]
        return self._encode([], {**self.meta, "choices": choices})

    @staticmethod
    def _encode(other_lines: list[str], obj: dict) -> bytes:
        prefix = "".join(line + "\n" for line in other_lines)
        return (prefix + "data: " + json.dumps(obj, separators=(",", ":")) + "\n\n").encode()

    def process(self, data_lines: list[str], other_lines: list[str]) -> tuple[list[bytes], bool]:
        """Returns (frames to send, stream_is_done)."""
        if not data_lines:  # comment / keep-alive / bare "event:" line: pass through
            return [("".join(l + "\n" for l in other_lines) + "\n").encode()], False

        data = "\n".join(data_lines)
        if data.strip() == "[DONE]":
            outs = self._flush_chunks()
            outs.append(("".join(l + "\n" for l in other_lines) + "data: [DONE]\n\n").encode())
            return outs, True

        try:
            obj = json.loads(data)
        except ValueError:
            raise BadEvent from None
        if not isinstance(obj, dict):
            raise BadEvent

        for k in ("id", "object", "created", "model"):
            if k in obj:
                self.meta[k] = obj[k]

        frames: list[bytes] = []
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            kept, tails = [], {}
            for choice in choices:
                if not isinstance(choice, dict):
                    raise BadEvent
                keep, tail = self._process_choice(choice)
                if tail:
                    tails[choice["index"]] = tail
                if keep:
                    kept.append(choice)
            if tails:  # held-back text goes out as its own chunk, ahead of the finish chunk
                frames.append(self._tail_frame(tails))
            if not kept:
                return frames, False  # chunk only carried text we are still holding back
            obj["choices"] = kept
        frames.append(self._encode(other_lines, obj))
        return frames, False

    def _process_choice(self, choice: dict) -> tuple[bool, str]:
        """Redacts one choice in place. Returns (keep_this_chunk, text_to_send_before_it)."""
        idx = choice.get("index", 0)
        if not isinstance(idx, int) or isinstance(idx, bool):
            idx = 0
        choice["index"] = idx
        red = self.reds.setdefault(idx, StreamRedactor())
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}
        # logprobs repeat the generated tokens, so they would leak what we are redacting
        if choice.get("logprobs") is not None:
            choice["logprobs"] = None

        content = delta.get("content")
        text = red.feed(content) if isinstance(content, str) else ""
        finishing = choice.get("finish_reason") is not None
        before = ""
        if finishing:
            before, text = text + red.flush(), ""  # whatever we hold must precede the finish chunk

        has_other = any(k != "content" for k in delta)
        if text or "content" in delta:
            delta["content"] = text
        choice["delta"] = delta
        return (bool(text) or has_other or finishing), before


async def redacted_stream(upstream: httpx.Response):
    proc = EventProcessor()
    data_lines: list[str] = []
    other_lines: list[str] = []
    done = False
    try:
        async for line in upstream.aiter_lines():
            if line != "":
                if line.startswith("data:"):
                    d = line[5:]
                    data_lines.append(d[1:] if d.startswith(" ") else d)
                else:
                    other_lines.append(line)
                continue
            if data_lines or other_lines:
                frames, done = proc.process(data_lines, other_lines)
                for f in frames:
                    yield f
            data_lines, other_lines = [], []
            if done:
                break
        if not done:  # upstream closed early: handle a half-finished event, then release the held tail
            if data_lines or other_lines:
                frames, _ = proc.process(data_lines, other_lines)
                for f in frames:
                    yield f
            for f in proc._flush_chunks():
                yield f
    except BadEvent:
        # Fail closed: we can't prove an event we can't parse is clean, so we end the stream.
        log.warning("unparseable upstream event, closing stream")
        yield b'data: {"error":{"type":"upstream_error","message":"Upstream response could not be processed"}}\n\n'
    except httpx.HTTPError as exc:
        log.warning("upstream stream failed: %s", type(exc).__name__)
        yield b'data: {"error":{"type":"upstream_error","message":"Upstream stream interrupted"}}\n\n'
    finally:
        await upstream.aclose()  # also runs when the client disconnects


async def read_limited(request: Request) -> bytes:
    chunks, total = [], 0
    async for c in request.stream():
        total += len(c)
        if total > MAX_BODY_BYTES:
            raise ValueError("too large")
        chunks.append(c)
    return b"".join(chunks)


async def chat_completions(request: Request) -> Response:
    st = request.app.state
    try:
        body = json.loads(await read_limited(request))
    except ValueError:
        return gw_error(400, "invalid_request", "Request body must be valid JSON under 1 MB")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return gw_error(400, "invalid_request", "Body must be a JSON object with a messages array")
    stream = body.get("stream") is True

    headers = {"content-type": "application/json"}
    if st.api_key:  # the gateway's own provider key; the client's Authorization is never forwarded
        headers["authorization"] = f"Bearer {st.api_key}"
    req = st.client.build_request("POST", st.upstream_url, content=json.dumps(body).encode(), headers=headers)
    try:
        upstream = await st.client.send(req, stream=True)
    except httpx.TimeoutException:
        return gw_error(504, "upstream_timeout", "The model provider timed out")
    except httpx.HTTPError as exc:
        log.warning("upstream unreachable: %s", type(exc).__name__)
        return gw_error(502, "upstream_unavailable", "The model provider is unavailable")

    if upstream.status_code != 200:
        log.warning("upstream returned %s", upstream.status_code)
        await upstream.aclose()
        if upstream.status_code == 429:
            return gw_error(429, "upstream_rate_limited", "The model provider is rate limiting requests")
        return gw_error(502, "upstream_error", "The model provider returned an error")

    if not stream:
        try:
            data = json.loads(await upstream.aread())
        except ValueError:
            return gw_error(502, "upstream_error", "The model provider returned an invalid response")
        finally:
            await upstream.aclose()
        for choice in data.get("choices", []) if isinstance(data, dict) else []:
            msg = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                msg["content"] = redact_text(msg["content"])
            if isinstance(choice, dict):
                choice["logprobs"] = None
        return JSONResponse(data)

    return StreamingResponse(
        redacted_stream(upstream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(upstream_url: str, api_key: str | None = None, client: httpx.AsyncClient | None = None) -> Starlette:
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=60, write=10, pool=5))
    app = Starlette(routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])])
    app.state.client = client
    app.state.upstream_url = upstream_url
    app.state.api_key = api_key
    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app(os.environ.get("UPSTREAM_URL", "http://127.0.0.1:9100/v1/chat/completions"),
                     os.environ.get("UPSTREAM_API_KEY"))
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8081")), log_level="info")
