"""Integration tests. They start the mock provider and the gateway as real HTTP servers
(httpx's in-process ASGI transport buffers whole responses, which would hide streaming bugs)."""
import asyncio
import contextlib
import json
import time

import httpx
import pytest
import uvicorn

from gateway import EventProcessor, create_app
from mock_provider import create_mock_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@contextlib.asynccontextmanager
async def serve(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="off"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


class Stack:
    def __init__(self, mock, gw_url):
        self.mock, self.url = mock, gw_url


@pytest.fixture
async def stack():
    mock = create_mock_app()
    async with serve(mock) as mock_url:
        app = create_app(f"{mock_url}/v1/chat/completions", api_key="provider-secret")
        async with serve(app) as gw_url:
            yield Stack(mock, gw_url + "/v1/chat/completions")
        await app.state.client.aclose()


BODY = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True}


async def read_sse(url, body=BODY):
    """Returns (events, raw_text, times). events = parsed JSON dicts or the string [DONE]."""
    events, raw, times = [], [], []
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=20) as c:
        async with c.stream("POST", url, json=body) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            async for line in r.aiter_lines():
                raw.append(line)
                if line.startswith("data: "):
                    d = line[6:]
                    events.append("[DONE]" if d == "[DONE]" else json.loads(d))
                    times.append(time.perf_counter() - t0)
    return events, "\n".join(raw), times


def text_of(events, idx=0):
    return "".join(
        ch["delta"].get("content") or ""
        for e in events if isinstance(e, dict)
        for ch in e.get("choices", []) if ch["index"] == idx
    )


# ------------------------------------------------------------------ redaction end to end
async def test_pii_split_across_chunks_is_redacted(stack):
    stack.mock.state.script = {"deltas": [
        "Reach me at jo", "hn.doe", "@exa", "mple.com or ", "SSN 123-4", "5-6789 card 4111 11", "11 1111 1111", ". Bye",
    ]}
    events, raw, _ = await read_sse(stack.url)
    assert text_of(events) == "Reach me at [REDACTED] or SSN [REDACTED] card [REDACTED]. Bye"
    for frag in ("john", "doe", "example", "123-4", "6789", "4111", "1111"):
        assert frag not in raw  # nothing leaked on the wire, not even a fragment
    assert events[-1] == "[DONE]"


async def test_frames_split_mid_line_on_the_wire_still_work(stack):
    stack.mock.state.script = {"split": 5}
    events, raw, _ = await read_sse(stack.url)
    t = text_of(events)
    assert "dana.ross" not in raw and "123-45" not in raw and "4111" not in raw
    assert t.count("[REDACTED]") == 3 and t.endswith("Anything else?")


async def test_held_tail_is_released_before_finish_chunk(stack):
    stack.mock.state.script = {"deltas": ["Email ", "bob@example.com"]}  # PII is the very last thing
    events, _, _ = await read_sse(stack.url)
    dicts = [e for e in events if isinstance(e, dict)]
    assert text_of(events) == "Email [REDACTED]"
    finish_at = next(i for i, d in enumerate(dicts) if d["choices"][0]["finish_reason"] == "stop")
    # every piece of content arrived before the chunk that carries finish_reason
    assert not any(d["choices"][0]["delta"].get("content") for d in dicts[finish_at:])
    assert events[-1] == "[DONE]"


async def test_role_chunk_and_structure_preserved(stack):
    events, _, _ = await read_sse(stack.url)
    first = events[0]
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert first["object"] == "chat.completion.chunk" and first["id"] == "chatcmpl-mock"


async def test_multiple_choices_are_tracked_separately(stack):
    stack.mock.state.script = {"choices": [
        ["a: x@y", ".com done"],
        ["b: 123-45", "-6789 done"],
    ]}
    body = {**BODY, "n": 2}
    events, raw, _ = await read_sse(stack.url, body)
    assert text_of(events, 0) == "a: [REDACTED] done"
    assert text_of(events, 1) == "b: [REDACTED] done"
    assert "x@y" not in raw and "6789" not in raw


# ------------------------------------------------------------------ latency / streaming behaviour
async def test_first_content_arrives_long_before_upstream_finishes(stack):
    stack.mock.state.script = {"deltas": ["Hello ", "wide ", "world ", "of ", "tokens "], "delay": 0.3}
    events, _, times = await read_sse(stack.url)
    first_content = next(t for e, t in zip(events, times) if isinstance(e, dict) and text_of([e]))
    assert first_content < 0.2, f"first token took {first_content:.2f}s (gateway is buffering?)"
    assert times[-1] > 1.0  # and the upstream really was slow, so the above means something


async def test_token_gaps_are_preserved_not_batched(stack):
    stack.mock.state.script = {"deltas": ["one ", "two ", "three ", "four "], "delay": 0.25}
    events, _, times = await read_sse(stack.url)
    content_times = [t for e, t in zip(events, times) if isinstance(e, dict) and text_of([e])]
    assert len(content_times) >= 4
    assert content_times[-1] - content_times[0] > 0.5  # spread out over time, not delivered in one lump


# ------------------------------------------------------------------ non-streaming and errors
async def test_non_streaming_response_is_redacted_too(stack):
    async with httpx.AsyncClient() as c:
        r = await c.post(stack.url, json={**BODY, "stream": False})
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content.count("[REDACTED]") == 3 and "@" not in content and "4111" not in content


async def test_provider_key_added_and_client_auth_not_forwarded(stack):
    async with httpx.AsyncClient() as c:
        await c.post(stack.url, json={**BODY, "stream": False}, headers={"Authorization": "Bearer client-token"})
    sent = stack.mock.state.requests[-1]["headers"]
    assert sent["authorization"] == "Bearer provider-secret"


@pytest.mark.parametrize("status,expected_status,etype", [(500, 502, "upstream_error"), (429, 429, "upstream_rate_limited"), (404, 502, "upstream_error")])
async def test_upstream_errors_are_sanitised(stack, status, expected_status, etype):
    stack.mock.state.script = {"status": status}
    async with httpx.AsyncClient() as c:
        r = await c.post(stack.url, json=BODY)
    assert r.status_code == expected_status
    assert r.json()["error"]["type"] == etype
    assert "Traceback" not in r.text and "internal.py" not in r.text


async def test_upstream_down_is_sanitised():
    app = create_app("http://127.0.0.1:1/v1/chat/completions")
    async with serve(app) as url:
        async with httpx.AsyncClient() as c:
            r = await c.post(url + "/v1/chat/completions", json=BODY)
    assert r.status_code == 502
    assert r.json() == {"error": {"type": "upstream_unavailable", "message": "The model provider is unavailable"}}
    await app.state.client.aclose()


@pytest.mark.parametrize("payload", [b"not json", b"[]", b'{"model":"x"}', b'{"messages":"hi"}'])
async def test_bad_requests_get_400(stack, payload):
    async with httpx.AsyncClient() as c:
        r = await c.post(stack.url, content=payload, headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request"


# ------------------------------------------------------------------ event processor unit tests
def test_unparseable_event_fails_closed():
    from gateway import BadEvent
    with pytest.raises(BadEvent):
        EventProcessor().process(["{broken"], [])


def test_logprobs_are_stripped():
    p = EventProcessor()
    ev = {"choices": [{"index": 0, "delta": {"content": "hi "}, "logprobs": {"content": [{"token": "hi"}]}}]}
    frames, _ = p.process([json.dumps(ev)], [])
    assert b'"logprobs":null' in frames[0] and b'"token"' not in frames[0]


def test_comment_lines_pass_through():
    frames, done = EventProcessor().process([], [": keep-alive"])
    assert frames == [b": keep-alive\n\n"] and not done
