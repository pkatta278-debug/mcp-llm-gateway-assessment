"""End-to-end: client -> gateway (real HTTP) -> primary / secondary mock providers (real HTTP)."""
import asyncio
import json
import os

import httpx
import pytest

from gateway import create_app, hash_key, parse_tenants
from helpers import serve
from mock_provider import create_mock_app
from rate_limiter import SlidingWindowRateLimiter
from router import ModelRouter, Provider

pytestmark = pytest.mark.anyio

KEY = "sk-tenant-secret-123"
BODY = {"model": "big", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}
AUTH = {"Authorization": f"Bearer {KEY}"}


class Stack:
    pass


@pytest.fixture
async def stack(tmp_path):
    s = Stack()
    s.primary, s.secondary = create_mock_app("primary"), create_mock_app("secondary")
    s.db = str(tmp_path / "rl.db")
    async with serve(s.primary) as pu, serve(s.secondary) as su:
        s.router = ModelRouter(
            [Provider("primary", f"{pu}/v1/chat/completions", timeout_s=0.4),
             Provider("secondary", f"{su}/v1/chat/completions", timeout_s=2)],
            SlidingWindowRateLimiter(s.db),
        )
        s.app = create_app(s.router, parse_tenants(f"{KEY}:50000,small-key:2000"))
        async with serve(s.app) as gw:
            s.url = gw + "/v1/chat/completions"
            yield s
        await s.router.aclose()


async def post(url, body=BODY, headers=AUTH, **kw):
    async with httpx.AsyncClient(timeout=15) as c:
        return await c.post(url, json=body, headers=headers, **kw)


def assert_error_shape(r, status, type_):
    assert r.status_code == status
    err = r.json()["error"]
    assert set(err) == {"type", "message", "request_id"} and err["type"] == type_
    assert err["request_id"].startswith("gw-") and r.headers["x-request-id"] == err["request_id"]
    for leak in ("Traceback", "internal.py", "/srv/", "127.0.0.1", "RuntimeError", "sqlite"):
        assert leak not in r.text


# ------------------------------------------------------------------ happy path + headers
async def test_success_passthrough_and_headers(stack):
    stack.primary.state.script.update(usage_total=1_234, content="hi from model")
    r = await post(stack.url)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hi from model"
    assert r.headers["x-gateway-failover"] == "0"
    assert r.headers["x-ratelimit-limit-tokens"] == "50000"
    assert r.headers["x-ratelimit-remaining-tokens"] == str(50_000 - 1_234)


async def test_failover_is_invisible_except_for_one_header(stack):
    stack.primary.state.script.update(status=429)
    r = await post(stack.url)
    assert r.status_code == 200 and r.json()["served_by"] == "secondary"
    assert r.headers["x-gateway-failover"] == "1"


async def test_timeout_failover_over_http(stack):
    stack.primary.state.script.update(hang=5)
    r = await post(stack.url)
    assert r.status_code == 200 and r.headers["x-gateway-failover"] == "1"


# ------------------------------------------------------------------ auth
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic abc"},
                                     {"Authorization": "Bearer"}, {"Authorization": KEY}])
async def test_bad_or_missing_key_is_401(stack, headers):
    r = await post(stack.url, headers=headers)
    assert_error_shape(r, 401, "unauthorized")
    assert len(stack.primary.state.requests) == 0


async def test_raw_api_key_is_never_written_to_disk(stack):
    await post(stack.url)
    for suffix in ("", "-wal", "-shm"):
        path = stack.db + suffix
        if os.path.exists(path):
            assert KEY.encode() not in open(path, "rb").read()
    import sqlite3
    assert sqlite3.connect(stack.db).execute("SELECT key_hash FROM usage").fetchone()[0] == hash_key(KEY)


# ------------------------------------------------------------------ rate limit over HTTP
async def test_429_with_retry_after_and_standard_body(stack):
    big = {**BODY, "max_tokens": 40_000}
    assert (await post(stack.url, big)).status_code == 200
    r = await post(stack.url, big)
    assert_error_shape(r, 429, "rate_limit_exceeded")
    assert 1 <= int(r.headers["retry-after"]) <= 60
    assert len(stack.primary.state.requests) == 1


async def test_per_tenant_limit_from_config(stack):
    r = await post(stack.url, {**BODY, "max_tokens": 3_000}, {"Authorization": "Bearer small-key"})
    assert_error_shape(r, 400, "request_too_large")  # bigger than that tenant's 2000/min, never retryable


async def test_parallel_http_burst_respects_budget(stack):
    body = {**BODY, "max_tokens": 960}
    rs = await asyncio.gather(*(post(stack.url, body) for _ in range(80)))
    ok = sum(r.status_code == 200 for r in rs)
    assert 45 <= ok <= 55  # about 50 requests of ~1000 tokens fit in 50k, never more
    assert all(r.status_code in (200, 429) for r in rs)
    assert len(stack.primary.state.requests) == ok


# ------------------------------------------------------------------ sanitised errors over HTTP
async def test_all_providers_down_is_503_and_clean(stack):
    stack.primary.state.script.update(status=500)
    stack.secondary.state.script.update(status=429)
    r = await post(stack.url)
    assert_error_shape(r, 503, "providers_unavailable")
    assert r.headers["retry-after"] == "5"


async def test_unexpected_crash_returns_generic_500(stack, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("secret db password hunter2 in /srv/app/db.py line 12")
    monkeypatch.setattr(stack.router, "complete", boom)
    r = await post(stack.url)
    assert_error_shape(r, 500, "internal_error")
    assert "hunter2" not in r.text and "db.py" not in r.text


async def test_database_failure_is_generic_500_not_a_stack_trace(stack):
    os.remove(stack.db)  # break the limiter's table out from under the running gateway
    for suffix in ("-wal", "-shm"):
        if os.path.exists(stack.db + suffix):
            os.remove(stack.db + suffix)
    r = await post(stack.url)
    assert_error_shape(r, 500, "internal_error")
    assert "no such table" not in r.text


@pytest.mark.parametrize("payload", [b"not json", b"[]", b'{"model":"x"}', b'{"messages":[]}'])
async def test_bad_request_bodies_are_400(stack, payload):
    async with httpx.AsyncClient() as c:
        r = await c.post(stack.url, content=payload, headers={**AUTH, "content-type": "application/json"})
    assert_error_shape(r, 400, "invalid_request")


async def test_oversized_body_is_rejected(stack):
    r = await post(stack.url, {**BODY, "messages": [{"role": "user", "content": "x" * 1_100_000}]})
    assert_error_shape(r, 413, "invalid_request")


async def test_streaming_is_refused_with_clear_message(stack):
    r = await post(stack.url, {**BODY, "stream": True})
    assert_error_shape(r, 400, "invalid_request")
    assert "Streaming" in r.json()["error"]["message"]


# ------------------------------------------------------------------ parse_tenants
def test_parse_tenants():
    t = parse_tenants("a:100, b ,c:7")
    assert t == {hash_key("a"): 100, hash_key("b"): 50_000, hash_key("c"): 7}
