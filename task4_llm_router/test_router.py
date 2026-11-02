"""Router tests against two REAL mock provider servers (primary + secondary) on real sockets."""
import asyncio
import json
import time

import pytest

from helpers import serve
from mock_provider import create_mock_app
from rate_limiter import SlidingWindowRateLimiter
from router import GatewayError, ModelRouter, Provider, estimate_tokens

pytestmark = pytest.mark.anyio

BODY = {"model": "gpt-big", "max_tokens": 100, "messages": [{"role": "user", "content": "hello there"}]}
TENANT = "tenant-hash"
LEAKS = ("Traceback", "internal.py", "/srv/", "127.0.0.1", "mock-", "gpt-backup", "Connect", "httpx")


class Stack:
    pass


@pytest.fixture
async def stack(tmp_path):
    s = Stack()
    s.primary, s.secondary = create_mock_app("primary"), create_mock_app("secondary")
    async with serve(s.primary) as p_url, serve(s.secondary) as s_url:
        s.p_url, s.s_url = p_url, s_url
        s.limiter = SlidingWindowRateLimiter(str(tmp_path / "rl.db"))
        s.router = ModelRouter(
            [Provider("primary", f"{p_url}/v1/chat/completions", "pk", timeout_s=0.4),
             Provider("secondary", f"{s_url}/v1/chat/completions", "sk", model="gpt-backup", timeout_s=2)],
            s.limiter,
        )
        yield s
        await s.router.aclose()


def assert_clean(err: GatewayError):
    blob = json.dumps(err.body("x")) + err.message
    for leak in LEAKS:
        assert leak not in blob, f"{leak!r} leaked: {blob}"
    assert set(err.body("x")["error"]) == {"type", "message", "request_id"}


# ------------------------------------------------------------------ happy path
async def test_primary_serves_when_healthy(stack):
    stack.primary.state.script.update(usage_total=300)
    r = await stack.router.complete(TENANT, BODY)
    assert r.provider == "primary" and not r.failed_over and r.data["served_by"] == "primary"
    assert len(stack.primary.state.requests) == 1 and len(stack.secondary.state.requests) == 0
    assert stack.primary.state.requests[0]["headers"]["authorization"] == "Bearer pk"


async def test_actual_usage_replaces_the_estimate(stack):
    stack.primary.state.script.update(usage_total=300)
    r = await stack.router.complete(TENANT, BODY)
    assert r.used == 300 and await stack.limiter.usage(TENANT) == 300


async def test_without_usage_block_the_estimate_is_kept(stack):
    await stack.router.complete(TENANT, BODY)
    assert await stack.limiter.usage(TENANT) == estimate_tokens(BODY)


# ------------------------------------------------------------------ failover triggers
async def test_429_from_primary_fails_over_to_secondary(stack):
    stack.primary.state.script.update(status=429)
    stack.secondary.state.script.update(usage_total=50)
    r = await stack.router.complete(TENANT, BODY)
    assert r.provider == "secondary" and r.failed_over and r.attempts == ["primary:rate_limited"]
    assert r.data["served_by"] == "secondary"
    assert await stack.limiter.usage(TENANT) == 50  # charged once, for the provider that answered


async def test_model_name_is_swapped_for_the_backup_provider(stack):
    stack.primary.state.script.update(status=429)
    await stack.router.complete(TENANT, BODY)
    assert stack.secondary.state.requests[0]["body"]["model"] == "gpt-backup"
    assert stack.secondary.state.requests[0]["headers"]["authorization"] == "Bearer sk"


async def test_primary_timeout_fails_over_fast(stack):
    stack.primary.state.script.update(hang=5)
    t0 = time.perf_counter()
    r = await stack.router.complete(TENANT, BODY)
    elapsed = time.perf_counter() - t0
    assert r.provider == "secondary" and r.attempts == ["primary:timeout"]
    assert 0.4 <= elapsed < 1.0, elapsed  # waited for the deadline, then moved on; did NOT wait 5s


async def test_spec_value_3000ms_timeout(tmp_path):
    """The real number from the assignment: primary hangs, gateway gives up at 3000 ms."""
    primary, secondary = create_mock_app("primary"), create_mock_app("secondary")
    primary.state.script.update(hang=10)
    async with serve(primary) as pu, serve(secondary) as su:
        router = ModelRouter(
            [Provider("primary", f"{pu}/v1/chat/completions", timeout_s=3.0),
             Provider("secondary", f"{su}/v1/chat/completions")],
            SlidingWindowRateLimiter(str(tmp_path / "rl.db")),
        )
        t0 = time.perf_counter()
        r = await router.complete(TENANT, BODY)
        elapsed = time.perf_counter() - t0
        await router.aclose()
    assert r.provider == "secondary"
    assert 3.0 <= elapsed < 3.6, elapsed


async def test_slow_but_within_deadline_is_not_failed_over(stack):
    stack.primary.state.script.update(hang=0.15)
    r = await stack.router.complete(TENANT, BODY)
    assert r.provider == "primary" and len(stack.secondary.state.requests) == 0


@pytest.mark.parametrize("script", [{"status": 500}, {"status": 503}, {"status": 401}, {"status": 404},
                                    {"invalid_json": True}])
async def test_other_provider_failures_also_fail_over(stack, script):
    stack.primary.state.script.update(script)
    r = await stack.router.complete(TENANT, BODY)
    assert r.provider == "secondary" and r.failed_over


async def test_connection_refused_fails_over(tmp_path):
    secondary = create_mock_app("secondary")
    async with serve(secondary) as su:
        router = ModelRouter(
            [Provider("primary", "http://127.0.0.1:1/v1/chat/completions"),
             Provider("secondary", f"{su}/v1/chat/completions")],
            SlidingWindowRateLimiter(str(tmp_path / "rl.db")),
        )
        r = await router.complete(TENANT, BODY)
        await router.aclose()
    assert r.provider == "secondary" and r.attempts == ["primary:unavailable"]


@pytest.mark.parametrize("status", [400, 413, 422])
async def test_client_errors_do_not_fail_over(stack, status):
    stack.primary.state.script.update(status=status)
    with pytest.raises(GatewayError) as e:
        await stack.router.complete(TENANT, BODY)
    assert e.value.status == 400 and e.value.type == "invalid_request"
    assert len(stack.secondary.state.requests) == 0
    assert await stack.limiter.usage(TENANT) == 0
    assert_clean(e.value)


# ------------------------------------------------------------------ everything is down
@pytest.mark.parametrize("p,s", [
    ({"status": 429}, {"status": 429}),
    ({"hang": 5}, {"status": 500}),
    ({"status": 500}, {"invalid_json": True}),
])
async def test_all_providers_failing_gives_clean_503_and_refunds_tokens(stack, p, s):
    stack.primary.state.script.update(p)
    stack.secondary.state.script.update(s)
    stack.router.providers[1] = Provider("secondary", f"{stack.s_url}/v1/chat/completions", timeout_s=0.4)
    with pytest.raises(GatewayError) as e:
        await stack.router.complete(TENANT, BODY)
    assert e.value.status == 503 and e.value.type == "providers_unavailable"
    assert e.value.headers["Retry-After"] == "5"
    assert_clean(e.value)
    assert await stack.limiter.usage(TENANT) == 0  # failed request cost the tenant nothing


async def test_unreachable_everything_is_clean(tmp_path):
    router = ModelRouter(
        [Provider("a", "http://127.0.0.1:1/x"), Provider("b", "http://localhost:2/y")],
        SlidingWindowRateLimiter(str(tmp_path / "rl.db")),
    )
    with pytest.raises(GatewayError) as e:
        await router.complete(TENANT, BODY)
    assert e.value.status == 503
    assert_clean(e.value)
    await router.aclose()


# ------------------------------------------------------------------ rate limiting through the router
async def test_over_budget_is_429_and_providers_are_not_called(stack):
    big = {**BODY, "max_tokens": 40_000}
    await stack.router.complete(TENANT, big)
    with pytest.raises(GatewayError) as e:
        await stack.router.complete(TENANT, big)
    assert e.value.status == 429 and e.value.type == "rate_limit_exceeded"
    assert 1 <= int(e.value.headers["Retry-After"]) <= 60
    assert len(stack.primary.state.requests) == 1  # second request never reached a provider
    assert_clean(e.value)


async def test_request_larger_than_whole_limit_is_400_not_429(stack):
    with pytest.raises(GatewayError) as e:
        await stack.router.complete(TENANT, {**BODY, "max_tokens": 60_000})
    assert e.value.status == 400 and e.value.type == "request_too_large"
    assert "Retry-After" not in e.value.headers and len(stack.primary.state.requests) == 0


async def test_tenants_have_separate_budgets(stack):
    big = {**BODY, "max_tokens": 49_000}
    await stack.router.complete("tenant-a", big)
    await stack.router.complete("tenant-b", big)


# ------------------------------------------------------------------ concurrency
async def test_burst_of_100_requests_admits_exactly_the_budget(stack):
    body = {**BODY, "max_tokens": 960}
    cost = estimate_tokens(body)  # about 1000 tokens per request, so about 50 fit in 50k
    assert 900 <= cost <= 1100
    results = await asyncio.gather(*(stack.router.complete(TENANT, body) for _ in range(100)), return_exceptions=True)
    ok = [r for r in results if not isinstance(r, Exception)]
    limited = [r for r in results if isinstance(r, GatewayError) and r.status == 429]
    assert len(ok) == 50_000 // cost and len(ok) + len(limited) == 100
    assert len(stack.primary.state.requests) == len(ok)  # rejected requests never hit the provider


async def test_many_concurrent_timeouts_all_fail_over_in_parallel(stack):
    stack.primary.state.script.update(hang=5)
    t0 = time.perf_counter()
    results = await asyncio.gather(*(stack.router.complete(f"t{i}", BODY) for i in range(30)))
    elapsed = time.perf_counter() - t0
    assert all(r.provider == "secondary" for r in results)
    assert elapsed < 1.2, elapsed  # timeouts overlap (0.4s each), they are not serialised (30 x 0.4 = 12s)


async def test_late_primary_answer_is_ignored_after_failover(stack):
    """Primary answers at 0.5s, just after the 0.4s deadline. We must use the secondary exactly once."""
    stack.primary.state.script.update(hang=0.5, usage_total=999)
    stack.secondary.state.script.update(usage_total=7)
    r = await stack.router.complete(TENANT, BODY)
    await asyncio.sleep(0.3)  # let the late primary answer arrive, if it ever would
    assert r.provider == "secondary"
    assert await stack.limiter.usage(TENANT) == 7  # the late answer was not double-charged


async def test_cancelled_request_releases_its_reservation(stack):
    stack.primary.state.script.update(hang=5)
    stack.router.providers[0] = Provider("primary", f"{stack.p_url}/v1/chat/completions", timeout_s=10)
    task = asyncio.create_task(stack.router.complete(TENANT, BODY))
    await asyncio.sleep(0.2)
    assert await stack.limiter.usage(TENANT) > 0  # reserved while in flight
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await stack.limiter.usage(TENANT) == 0  # client went away: tokens refunded


# ------------------------------------------------------------------ validation / estimation
@pytest.mark.parametrize("body", [[], {}, {"messages": []}, {"messages": "hi"}, {"messages": ["x"]},
                                  {"messages": [{"role": "user", "content": "x"}], "stream": True}])
async def test_invalid_bodies_are_rejected_before_any_work(stack, body):
    with pytest.raises(GatewayError) as e:
        await stack.router.complete(TENANT, body)
    assert e.value.status == 400 and await stack.limiter.usage(TENANT) == 0


def test_token_estimate_scales_with_prompt_and_max_tokens():
    short = estimate_tokens({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10})
    long = estimate_tokens({"messages": [{"role": "user", "content": "x" * 4000}], "max_tokens": 10})
    assert long - short >= 990
    assert estimate_tokens({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 500}) - short == 490
    # missing / junk max_tokens falls back to the default reservation instead of 0
    for junk in (None, 0, -5, "100", True):
        assert estimate_tokens({"messages": [{"role": "user", "content": "hi"}], "max_tokens": junk}) >= 1024
