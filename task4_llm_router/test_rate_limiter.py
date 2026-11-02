"""Rate limiter tests: window maths use a fake clock (no sleeping), races use real threads/processes."""
import asyncio
import multiprocessing as mp
import sqlite3
import threading
import time

import pytest

from rate_limiter import SlidingWindowRateLimiter

LIMIT = 50_000


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def rl(tmp_path, clock):
    return SlidingWindowRateLimiter(str(tmp_path / "rl.db"), LIMIT, 60, clock)


# ------------------------------------------------------------------ basic budget
def test_allows_exactly_up_to_the_limit(rl):
    assert rl.reserve_sync("a", 30_000).allowed
    d = rl.reserve_sync("a", 20_000)
    assert d.allowed and d.used == 50_000 and d.remaining == 0   # exactly at the limit is fine
    assert not rl.reserve_sync("a", 1).allowed                   # one token more is not


def test_tenants_are_isolated(rl):
    assert rl.reserve_sync("a", 50_000).allowed
    assert not rl.reserve_sync("a", 1).allowed
    assert rl.reserve_sync("b", 50_000).allowed


def test_per_tenant_limit_override(rl):
    assert not rl.reserve_sync("small", 600, limit=500).allowed
    assert rl.reserve_sync("big", 90_000, limit=100_000).allowed


def test_request_bigger_than_limit_is_flagged_never_retryable(rl):
    d = rl.reserve_sync("a", LIMIT + 1)
    assert not d.allowed and d.too_large and d.retry_after_ms == 0
    assert rl.usage_sync("a") == 0  # nothing was recorded


# ------------------------------------------------------------------ sliding behaviour
def test_window_slides_instead_of_resetting_all_at_once(rl, clock):
    assert rl.reserve_sync("a", 30_000).allowed      # t=0
    clock.advance(30)
    assert rl.reserve_sync("a", 20_000).allowed      # t=30  -> full
    clock.advance(29.9)                              # t=59.9: nothing has expired
    assert not rl.reserve_sync("a", 1).allowed
    clock.advance(0.2)                               # t=60.1: only the t=0 batch expired
    assert rl.usage_sync("a") == 20_000
    assert rl.reserve_sync("a", 30_000).allowed      # the freed 30k is usable immediately...
    assert not rl.reserve_sync("a", 1).allowed       # ...but the t=30 batch is still counted
    clock.advance(30)                                # t=90.1: t=30 batch is gone, the new 30k remains
    assert rl.usage_sync("a") == 30_000


def test_event_exactly_at_window_edge_is_expired(rl, clock):
    rl.reserve_sync("a", 50_000)
    clock.advance(59.999)
    assert not rl.reserve_sync("a", 1).allowed
    clock.advance(0.001)
    assert rl.reserve_sync("a", 50_000).allowed


def test_retry_after_is_exact_and_actually_works(rl, clock):
    rl.reserve_sync("a", 20_000)       # t=0
    clock.advance(10)
    rl.reserve_sync("a", 30_000)       # t=10 -> full
    clock.advance(5)                   # t=15
    d = rl.reserve_sync("a", 25_000)   # needs 25k freed: the t=0 batch (20k) is not enough, need t=10 batch too
    assert not d.allowed
    assert d.retry_after_ms == 55_000  # t=10 + 60 - 15
    clock.advance(54.9)
    assert not rl.reserve_sync("a", 25_000).allowed
    clock.advance(0.1)
    assert rl.reserve_sync("a", 25_000).allowed


def test_retry_after_when_a_small_expiry_is_enough(rl, clock):
    rl.reserve_sync("a", 20_000)       # t=0
    clock.advance(10)
    rl.reserve_sync("a", 30_000)       # t=10
    clock.advance(5)
    d = rl.reserve_sync("a", 15_000)   # the t=0 batch alone frees enough
    assert d.retry_after_ms == 45_000 and d.retry_after_s == 45


def test_denied_request_does_not_consume_budget(rl):
    rl.reserve_sync("a", 40_000)
    for _ in range(10):
        assert not rl.reserve_sync("a", 20_000).allowed
    assert rl.usage_sync("a") == 40_000


# ------------------------------------------------------------------ settle / release
def test_settle_down_gives_tokens_back(rl):
    d = rl.reserve_sync("a", 40_000)
    assert not rl.reserve_sync("a", 20_000).allowed
    used = rl.settle_sync(d.reservation_id, 1_000, "a")
    assert used == 1_000
    assert rl.reserve_sync("a", 20_000).allowed


def test_settle_up_when_provider_used_more_than_estimated(rl):
    d = rl.reserve_sync("a", 10_000)
    assert rl.settle_sync(d.reservation_id, 49_500, "a") == 49_500
    assert not rl.reserve_sync("a", 1_000).allowed


def test_release_removes_the_reservation(rl):
    d = rl.reserve_sync("a", 50_000)
    rl.release_sync(d.reservation_id)
    assert rl.usage_sync("a") == 0 and rl.reserve_sync("a", 50_000).allowed


# ------------------------------------------------------------------ eviction / persistence
def test_expired_rows_are_deleted_from_disk(rl, clock):
    for _ in range(5):
        rl.reserve_sync("a", 100)
    for _ in range(5):
        rl.reserve_sync("idle-tenant", 100)
    assert rl.row_count_sync() == 10
    clock.advance(61)
    rl.reserve_sync("a", 100)  # one call by ONE tenant evicts its own rows and purges idle tenants too
    assert rl.row_count_sync() == 1


def test_tenant_eviction_works_between_global_purges(rl, clock):
    """The global idle-tenant purge only runs once per window. The per-tenant delete in reserve() must
    be what frees an expired row when a call lands in between two purges."""
    assert rl.reserve_sync("a", 50_000).allowed      # t=0   (also triggers the first global purge)
    clock.advance(50)
    assert not rl.reserve_sync("a", 1).allowed       # t=50  -> no global purge due until t=60
    clock.advance(11)                                # t=61: the t=0 batch is expired, global purge NOT due
    assert rl.reserve_sync("a", 50_000).allowed
    assert rl.row_count_sync() == 1


def test_file_does_not_grow_over_many_windows(rl, clock):
    for _ in range(20):                 # 20 minutes of steady traffic
        for _ in range(50):
            rl.reserve_sync("a", 10)
            clock.advance(1.2)
    assert rl.row_count_sync() <= 51    # roughly one window's worth, not 1000


def test_state_survives_restart(tmp_path, clock):
    path = str(tmp_path / "rl.db")
    SlidingWindowRateLimiter(path, LIMIT, 60, clock).reserve_sync("a", 50_000)
    reopened = SlidingWindowRateLimiter(path, LIMIT, 60, clock)
    assert not reopened.reserve_sync("a", 1).allowed
    clock.advance(61)
    assert reopened.reserve_sync("a", 50_000).allowed


def test_uses_a_real_file_with_wal(rl):
    assert sqlite3.connect(rl.db_path).execute("PRAGMA journal_mode").fetchone()[0] == "wal"


@pytest.mark.parametrize("path", [":memory:", ""])
def test_in_memory_database_is_refused(path):
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(path)


# ------------------------------------------------------------------ concurrency
@pytest.mark.anyio
async def test_in_process_race_never_oversells(rl):
    results = await asyncio.gather(*(rl.reserve("a", 1_000) for _ in range(200)))
    assert sum(r.allowed for r in results) == 50
    assert rl.usage_sync("a") == 50_000


# Race tests. Oversell only happens when workers collide on the LAST slot of a budget, so a single shared
# budget gives very few chances. Instead: 200 tenants that each have room for exactly ONE request, and every
# worker walks through all of them at once. A non-atomic check-then-insert double-grants some tenant.
KEYS = [f"k{i}" for i in range(200)]


def _grants_by_key(grant_lists):
    from collections import Counter
    return Counter(k for grants in grant_lists for k in grants)


def _hammer(args):
    path, start_at = args
    rl = SlidingWindowRateLimiter(path, 1_000, 60)
    while time.time() < start_at:  # every worker begins at the same instant -> real contention
        pass
    return [k for k in KEYS if rl.reserve_sync(k, 1_000).allowed]


def test_multi_process_race_never_double_grants(tmp_path):
    """Six processes (think: six gateway replicas) share one file and start together.
    BEGIN IMMEDIATE must make check-and-insert atomic across them: each tenant is granted exactly once."""
    path = str(tmp_path / "shared.db")
    SlidingWindowRateLimiter(path, 1_000)  # create schema once
    start_at = time.time() + 1.5           # spawn start-up is slow; wait until everyone is ready
    with mp.get_context("spawn").Pool(6) as pool:
        grants = _grants_by_key(pool.map(_hammer, [(path, start_at)] * 6))
    assert set(grants) == set(KEYS) and max(grants.values()) == 1, \
        {k: v for k, v in grants.items() if v != 1}


# NOTE: reserve_sync starts its transaction with BEGIN IMMEDIATE. Because the first statement is a DELETE
# a plain BEGIN would behave the same today, but IMMEDIATE keeps it safe if someone reorders the statements.
def test_threads_bypassing_the_asyncio_lock_never_double_grant(tmp_path):
    """16 threads call the sync core directly (no asyncio.Lock in the way), released together by a barrier.
    Proves contention never surfaces as 'database is locked' errors and never double-grants. (The threads
    are too serialised by the GIL to expose a missing transaction; the multi-process test above does that.)"""
    rl = SlidingWindowRateLimiter(str(tmp_path / "t.db"), 1_000, 60)
    barrier, errors, results = threading.Barrier(16), [], []

    def work():
        barrier.wait()
        got = []
        for k in KEYS:
            try:
                if rl.reserve_sync(k, 1_000).allowed:
                    got.append(k)
            except Exception as exc:  # losing a race must never surface as an error either
                errors.append(exc)
        results.append(got)

    threads = [threading.Thread(target=work) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    grants = _grants_by_key(results)
    assert not errors, errors[:2]
    assert set(grants) == set(KEYS) and max(grants.values()) == 1, {k: v for k, v in grants.items() if v != 1}
