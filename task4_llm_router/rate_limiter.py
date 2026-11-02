"""Task 4 - token-aware sliding-window rate limiter backed by an on-disk SQLite file.

Algorithm: "sliding window log". Every request writes one row (key_hash, timestamp, tokens).
A tenant's usage is the SUM of tokens whose timestamp is inside the last `window` seconds,
so the budget frees up smoothly as old requests age out (no fixed-window burst at the minute edge).

Flow for one request:
    reserve(cost)   check + insert in ONE transaction (BEGIN IMMEDIATE), so concurrent requests
                    can never both squeeze under the limit
    settle(id, n)   after the provider answers, replace the estimate with the real token count
    release(id)     provider failed / request cancelled: give the reservation back

Why SQLite is safe here:
    * BEGIN IMMEDIATE takes the write lock before reading, so check-then-insert is atomic across
      threads AND across processes sharing the same file (several gateway workers).
    * WAL mode: readers never block the writer, writes survive a restart.
    * API keys are hashed before they get here, the raw key is never written to disk.

Eviction: expired rows of the calling tenant are deleted inside every reserve(), and rows of ALL
tenants are purged at most once per window, so the file does not grow without bound.
"""
from __future__ import annotations

import asyncio
import math
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable

DEFAULT_LIMIT = 50_000
DEFAULT_WINDOW_S = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash TEXT    NOT NULL,
    ts_ms    INTEGER NOT NULL,
    tokens   INTEGER NOT NULL CHECK (tokens >= 0)
);
CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON usage (key_hash, ts_ms);
"""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reservation_id: int | None  # set when allowed
    limit: int
    used: int                   # tokens in the window, including this reservation when allowed
    retry_after_ms: int         # when denied: earliest time the same request could fit (0 if never)
    too_large: bool = False     # cost > limit: no amount of waiting will ever help

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def retry_after_s(self) -> int:
        return max(1, math.ceil(self.retry_after_ms / 1000)) if self.retry_after_ms else 0


class SlidingWindowRateLimiter:
    def __init__(
        self,
        db_path: str,
        default_limit: int = DEFAULT_LIMIT,
        window_s: float = DEFAULT_WINDOW_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if db_path == ":memory:" or not db_path:
            raise ValueError("an on-disk database path is required")
        self.db_path = db_path
        self.default_limit = default_limit
        self.window_ms = int(window_s * 1000)
        self._clock = clock  # wall clock on purpose: timestamps must stay meaningful after a restart
        self._last_purge_ms = 0
        # One DB operation at a time per process; BEGIN IMMEDIATE covers other processes.
        self._lock = asyncio.Lock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # persistent setting of the file
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ plumbing
    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None -> we issue BEGIN/COMMIT ourselves. timeout = busy wait for other processes.
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    # ------------------------------------------------------------------ sync core (runs in a worker thread)
    def reserve_sync(self, key_hash: str, cost: int, limit: int | None = None) -> Decision:
        limit = self.default_limit if limit is None else limit
        if cost > limit:
            return Decision(False, None, limit, 0, 0, too_large=True)
        now = self._now_ms()
        cutoff = now - self.window_ms
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM usage WHERE key_hash = ? AND ts_ms <= ?", (key_hash, cutoff))
                if now - self._last_purge_ms >= self.window_ms:  # housekeeping for idle tenants
                    conn.execute("DELETE FROM usage WHERE ts_ms <= ?", (cutoff,))
                    self._last_purge_ms = now
                used = conn.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM usage WHERE key_hash = ?", (key_hash,)
                ).fetchone()[0]

                if used + cost <= limit:
                    cur = conn.execute(
                        "INSERT INTO usage (key_hash, ts_ms, tokens) VALUES (?, ?, ?)", (key_hash, now, cost)
                    )
                    decision = Decision(True, cur.lastrowid, limit, used + cost, 0)
                else:
                    # Walk the oldest rows until enough tokens would have expired to fit this request.
                    need, freed, retry_ms = used + cost - limit, 0, self.window_ms
                    for ts_ms, tokens in conn.execute(
                        "SELECT ts_ms, tokens FROM usage WHERE key_hash = ? ORDER BY ts_ms, id", (key_hash,)
                    ):
                        freed += tokens
                        if freed >= need:
                            retry_ms = ts_ms + self.window_ms - now
                            break
                    decision = Decision(False, None, limit, used, max(1, retry_ms))
                conn.execute("COMMIT")
                return decision
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def settle_sync(self, reservation_id: int, actual_tokens: int, key_hash: str) -> int:
        """Swap the estimate for the real count. Returns the tenant's usage afterwards."""
        conn = self._connect()
        try:
            conn.execute("UPDATE usage SET tokens = ? WHERE id = ?", (max(0, actual_tokens), reservation_id))
            cutoff = self._now_ms() - self.window_ms
            return conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM usage WHERE key_hash = ? AND ts_ms > ?", (key_hash, cutoff)
            ).fetchone()[0]
        finally:
            conn.close()

    def release_sync(self, reservation_id: int) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM usage WHERE id = ?", (reservation_id,))
        finally:
            conn.close()

    def usage_sync(self, key_hash: str) -> int:
        cutoff = self._now_ms() - self.window_ms
        conn = self._connect()
        try:
            return conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM usage WHERE key_hash = ? AND ts_ms > ?", (key_hash, cutoff)
            ).fetchone()[0]
        finally:
            conn.close()

    def row_count_sync(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0]
        finally:
            conn.close()

    # ------------------------------------------------------------------ async API (keeps the event loop free)
    async def reserve(self, key_hash: str, cost: int, limit: int | None = None) -> Decision:
        async with self._lock:
            return await asyncio.to_thread(self.reserve_sync, key_hash, cost, limit)

    async def settle(self, reservation_id: int, actual_tokens: int, key_hash: str) -> int:
        async with self._lock:
            return await asyncio.to_thread(self.settle_sync, reservation_id, actual_tokens, key_hash)

    async def release(self, reservation_id: int) -> None:
        async with self._lock:
            await asyncio.to_thread(self.release_sync, reservation_id)

    async def usage(self, key_hash: str) -> int:
        async with self._lock:
            return await asyncio.to_thread(self.usage_sync, key_hash)
