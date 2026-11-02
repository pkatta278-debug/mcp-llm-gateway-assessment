# Task 4 - Rate limiting and model fallback router for an LLM gateway

An OpenAI-compatible endpoint (`POST /v1/chat/completions`) that checks a **per-tenant token budget**
(50,000 tokens/minute, sliding window, kept in an on-disk SQLite file), sends the request to a **primary model
provider**, and **fails over to a backup provider** on a 429 or a 3000 ms timeout. Errors always use one
sanitised body. Python, Starlette + httpx, same stack as Task 3.

```bash
pip install -r requirements.txt
python mock_provider.py 9100 &      # fake primary on :9100
python mock_provider.py 9101 &      # fake secondary on :9101
TENANT_KEYS=demo-key:50000 DB_PATH=./ratelimit.db \
PRIMARY_URL=http://127.0.0.1:9100/v1/chat/completions \
SECONDARY_URL=http://127.0.0.1:9101/v1/chat/completions SECONDARY_MODEL=backup-model \
python gateway.py &                 # gateway on :8082

curl -s localhost:8082/v1/chat/completions -H 'Authorization: Bearer demo-key' -H 'content-type: application/json' \
  -d '{"model":"big","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}'

# make the primary misbehave (the mock has a control endpoint):
curl -s localhost:9100/_script -d '{"status":429}'   # -> answered by the secondary, header X-Gateway-Failover: 1
curl -s localhost:9100/_script -d '{"hang":10}'      # -> gateway waits 3000 ms, then the secondary answers

python -m pytest -q                 # 79 tests, ~40s (one test really waits 3 s, every test starts real servers)
```

## Requirements, one by one

| Requirement (from the assignment) | Where | Proven by |
|---|---|---|
| Token-aware sliding window, 50,000 tokens/minute per tenant key | `rate_limiter.py` (sum of tokens of rows younger than 60 s); default in `DEFAULT_LIMIT`, per-key override via `TENANT_KEYS=key:limit` | `test_rate_limiter.py`: exact-limit, window slides (not resets), edge-of-window, per-tenant isolation |
| Failover to a backup provider on **429** | `ModelRouter._call` -> `UpstreamFailure("rate_limited")` -> next provider | `test_429_from_primary_fails_over_to_secondary` + live run |
| Failover on **timeout after 3000 ms** | primary `Provider(timeout_s=3.0)` is the default (`PRIMARY_TIMEOUT_MS`), enforced with one hard `asyncio.timeout` | `test_spec_value_3000ms_timeout` (real 3 s: failover lands between 3.0 and 3.6 s) |
| Standardised error payload, no stack traces / internals | one body shape `{"error": {"type", "message", "request_id"}}`, built only from `GatewayError` | `assert_clean` in every error test; fake tracebacks, paths, IPs, model names are checked absent |
| **On-disk SQLite** for the database | `SlidingWindowRateLimiter(db_path)` refuses `:memory:`; WAL mode | `test_uses_a_real_file_with_wal`, `test_state_survives_restart`, `test_in_memory_database_is_refused` |
| Async concurrency and timeout races | see "Races" below | burst, parallel-timeout, late-answer and cancellation tests |
| Accurate eviction and token tracking | expired rows deleted on every `reserve()`, all tenants purged once per window; estimate replaced by real usage | eviction, file-size and settle tests |

## How a request flows

```
client --Bearer key--> gateway
   1. key -> sha256 -> tenant (unknown key: 401, nothing else happens)
   2. estimate cost = ceil(prompt chars / 4) + max_tokens   (default 1024 if the client gave none)
   3. RESERVE the cost in SQLite, atomically
        over budget  -> 429 + Retry-After (exact seconds until enough old tokens expire), providers never called
        cost > limit -> 400 request_too_large (waiting could never help, so no Retry-After)
   4. primary (3000 ms hard deadline) -> on failure -> secondary (10 s)
   5. success -> replace the estimate with usage.total_tokens from the provider, return the body
      failure -> RELEASE the reservation, return 503 providers_unavailable
```

Why reserve first and settle later: a check-then-record scheme lets 100 parallel requests all see "budget free" and
all go through. Reserving inside the same transaction as the check closes that hole, and settling afterwards means
tenants are charged for what the model really used, not for the worst case.

## Failover rules

| Provider result | Action |
|---|---|
| 429, timeout (3000 ms), 5xx, connection refused / reset / DNS, 200 with a body that is not a JSON object | next provider |
| 401 / 403 / 404 from the provider | next provider (our own provider config is wrong; log it, keep serving) |
| 400 / 413 / 422 | **stop**: the request itself is bad, another provider will not fix it. Client gets a generic 400 |
| every provider failed | 503 `providers_unavailable`, `Retry-After: 5` |

The assignment names 429 and timeouts. 5xx and connection errors are added on purpose: a provider that is down is
the most common reason to need a backup. The model name is swapped per provider (`SECONDARY_MODEL`) because
providers name models differently. Only the provider that actually answered is charged.

## Races and timeouts

- **Atomic budget check.** `reserve()` is one `BEGIN IMMEDIATE` transaction (delete expired, sum, insert). SQLite takes
  the write lock before reading, so two requests can never both take the last tokens, **even from different
  processes** (several gateway replicas can share the file). Tests: 6 processes starting at the same instant and
  16 threads, each walking 200 tenants that have room for exactly one request; every tenant must be granted exactly once.
- **One hard deadline per attempt.** `asyncio.timeout()` wraps connect + send + wait + reading the body. httpx's own
  timeouts are per phase, so a provider that trickles bytes could otherwise outlive 3000 ms.
- **Late answers are discarded.** When the deadline fires the request is cancelled, so a primary that answers at 3.1 s
  is never used or charged (`test_late_primary_answer_is_ignored_after_failover`). Timeouts of concurrent requests
  overlap instead of queueing (30 hung requests finish in ~0.5 s, not 30 x 0.4 s).
- **No leaked reservations.** Reserved tokens are released in a `finally`, including when the task is cancelled (client
  disconnect). The release is `shield`ed so it still completes during cancellation.
- **Event loop stays free.** SQLite calls run in a worker thread; one in-process lock serialises them, the transaction
  covers other processes.

## Rate limiter details

- Table `usage(id, key_hash, ts_ms, tokens)` with an index on `(key_hash, ts_ms)`. A row is "in the window" while
  `ts_ms > now - 60 s`; at exactly 60 s it is gone.
- `Retry-After` is exact: it walks the oldest rows until enough tokens would have expired for this request to fit.
  Tests wait that long on a fake clock and check it works, and that 0.1 s earlier it does not.
- Eviction: the caller's expired rows go on every request; rows of idle tenants are purged at most once per window.
  A 20-minute simulated run keeps the file at about one window of rows, not thousands.
- API keys are stored as SHA-256 only. The raw key never touches the disk or the logs
  (`test_raw_api_key_is_never_written_to_disk` scans the DB, WAL and SHM files for it).
- Timestamps use the wall clock, not a monotonic one, because they must still mean something after a restart.

## Error contract

```json
{"error": {"type": "rate_limit_exceeded", "message": "Token rate limit of 50000 per minute reached. Retry in 60s.", "request_id": "gw-62133890eab23ab9"}}
```

| Status | `type` | When |
|---|---|---|
| 400 | `invalid_request` / `request_too_large` | bad body, `stream: true`, provider said the request is invalid, estimate > whole limit |
| 401 | `unauthorized` | missing / unknown key |
| 413 | `invalid_request` | body over 1 MB |
| 429 | `rate_limit_exceeded` | tenant over budget (+ `Retry-After`) |
| 500 | `internal_error` | any bug or database failure; the exception goes to the log only |
| 503 | `providers_unavailable` | all providers failed (+ `Retry-After: 5`) |

Messages are fixed strings written for clients. Provider status text, bodies, stack traces, URLs and model names are
logged next to the `request_id` and never sent. The client gets the same id (also in `X-Request-Id`) so support can
find the log line. Success responses add `X-Gateway-Failover: 0|1` and `X-RateLimit-Limit/Remaining-Tokens`.

## Known limits

- **Non-streaming only.** `stream: true` is refused with a clear 400 (Task 3 is the streaming gateway). Streaming
  failover would only be possible until the first byte, and "3000 ms" would then mean time to first token.
- **The reservation is an estimate** (characters / 4). It is corrected to the provider's real count afterwards, but a
  request whose real usage is larger than its estimate can push a tenant slightly over the limit until the window
  slides. If the provider reports no usage, the estimate is kept.
- **Database failure returns 500, not "allow".** If SQLite cannot be reached the gateway refuses to serve rather
  than serve without limits (fails closed).
- A provider that fails is retried on the next request; there is no circuit breaker, so a provider that is fully down
  costs every request its timeout (3 s for the primary) before failover.
- Tenants come from the `TENANT_KEYS` env var; a real deployment would load them from a secrets store.

## Test summary (79)

- `test_rate_limiter.py` (22): window maths on a fake clock, exact `Retry-After`, settle / release, eviction from disk,
  restart persistence, the in-process, thread and multi-process races.
- `test_router.py` (35): every failover trigger, no failover on client errors, all-providers-down refunds tokens,
  burst of 100 admits exactly the budget, 30 parallel timeouts, late primary answer, cancellation releases tokens.
- `test_gateway.py` (22): full HTTP path, auth, headers, error shape on every error, crash and DB failure give a
  generic 500, key never on disk.

Tests were also checked by breaking the code on purpose (no transaction, no release on failure, no eviction, no failover
on 429, wrong `Retry-After`, client errors failing over, usage not settled): each of these makes at least one test fail.
