# AI Field Developer / Solutions Engineer assessment - complete project

Four tasks, one folder each, all Python. Each folder has its own README, `requirements.txt` and tests.

| Task | Folder | What it is | Tests |
|---|---|---|---|
| 1 | `task1_mcp_server/` | MCP server over stdio with `get_customer_record` and `trigger_refund`, strict validation, stdout kept for JSON-RPC only | 73 |
| 2 | `task2_mcp_gateway/` | MCP security gateway proxy: role from the Bearer token, `admin_` tools blocked with `-32001` | 42 |
| 3 | `task3_llm_gateway/` | LLM gateway that streams the answer back with emails, SSNs and card numbers replaced by `[REDACTED]` | 80 |
| 4 | `task4_llm_router/` | Token-aware sliding-window rate limiter (on-disk SQLite) plus failover to a backup model on 429 or a 3000 ms timeout | 79 |

## Run everything

Needs **Python 3.11+** (Task 4 uses `asyncio.timeout`). Verified on Python 3.12.3 in a fresh virtualenv.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r task1_mcp_server/requirements.txt -r task2_mcp_gateway/requirements.txt \
            -r task3_llm_gateway/requirements.txt -r task4_llm_router/requirements.txt
./run_all_tests.sh        # 274 tests, about a minute (Task 4 has one test that really waits 3 s)
```

Run tests from inside each task folder (the script does this): each task imports its own modules.
See each task's README for how to start the servers by hand.

## Verification against the assessment PDF

Besides the unit and integration tests, every requirement in the PDF was checked by running the real code
(real processes, real sockets) and reading the result. Last run: all 274 tests pass in a clean install.

| Task | PDF requirement | Result |
|---|---|---|
| 1 | Two tools; `customer_id` = `CUST-XXXXX`; `amount` positive float; `reason` min 10 chars | Pass. 19 live probes over stdio: wrong case, 4 / 6 digits, trailing newline, non-ASCII digits, wrong types, missing / extra fields, amount 0 / negative / string / bool, reason 9 chars / 10 spaces |
| 1 | Reject invalid input with standard JSON-RPC error codes | Pass. Every invalid input returned `-32602` |
| 1 | stdout only JSON-RPC, logs on stderr | Pass. Every stdout line parsed as JSON, nothing left on stdout at shutdown, 20 log lines on stderr, no `print()` in the code |
| 2 | Role from `Bearer <token>` | Pass. Viewer and admin tokens map to the right role; unknown / missing role falls back to viewer |
| 2 | `tools/list` forwarded transparently | Pass for both roles |
| 2 | `admin_*` needs admin, else `-32001 Unauthorized Tool Call` without calling downstream | Pass. Viewer got `-32001` and the downstream access log showed no request; admin got through. Also stopped before the downstream: `ADMIN_` in capitals, a forged `role=admin` token with a bad signature, an admin call hidden in a batch |
| 3 | Intercept the stream, redact emails / SSNs / cards as `[REDACTED]` | Pass. Values split across 3-5 chunks were redacted; no fragment appeared on the wire; Luhn-invalid numbers and order ids were left alone |
| 3 | No full-response buffering, low TTFT | Pass. 400,000 chunks (83 MB on the wire) through the gateway: its peak memory grew from 34 MB to 36 MB; all 8,000 injected emails redacted, none leaked; first token after 62 ms. With a slow upstream (one token per 400 ms) the first token still arrived in 23 ms |
| 4 | Sliding window, 50,000 tokens/minute per tenant key | Pass. Second 40k request got `429` with `Retry-After: 60` and never reached a provider |
| 4 | Failover on 429 or 3000 ms timeout | Pass. Primary 429 -> secondary in 27 ms; primary hanging -> secondary after 3.008 s |
| 4 | Standardised error payload, no stack traces or internals | Pass. Fake tracebacks, file paths, IPs and model names from providers never appear in a response; a crash and a broken database both return a generic 500 |
| 4 | On-disk SQLite | Pass. WAL file on disk, state survives restart, `:memory:` refused; raw API key never written to disk |

## Issues found

Nothing fails a stated PDF requirement. These are the gaps I found while testing, most important first.

| # | Where | Issue | Severity | Suggested fix |
|---|---|---|---|---|
| 1 | Task 2 `auth.py` | A correctly signed token with **no `exp` claim is accepted and never expires**, so an admin token minted without an expiry works forever. Expired tokens are rejected correctly. | Medium | In `verify_token`, raise `AuthError` when `exp` is missing, and add a test |
| 2 | Task 2 `auth.py` | An `nbf` ("not before") claim in the future is ignored. | Low | Reject the token while `nbf > now`, or document that `nbf` is not supported |
| 3 | Task 3 `gateway.py` | httpx logs the full upstream URL at INFO level, so a provider URL with a key in its query string would end up in the server log (never in a client response). | Low | `logging.getLogger("httpx").setLevel(logging.WARNING)`. Task 4 already does this |

Design limits that are documented, not defects:
- **Task 4:** non-streaming only (`stream: true` returns a clear 400), the reservation is an estimate (characters / 4)
  that is corrected to real usage afterwards, no circuit breaker (a fully dead primary costs each request its 3 s
  before failover), and a database failure returns 500 instead of serving without limits.
- **Task 1:** a well-formed call for an unknown customer returns a normal result with `isError=true`, not a JSON-RPC
  error. That follows the MCP spec; JSON-RPC errors are used for invalid input, as the PDF asks.

Fixed during this audit (Task 4, my own code): two unused imports, and the concurrency tests were rewritten after
a deliberate-breakage check showed they could not detect a missing transaction.
