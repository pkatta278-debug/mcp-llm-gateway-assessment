# Task 1 - Custom MCP Server (strict validation + stdio)

An MCP server (Python, official `mcp` SDK + Pydantic) exposing two tools over **stdio**:

| Tool | Inputs | Rules |
|---|---|---|
| `get_customer_record` | `customer_id` | must match `CUST-XXXXX` (5 digits) |
| `trigger_refund` | `customer_id`, `amount`, `reason` | `amount` > 0 finite float, `reason` >= 10 chars |

## Run

```bash
pip install -r requirements.txt
python server.py          # speaks JSON-RPC on stdin/stdout, logs on stderr
python -m pytest -q       # 73 tests, ~8s
```

Claude Desktop / any MCP client config:

```json
{ "mcpServers": { "customer-ops": { "command": "python", "args": ["/abs/path/to/task1_mcp_server/server.py"] } } }
```

Sample customers: `CUST-10001` (active, $1250 paid), `CUST-10002` (active, $480.50), `CUST-10003` (suspended).

## Design decisions

### 1. STDIO isolation (stdout = JSON-RPC only)
- `server.py` never calls `print()` or touches `sys.stdout`; `logging` is configured with `stream=sys.stderr`.
- Logs record tool name and outcome only, never arguments (a refund reason can contain personal data).
- The SDK's stdio transport additionally re-points fd 1 at stderr while serving, so a stray `print()` from any dependency lands on stderr instead of corrupting the wire.
- Tests prove this four ways: raw-wire check that every stdout line is a JSON-RPC 2.0 frame; an AST scan for `print`/`sys.stdout`; a check that the logging handler targets stderr; and a hostile test where a handler deliberately prints (stdout stays clean, the text shows up on stderr). I also confirmed the checks fail when a `print()` is injected.

### 2. Two kinds of failure, two kinds of response
| Situation | Response |
|---|---|
| Malformed input (bad ID, amount <= 0, short reason, missing/extra field, wrong type) | **JSON-RPC error `-32602 Invalid params`**, with `data.errors = [{field, type, message}]` for every problem at once |
| Unknown tool name | `-32602` (what the MCP spec prescribes) |
| Valid call that can't be done (unknown customer, suspended account, refund over total paid) | Normal result with **`isError: true`** (MCP "tool execution error") |

Validation errors never echo the submitted values back (tested).

### 3. Validation details
- **Strict mode**: no coercion. `"10.5"` is not a number, `true` is not an amount, `12345` is not an ID.
- `customer_id` uses `[0-9]`, not `\d`, so fullwidth/Arabic-Indic digits are rejected; `"CUST-10001\n"` is rejected (trailing-newline trap).
- `amount`: `gt=0`, `NaN`/`Infinity` rejected, integers accepted (`50` is a valid JSON number).
- `reason`: min 10, max 500, and whitespace-only padding doesn't count (`" " * 10` is rejected).
- `additionalProperties: false`: unknown fields are rejected, and the same Pydantic model generates the `inputSchema` shown in `tools/list`, so docs and enforcement cannot drift.

### 4. Business logic (kept small, but correct)
Refund total per customer cannot exceed what they paid; the check-then-write is serialised with an `asyncio.Lock` (a test fires 5 concurrent refunds and verifies no overspend).

## Assumptions to be aware of
These go slightly beyond the literal spec; each is one line to relax in `server.py`:
- `XXXXX` = **five digits** (spec doesn't say digits vs alphanumeric).
- `amount` is limited to **2 decimal places** (no sub-cent refunds) and `reason` to **500 chars**.
- Reason must have >= 10 characters **after trimming** whitespace.

## Notes
- The server exits when stdin closes (the standard MCP stdio shutdown signal). Clients should keep stdin open until they have received their responses.
- Data is in-memory mock data; the ledger resets on restart.
