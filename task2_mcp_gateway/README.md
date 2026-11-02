# Task 2 - MCP Security Gateway

A small reverse proxy that sits between an AI agent and a downstream MCP server and decides
who is allowed to call which tool. Python, Starlette + httpx.

```
agent --(Bearer JWT)--> gateway :8080 --> downstream MCP server :9000
```

## Run

```bash
pip install -r requirements.txt
export GATEWAY_JWT_SECRET=change-me
python mock_server.py &      # fake downstream on :9000
python gateway.py &          # gateway on :8080

TOKEN=$(python auth.py mint --role viewer)      # or --role admin
curl -s localhost:8080/mcp -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{}}}'
# viewer -> {"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unauthorized Tool Call",...}}

python -m pytest -q          # 42 tests, no ports needed
```

## What it does

| Incoming | Result |
|---|---|
| no / bad / expired token | HTTP 401, JSON-RPC error `-32002`, downstream not called |
| `tools/list` | forwarded as is |
| `tools/call`, name starts with `admin_`, role is not `admin` | gateway answers `-32001 Unauthorized Tool Call` with the same `id`, downstream not called |
| `tools/call`, any other tool, or admin role | forwarded |
| any other method (`initialize`, `ping`, notifications) | forwarded, so a real client can still do its handshake |
| bad JSON / not JSON-RPC 2.0 / batch | `-32700` / `-32600`, HTTP 400 |

Role comes from the `role` claim of an HS256 JWT (secret in `GATEWAY_JWT_SECRET`). Unknown roles count as `viewer`.

## Decisions worth knowing about

- **Forward what I parsed, not the raw bytes.** If a body has duplicate keys (`"name":"admin_x","name":"get_status"`), different JSON parsers pick different ones. I check the object I parsed and re-serialise that same object downstream, so the tool I authorised is the tool that runs. There are tests for both orderings.
- **Batches are rejected.** MCP removed batching in the 2025-06-18 spec, and forwarding a batch I never inspected would let an `admin_` call hide inside it.
- **JWT checks:** algorithm is pinned to HS256 (so `alg: none` fails), constant-time signature compare, `exp` checked, role claim tampering fails the signature.
- **Prefix match is case-insensitive** (`ADMIN_x` is blocked too). Slightly stricter than the spec, but safe if the downstream ever ignores case.
- **Authorization header is not passed downstream**; only an allow-list of headers is forwarded.
- **Responses are streamed**, so SSE replies from a streamable-HTTP server work.
- **Downstream failures are sanitised:** 502 / 504 with a generic message. The real reason only goes to the gateway log, never to the client.
- **Unauthorised notifications** (a `tools/call` with no `id`) get an empty 202 and are not forwarded, since notifications never get a reply.
- Logs go to stderr and contain the tool name and role, never arguments or tokens.

## Not done on purpose

- `tools/list` is not filtered for viewers (the task says forward transparently). Hiding `admin_*` from the list would be a nice extra.
- No rate limiting, no TLS, GET/SSE listening stream and DELETE (session end) are not proxied. POST only.
