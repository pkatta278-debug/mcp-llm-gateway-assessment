# AI Field Developer / Solutions Engineer assessment - complete project

Four tasks, one folder each, all Python. Each folder has its own README, `requirements.txt` and tests.

| Task | Folder | What it is | Tests |
|---|---|---|---|
| 1 | `task1_mcp_server/` | MCP server over stdio with `get_customer_record` and `trigger_refund`, strict input validation, stdout kept for JSON-RPC only | 73 |
| 2 | `task2_mcp_gateway/` | MCP security gateway proxy: role from the Bearer token, `admin_` tools blocked with `-32001` | 42 |
| 3 | `task3_llm_gateway/` | LLM gateway that streams the answer back with emails, SSNs and card numbers replaced by `[REDACTED]` | 80 |
| 4 | `task4_llm_router/` | Token-aware sliding-window rate limiter (on-disk SQLite) plus failover to a backup model on 429 or a 3000 ms timeout | 79 |

## Run everything

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r task1_mcp_server/requirements.txt -r task2_mcp_gateway/requirements.txt \
            -r task3_llm_gateway/requirements.txt -r task4_llm_router/requirements.txt
./run_all_tests.sh        # 274 tests, about a minute (Task 4 has one test that really waits 3 s)
```

Tests must be run from inside each task folder (the script does this), because each task imports its own modules.
See each task's README for how to start the servers by hand.
