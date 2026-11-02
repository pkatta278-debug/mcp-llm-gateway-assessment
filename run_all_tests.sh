#!/usr/bin/env bash
# Runs every task's test suite in its own folder (each task has its own imports and requirements).
set -u
cd "$(dirname "$0")"
fail=0
for d in task1_mcp_server task2_mcp_gateway task3_llm_gateway task4_llm_router; do
  echo "=== $d"
  (cd "$d" && python -m pytest -q -p no:cacheprovider 2>&1 | tail -2) || fail=1
done
exit $fail
