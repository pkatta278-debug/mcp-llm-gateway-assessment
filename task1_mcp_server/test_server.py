"""Tests for Task 1.

Four layers:
  1. dispatch_tool unit tests      -> validation edge cases, error mapping
  2. end-to-end via the MCP client -> real stdio transport, real JSON-RPC
  3. raw wire test                 -> every byte on stdout is a JSON-RPC frame
  4. hostile test                  -> a handler that print()s must NOT corrupt stdout
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import mcp_types as types
import pytest
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError

import server

HERE = Path(__file__).parent
SERVER_PY = HERE / "server.py"
INVALID_PARAMS = -32602


@pytest.fixture(autouse=True)
def _clean_ledger():
    server.REFUNDS.clear()
    yield
    server.REFUNDS.clear()


def run(coro):
    return asyncio.run(coro)


def call(name, arguments):
    return run(server.dispatch_tool(name, arguments))


def payload(result: types.CallToolResult) -> dict:
    return json.loads(result.content[0].text)


def assert_invalid_params(name, arguments):
    with pytest.raises(MCPError) as exc:
        call(name, arguments)
    assert exc.value.code == INVALID_PARAMS
    return exc.value


# =========================================================================== #
# 1. Unit tests
# =========================================================================== #
class TestGetCustomerRecord:
    def test_happy_path(self):
        res = call("get_customer_record", {"customer_id": "CUST-10001"})
        assert res.is_error is False
        data = payload(res)
        assert data["customer_id"] == "CUST-10001"
        assert data["name"] == "Asha Verma"
        assert "total_paid" not in data  # internal field must not leak

    def test_unknown_customer_is_tool_error_not_protocol_error(self):
        res = call("get_customer_record", {"customer_id": "CUST-99999"})
        assert res.is_error is True
        assert payload(res)["error"] == "customer_not_found"

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",  # empty
            "CUST-1234",  # too short
            "CUST-123456",  # too long
            "CUST-ABCDE",  # letters instead of digits
            "cust-10001",  # wrong case
            "CUST10001",  # missing dash
            " CUST-10001",  # leading space
            "CUST-10001 ",  # trailing space
            "CUST-10001\n",  # trailing newline ($-anchor trap)
            "CUST-1000１",  # fullwidth digit (\d-trap)
            "CUST-١٢٣٤٥",  # Arabic-Indic digits
            "XCUST-10001",
            "CUST-10001-extra",
            None,
            12345,
            ["CUST-10001"],
            {"id": "CUST-10001"},
        ],
    )
    def test_invalid_customer_id(self, bad_id):
        assert_invalid_params("get_customer_record", {"customer_id": bad_id})

    @pytest.mark.parametrize("args", [{}, None, {"id": "CUST-10001"}])
    def test_missing_customer_id(self, args):
        assert_invalid_params("get_customer_record", args)

    def test_unexpected_extra_field_rejected(self):
        assert_invalid_params("get_customer_record", {"customer_id": "CUST-10001", "admin": True})


class TestTriggerRefund:
    GOOD = {"customer_id": "CUST-10001", "amount": 25.5, "reason": "Duplicate charge on invoice"}

    def test_happy_path(self):
        res = call("trigger_refund", self.GOOD)
        assert res.is_error is False
        data = payload(res)
        assert data["refund_id"].startswith("RFD-")
        assert data["amount"] == 25.5
        assert data["status"] == "queued"
        assert "reason" not in data
        assert len(server.REFUNDS) == 1

    def test_integer_amount_is_accepted(self):
        # JSON "50" is a legitimate number; strict mode must not reject int -> float.
        res = call("trigger_refund", {**self.GOOD, "amount": 50})
        assert res.is_error is False

    @pytest.mark.parametrize(
        "bad_amount",
        [0, 0.0, -1, -0.01, -1e9, "10", "10.5", True, False, None, [], {}, math.nan, math.inf, -math.inf, 10.005, 0.001],
    )
    def test_invalid_amount(self, bad_amount):
        assert_invalid_params("trigger_refund", {**self.GOOD, "amount": bad_amount})
        assert server.REFUNDS == []  # nothing recorded on rejection

    @pytest.mark.parametrize(
        "bad_reason",
        ["", "short", "123456789", " " * 10, "  abc      ", None, 1234567890123, "x" * 501],
    )
    def test_invalid_reason(self, bad_reason):
        assert_invalid_params("trigger_refund", {**self.GOOD, "reason": bad_reason})

    def test_reason_boundaries(self):
        assert call("trigger_refund", {**self.GOOD, "reason": "x" * 10}).is_error is False
        assert call("trigger_refund", {**self.GOOD, "reason": "y" * 500, "amount": 1.0}).is_error is False

    def test_invalid_customer_id(self):
        assert_invalid_params("trigger_refund", {**self.GOOD, "customer_id": "CUST-12"})

    @pytest.mark.parametrize("missing", ["customer_id", "amount", "reason"])
    def test_missing_required_field(self, missing):
        args = {k: v for k, v in self.GOOD.items() if k != missing}
        err = assert_invalid_params("trigger_refund", args)
        assert any(e["field"] == missing for e in err.data["errors"])

    def test_all_problems_reported_at_once(self):
        err = assert_invalid_params("trigger_refund", {"customer_id": "bad", "amount": -1, "reason": "no"})
        assert {e["field"] for e in err.data["errors"]} == {"customer_id", "amount", "reason"}

    def test_unexpected_extra_field_rejected(self):
        assert_invalid_params("trigger_refund", {**self.GOOD, "override": True})

    # --- business rules: well-formed but not allowed -> isError result, no exception
    def test_unknown_customer(self):
        res = call("trigger_refund", {**self.GOOD, "customer_id": "CUST-99999"})
        assert res.is_error is True and payload(res)["error"] == "customer_not_found"

    def test_suspended_customer_not_eligible(self):
        res = call("trigger_refund", {**self.GOOD, "customer_id": "CUST-10003", "amount": 1.0})
        assert res.is_error is True and payload(res)["error"] == "customer_not_eligible"

    def test_refund_cannot_exceed_total_paid_cumulatively(self):
        # CUST-10002 total_paid = 480.50
        assert call("trigger_refund", {**self.GOOD, "customer_id": "CUST-10002", "amount": 400.0}).is_error is False
        res = call("trigger_refund", {**self.GOOD, "customer_id": "CUST-10002", "amount": 100.0})
        assert res.is_error is True
        data = payload(res)
        assert data["error"] == "refund_exceeds_total_paid"
        assert data["max_refundable"] == 80.5

    def test_concurrent_refunds_cannot_overspend(self):
        async def burst():
            return await asyncio.gather(
                *[
                    server.dispatch_tool(
                        "trigger_refund",
                        {"customer_id": "CUST-10002", "amount": 200.0, "reason": "Concurrent refund attempt"},
                    )
                    for _ in range(5)
                ]
            )

        results = run(burst())
        assert sum(1 for r in results if not r.is_error) == 2  # 2 x 200 <= 480.50 < 3 x 200


class TestProtocolErrors:
    def test_unknown_tool(self):
        with pytest.raises(MCPError) as exc:
            call("drop_database", {})
        assert exc.value.code == INVALID_PARAMS

    def test_validation_errors_never_echo_user_input(self):
        secret = "SECRET-TOKEN-abc123"
        err = assert_invalid_params("trigger_refund", {"customer_id": secret, "amount": 5, "reason": secret})
        assert secret not in json.dumps(err.error.model_dump())


# =========================================================================== #
# 2. End-to-end over the real stdio transport using the official client
# =========================================================================== #
def _params() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=[str(SERVER_PY)])


def with_session(fn):
    async def go():
        async with stdio_client(_params()) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                return await fn(session)

    return asyncio.run(go())


class TestEndToEnd:
    def test_lists_exactly_the_two_tools_with_schemas(self):
        tools = with_session(lambda s: s.list_tools()).tools
        by_name = {t.name: t for t in tools}
        assert set(by_name) == {"get_customer_record", "trigger_refund"}
        refund = by_name["trigger_refund"].input_schema
        assert set(refund["required"]) == {"customer_id", "amount", "reason"}
        assert refund["properties"]["reason"]["minLength"] == 10
        assert refund["properties"]["amount"]["exclusiveMinimum"] == 0
        assert refund["additionalProperties"] is False
        assert by_name["get_customer_record"].input_schema["properties"]["customer_id"]["pattern"] == "^CUST-[0-9]{5}$"

    def test_valid_calls(self):
        async def scenario(s):
            rec = await s.call_tool("get_customer_record", {"customer_id": "CUST-10001"})
            ref = await s.call_tool(
                "trigger_refund", {"customer_id": "CUST-10001", "amount": 10.0, "reason": "Customer was double billed"}
            )
            return rec, ref

        rec, ref = with_session(scenario)
        assert rec.is_error is False and json.loads(rec.content[0].text)["name"] == "Asha Verma"
        assert ref.is_error is False and json.loads(ref.content[0].text)["status"] == "queued"

    def test_invalid_input_surfaces_as_jsonrpc_minus_32602(self):
        async def scenario(s):
            with pytest.raises(MCPError) as exc:
                await s.call_tool("get_customer_record", {"customer_id": "nope"})
            return exc.value

        err = with_session(scenario)
        assert err.code == INVALID_PARAMS
        assert err.data["errors"][0]["field"] == "customer_id"

    def test_server_survives_bad_calls_and_keeps_serving(self):
        async def scenario(s):
            for _ in range(3):
                with pytest.raises(MCPError):
                    await s.call_tool("trigger_refund", {"customer_id": "x", "amount": -1, "reason": ""})
            return await s.call_tool("get_customer_record", {"customer_id": "CUST-10002"})

        assert with_session(scenario).is_error is False


# =========================================================================== #
# 3. Raw wire: stdout must contain ONLY JSON-RPC frames
# =========================================================================== #
def _frames(*messages: dict | str) -> bytes:
    return ("\n".join(m if isinstance(m, str) else json.dumps(m) for m in messages) + "\n").encode()


INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "raw", "version": "0"}},
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _call(id_, name, args):
    return {"jsonrpc": "2.0", "id": id_, "method": "tools/call", "params": {"name": name, "arguments": args}}


class RawResult:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _run_raw(stdin: bytes, wait_for_ids: set[int], script: Path = SERVER_PY) -> RawResult:
    """Behave like a real MCP client: send frames, keep stdin OPEN until every expected
    response has arrived, then close stdin (the stdio shutdown signal) and collect the rest.

    (Closing stdin immediately after writing would make the server exit on EOF while
    requests are still in flight, which is correct server behaviour but a bad test.)
    """
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=script.parent,
    )
    watchdog = threading.Timer(20, proc.kill)  # a hung server must fail the test, not hang it
    watchdog.start()
    try:
        proc.stdin.write(stdin)
        proc.stdin.flush()
        seen_lines: list[bytes] = []
        pending = set(wait_for_ids)
        while pending:
            line = proc.stdout.readline()
            if not line:  # server died / closed stdout early
                break
            seen_lines.append(line)
            try:
                pending.discard(json.loads(line).get("id"))
            except ValueError:
                pass  # non-JSON line: the purity assertions will report it
        proc.stdin.close()
        rest, err = proc.communicate(timeout=10)
    finally:
        watchdog.cancel()
        proc.kill()
    return RawResult(b"".join(seen_lines) + rest, err, proc.returncode)


def _assert_pure_jsonrpc(stdout: bytes) -> list[dict]:
    lines = stdout.decode().splitlines()
    assert lines, "server produced no output"
    out = []
    for line in lines:
        msg = json.loads(line)  # raises if any non-JSON byte sneaks in
        assert msg["jsonrpc"] == "2.0", line
        assert ("result" in msg) != ("error" in msg), line
        out.append(msg)
    return out


class TestStdoutIsolation:
    SESSION = (
        INIT,
        INITIALIZED,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        _call(3, "get_customer_record", {"customer_id": "CUST-10001"}),
        _call(4, "get_customer_record", {"customer_id": "bad"}),
        _call(5, "trigger_refund", {"customer_id": "CUST-10001", "amount": 5, "reason": "valid reason here"}),
        _call(6, "trigger_refund", {"customer_id": "CUST-10001", "amount": 0, "reason": "short"}),
        _call(7, "nope", {}),
    )

    def test_stdout_is_pure_jsonrpc_and_logs_go_to_stderr(self):
        proc = _run_raw(_frames(*self.SESSION), wait_for_ids={1, 2, 3, 4, 5, 6, 7})
        assert proc.returncode == 0  # clean shutdown on stdin close
        frames = _assert_pure_jsonrpc(proc.stdout)
        by_id = {f["id"]: f for f in frames}
        assert set(by_id) == {1, 2, 3, 4, 5, 6, 7}
        assert "result" in by_id[3] and "result" in by_id[5]
        for bad in (4, 6, 7):
            assert by_id[bad]["error"]["code"] == INVALID_PARAMS
        # logs exist, and they are on stderr
        assert b"[task1-mcp]" in proc.stderr
        assert b"[task1-mcp]" not in proc.stdout

    def test_nan_literal_on_the_wire_is_rejected(self):
        # Python's json accepts NaN; strict JSON does not. Either way it must not be accepted as an amount.
        nan_call = '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"trigger_refund","arguments":{"customer_id":"CUST-10001","amount":NaN,"reason":"valid reason here"}}}'
        proc = _run_raw(
            _frames(INIT, INITIALIZED, nan_call, _call(10, "get_customer_record", {"customer_id": "CUST-10001"})),
            wait_for_ids={10},
        )
        frames = {f["id"]: f for f in _assert_pure_jsonrpc(proc.stdout)}
        assert "result" not in frames.get(9, {})  # either an error frame or dropped, never success
        assert "result" in frames[10]  # and the server is still alive afterwards

    def test_malformed_json_line_does_not_corrupt_stream_or_kill_server(self):
        proc = _run_raw(
            _frames(INIT, INITIALIZED, "{this is not json", _call(2, "get_customer_record", {"customer_id": "CUST-10001"})),
            wait_for_ids={2},
        )
        frames = _assert_pure_jsonrpc(proc.stdout)
        assert any(f.get("id") == 2 and "result" in f for f in frames)

    def test_server_py_contains_no_print_or_stdout_writes(self):
        tree = ast.parse(SERVER_PY.read_text())
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                offenders.append(f"print() at line {node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr == "stdout" and getattr(node.value, "id", "") == "sys":
                offenders.append(f"sys.stdout at line {node.lineno}")
        assert offenders == []

    def test_logging_handler_targets_stderr(self):
        # Checked in a fresh interpreter: pytest installs its own root handlers in-process,
        # which would hide what the real server process configures.
        code = (
            "import server, logging, sys\n"
            "hs = logging.getLogger().handlers\n"
            "assert hs, 'no logging handler configured'\n"
            "assert all(getattr(h, 'stream', None) is sys.stderr for h in hs), hs\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE, timeout=20)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == ""  # importing the server writes nothing to stdout


# =========================================================================== #
# 4. Hostile: a handler that prints must not break the wire
# =========================================================================== #
def test_stray_print_in_handler_cannot_corrupt_stdout(tmp_path):
    wrapper = tmp_path / "noisy_server.py"
    wrapper.write_text(
        textwrap.dedent(
            f"""
            import sys, asyncio
            sys.path.insert(0, {str(HERE)!r})
            import server

            _orig = server.dispatch_tool

            async def noisy(name, arguments):
                print("STRAY print() FROM HANDLER")
                sys.stdout.write("STRAY sys.stdout.write\\n")
                sys.stdout.flush()
                return await _orig(name, arguments)

            server.dispatch_tool = noisy
            asyncio.run(server.main())
            """
        )
    )
    proc = _run_raw(
        _frames(INIT, INITIALIZED, _call(2, "get_customer_record", {"customer_id": "CUST-10001"})),
        wait_for_ids={2},
        script=wrapper,
    )
    frames = _assert_pure_jsonrpc(proc.stdout)  # no STRAY text on stdout
    assert any(f.get("id") == 2 and "result" in f for f in frames)
    assert b"STRAY" in proc.stderr  # the stray output was diverted away from the wire
