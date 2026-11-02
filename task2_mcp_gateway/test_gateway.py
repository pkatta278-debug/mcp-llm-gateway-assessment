"""Gateway tests. Everything runs in-process: the gateway talks to the mock downstream
through httpx's ASGI transport, so no ports and no subprocesses."""
import json
import time

import httpx
import pytest

import auth
from gateway import create_app
from mock_server import create_mock_app

SECRET = "test-secret"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mock():
    return create_mock_app()


@pytest.fixture
async def gw(mock):
    downstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock), base_url="http://downstream")
    app = create_app("http://downstream/mcp", SECRET, client=downstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        yield c
    await downstream.aclose()


def bearer(role="viewer", **kw):
    return {"Authorization": "Bearer " + auth.mint_token(SECRET, role, **kw)}


def rpc(method, params=None, id=1):
    m = {"jsonrpc": "2.0", "method": method}
    if id is not None:
        m["id"] = id
    if params is not None:
        m["params"] = params
    return m


def call(name, id=1):
    return rpc("tools/call", {"name": name, "arguments": {}}, id)


# ---------------------------------------------------------------- forwarding
async def test_tools_list_is_forwarded_for_viewer(gw, mock):
    r = await gw.post("/mcp", json=rpc("tools/list"), headers=bearer("viewer"))
    assert r.status_code == 200
    names = [t["name"] for t in r.json()["result"]["tools"]]
    assert "admin_reset_key" in names  # transparent: not filtered
    assert len(mock.state.calls) == 1
    assert mock.state.calls[0]["body"] == rpc("tools/list")


async def test_viewer_can_call_normal_tool(gw, mock):
    r = await gw.post("/mcp", json=call("get_status"), headers=bearer("viewer"))
    assert r.json()["result"]["content"][0]["text"] == "get_status executed"
    assert len(mock.state.calls) == 1


async def test_initialize_and_ping_pass_through(gw, mock):
    r = await gw.post("/mcp", json=rpc("initialize", {}), headers=bearer("viewer"))
    assert r.json()["result"]["serverInfo"]["name"] == "mock-downstream"
    r = await gw.post("/mcp", json=rpc("ping", id=2), headers=bearer("viewer"))
    assert r.json()["id"] == 2


async def test_admin_can_call_admin_tool(gw, mock):
    r = await gw.post("/mcp", json=call("admin_reset_key"), headers=bearer("admin"))
    assert r.json()["result"]["content"][0]["text"] == "admin_reset_key executed"
    assert len(mock.state.calls) == 1


# ------------------------------------------------------------ authorisation
async def test_viewer_blocked_from_admin_tool_and_downstream_never_called(gw, mock):
    r = await gw.post("/mcp", json=call("admin_reset_key", id=42), headers=bearer("viewer"))
    body = r.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 42
    assert body["error"]["code"] == -32001
    assert body["error"]["message"] == "Unauthorized Tool Call"
    assert "result" not in body
    assert mock.state.calls == []


@pytest.mark.parametrize("name", ["admin_", "admin_delete_user", "ADMIN_reset_key", "Admin_x"])
async def test_prefix_variants_blocked(gw, mock, name):
    r = await gw.post("/mcp", json=call(name), headers=bearer("viewer"))
    assert r.json()["error"]["code"] == -32001
    assert mock.state.calls == []


async def test_name_that_only_contains_admin_is_not_blocked(gw, mock):
    r = await gw.post("/mcp", json=call("get_admin_status"), headers=bearer("viewer"))
    assert "error" not in r.json() or r.json()["error"]["code"] != -32001  # mock says unknown tool, gateway let it through
    assert len(mock.state.calls) == 1


async def test_unknown_role_is_treated_as_viewer(gw, mock):
    r = await gw.post("/mcp", json=call("admin_reset_key"), headers=bearer("superuser"))
    assert r.json()["error"]["code"] == -32001
    assert mock.state.calls == []


async def test_blocked_notification_gets_202_and_is_not_forwarded(gw, mock):
    r = await gw.post("/mcp", json=call("admin_reset_key", id=None), headers=bearer("viewer"))
    assert r.status_code == 202 and r.content == b""
    assert mock.state.calls == []


async def test_non_string_tool_name_is_invalid_params(gw, mock):
    for bad in (None, 123, ["admin_reset_key"], {"a": 1}):
        r = await gw.post("/mcp", json=rpc("tools/call", {"name": bad}), headers=bearer("viewer"))
        assert r.json()["error"]["code"] == -32602
    r = await gw.post("/mcp", json=rpc("tools/call"), headers=bearer("viewer"))  # no params at all
    assert r.json()["error"]["code"] == -32602
    assert mock.state.calls == []


# ------------------------------------------------- parser-differential tricks
async def test_duplicate_key_last_wins_and_what_we_checked_is_what_runs(gw, mock):
    # json.loads keeps the LAST duplicate. Forward re-serialised JSON so downstream sees the same thing.
    h = {**bearer("viewer"), "content-type": "application/json"}
    ok = '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key","name":"get_status"}}'
    r = await gw.post("/mcp", content=ok, headers=h)
    assert "error" not in r.json()
    assert mock.state.calls[-1]["body"]["params"]["name"] == "get_status"  # not admin_reset_key

    mock.state.calls.clear()
    bad = '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_status","name":"admin_reset_key"}}'
    r = await gw.post("/mcp", content=bad, headers=h)
    assert r.json()["error"]["code"] == -32001
    assert mock.state.calls == []


async def test_batch_with_admin_call_is_rejected_not_forwarded(gw, mock):
    batch = [rpc("tools/list", id=1), call("admin_reset_key", id=2)]
    r = await gw.post("/mcp", json=batch, headers=bearer("viewer"))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32600
    assert mock.state.calls == []


# ------------------------------------------------------------- authentication
async def test_missing_token_is_401(gw, mock):
    r = await gw.post("/mcp", json=rpc("tools/list"))
    assert r.status_code == 401
    assert r.json()["error"]["code"] == -32002
    assert r.headers["www-authenticate"].startswith("Bearer")
    assert mock.state.calls == []


@pytest.mark.parametrize("value", ["Basic abc", "Bearer", "Bearer ", "Bearer not.a.jwt", "token xyz", ""])
async def test_garbage_authorization_is_401(gw, mock, value):
    r = await gw.post("/mcp", json=rpc("tools/list"), headers={"Authorization": value})
    assert r.status_code == 401
    assert mock.state.calls == []


async def test_wrong_secret_expired_and_alg_none_are_401(gw, mock):
    wrong = auth.mint_token("other-secret", "admin")
    expired = auth.mint_token(SECRET, "admin", ttl=-10)
    # forge alg=none with role=admin
    b = lambda d: auth._b64e(json.dumps(d).encode())
    none_tok = f'{b({"alg": "none"})}.{b({"role": "admin", "exp": int(time.time()) + 999})}.'
    for tok in (wrong, expired, none_tok):
        r = await gw.post("/mcp", json=call("admin_reset_key"), headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 401, tok
    assert mock.state.calls == []


async def test_tampered_role_claim_fails_signature(gw, mock):
    tok = auth.mint_token(SECRET, "viewer")
    h, p, s = tok.split(".")
    claims = json.loads(auth._b64d(p))
    claims["role"] = "admin"
    forged = f"{h}.{auth._b64e(json.dumps(claims).encode())}.{s}"
    r = await gw.post("/mcp", json=call("admin_reset_key"), headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401


async def test_bearer_scheme_is_case_insensitive(gw):
    tok = auth.mint_token(SECRET, "viewer")
    r = await gw.post("/mcp", json=rpc("tools/list"), headers={"Authorization": f"bearer {tok}"})
    assert r.status_code == 200


async def test_authorization_header_not_leaked_downstream(gw, mock):
    await gw.post("/mcp", json=rpc("tools/list"), headers=bearer("admin"))
    assert "authorization" not in mock.state.calls[0]["headers"]


# ------------------------------------------------------------ wire-format errors
async def test_invalid_json_is_parse_error(gw, mock):
    r = await gw.post("/mcp", content=b"{not json", headers={**bearer(), "content-type": "application/json"})
    assert r.status_code == 400
    assert r.json() == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
    assert mock.state.calls == []


@pytest.mark.parametrize("payload", [
    {"id": 1, "method": "tools/list"},                          # no jsonrpc
    {"jsonrpc": "1.0", "id": 1, "method": "tools/list"},        # wrong version
    {"jsonrpc": "2.0", "id": 1},                                # no method
    {"jsonrpc": "2.0", "id": 1, "method": 5},                   # method not a string
    {"jsonrpc": "2.0", "id": {"x": 1}, "method": "tools/list"}, # id wrong type
    {"jsonrpc": "2.0", "id": True, "method": "tools/list"},     # bool is not an id
    "just a string", 7,
])
async def test_invalid_requests(gw, mock, payload):
    r = await gw.post("/mcp", json=payload, headers=bearer())
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32600
    assert mock.state.calls == []


async def test_string_and_int_ids_are_echoed(gw):
    for i in ("abc", 0, 99):
        r = await gw.post("/mcp", json=call("admin_reset_key", id=i), headers=bearer("viewer"))
        assert r.json()["id"] == i


async def test_oversized_body_is_413(gw, mock):
    big = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"pad": "x" * 1_100_000}}
    r = await gw.post("/mcp", json=big, headers=bearer())
    assert r.status_code == 413
    assert mock.state.calls == []


async def test_only_post_allowed(gw):
    assert (await gw.get("/mcp", headers=bearer())).status_code == 405


# ------------------------------------------------------------- downstream trouble
async def _gw_with_transport(handler):
    downstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app("http://internal-host.corp:9000/mcp", SECRET, client=downstream)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


async def test_downstream_down_gives_sanitised_502():
    def boom(request):
        raise httpx.ConnectError("[Errno 111] Connection refused internal-host.corp:9000")
    async with await _gw_with_transport(boom) as c:
        r = await c.post("/mcp", json=rpc("tools/list", id=5), headers=bearer())
    assert r.status_code == 502
    assert r.json()["id"] == 5 and r.json()["error"]["code"] == -32003
    assert "internal-host" not in r.text and "111" not in r.text


async def test_downstream_timeout_gives_504():
    def slow(request):
        raise httpx.ReadTimeout("took too long")
    async with await _gw_with_transport(slow) as c:
        r = await c.post("/mcp", json=rpc("tools/list"), headers=bearer())
    assert r.status_code == 504
    assert r.json()["error"]["code"] == -32003


async def test_sse_response_is_passed_through(gw):
    r = await gw.post("/mcp", json=call("stream_demo"), headers={**bearer(), "Accept": "application/json, text/event-stream"})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert b"event: message" in r.content and b'"chunk"' in r.content


async def test_empty_body_is_parse_error(gw, mock):
    r = await gw.post("/mcp", content=b"", headers=bearer())
    assert r.status_code == 400 and r.json()["error"]["code"] == -32700
    assert mock.state.calls == []
