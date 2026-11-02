"""Task 2 - MCP security gateway.

Sits between an agent and a downstream MCP server (streamable HTTP, JSON-RPC over POST).

  * Reads `Authorization: Bearer <jwt>` and pulls the role out of it.
  * tools/list            -> forwarded untouched.
  * tools/call admin_*    -> only role=admin gets through, otherwise the gateway answers
                             itself with JSON-RPC error -32001 and the downstream is never called.
  * everything else (initialize, ping, notifications, ...) is forwarded so a real client
    can still complete its handshake through us.

Run:  GATEWAY_JWT_SECRET=... DOWNSTREAM_URL=http://127.0.0.1:9000/mcp python gateway.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from auth import AuthError, role_from_header

# logs go to stderr, same habit as task 1
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s [gateway] %(message)s")
log = logging.getLogger("gateway")

# JSON-RPC standard codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
# gateway-specific codes (implementation-defined range)
UNAUTHORIZED_TOOL_CALL = -32001  # required by the task
AUTH_REQUIRED = -32002           # missing / bad bearer token
UPSTREAM_ERROR = -32003          # downstream unreachable or timed out

ADMIN_PREFIX = "admin_"
MAX_BODY_BYTES = 1_000_000

# Only these client headers go downstream. Notably NOT Authorization: the downstream
# should not receive the user's token, and an allow-list can't leak by accident.
FORWARD_REQUEST_HEADERS = {"content-type", "accept", "mcp-session-id", "mcp-protocol-version", "last-event-id"}
# httpx already decodes the body, so these would be wrong if we copied them back.
SKIP_RESPONSE_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}


def rpc_error(id_: Any, code: int, message: str, *, status: int = 200, data: Any = None) -> JSONResponse:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return JSONResponse({"jsonrpc": "2.0", "id": id_, "error": err}, status_code=status)


def is_admin_tool(name: str) -> bool:
    # casefold so "ADMIN_reset_key" can't sneak past if the downstream is case-insensitive
    return name.casefold().startswith(ADMIN_PREFIX)


class BodyTooLarge(Exception):
    pass


async def read_limited_body(request: Request) -> bytes:
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            raise BodyTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


async def forward(request: Request, msg: dict, id_: Any) -> Response:
    """Send `msg` downstream and stream whatever comes back (JSON or SSE) to the client."""
    state = request.app.state
    headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_REQUEST_HEADERS}
    headers["content-type"] = "application/json"
    # Re-serialise what WE parsed instead of forwarding the raw bytes. If the downstream
    # parsed duplicate keys differently from us, raw forwarding would let it run a
    # different tool than the one we authorised.
    upstream_req = state.client.build_request("POST", state.downstream_url, content=json.dumps(msg).encode(), headers=headers)
    try:
        upstream = await state.client.send(upstream_req, stream=True)
    except httpx.TimeoutException:
        log.warning("downstream timeout")
        return rpc_error(id_, UPSTREAM_ERROR, "Downstream server timed out", status=504)
    except httpx.HTTPError as exc:
        # log the real reason, tell the client nothing about our topology
        log.warning("downstream unreachable: %s", type(exc).__name__)
        return rpc_error(id_, UPSTREAM_ERROR, "Downstream server unavailable", status=502)

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in SKIP_RESPONSE_HEADERS}
    return StreamingResponse(
        upstream.aiter_bytes(),
        status_code=upstream.status_code,
        headers=resp_headers,
        background=BackgroundTask(upstream.aclose),
    )


async def handle_mcp(request: Request) -> Response:
    state = request.app.state

    # 1. who is calling?
    try:
        role = role_from_header(state.secret, request.headers.get("authorization"))
    except AuthError as exc:
        log.warning("auth rejected: %s", exc)
        resp = rpc_error(None, AUTH_REQUIRED, "Authentication required", status=401)
        resp.headers["WWW-Authenticate"] = 'Bearer realm="mcp-gateway"'
        return resp

    # 2. parse the JSON-RPC envelope
    try:
        raw = await read_limited_body(request)
    except BodyTooLarge:
        return rpc_error(None, INVALID_REQUEST, "Request body too large", status=413)
    try:
        msg = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return rpc_error(None, PARSE_ERROR, "Parse error", status=400)

    if isinstance(msg, list):
        # MCP dropped batching in 2025-06-18. Rejecting (instead of forwarding) also closes
        # the hole where an admin_ call hides inside a batch we never inspect.
        return rpc_error(None, INVALID_REQUEST, "Batch requests are not supported", status=400)
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return rpc_error(None, INVALID_REQUEST, "Invalid Request", status=400)

    id_ = msg.get("id")
    if "id" in msg and (isinstance(id_, bool) or not isinstance(id_, (str, int))):
        return rpc_error(None, INVALID_REQUEST, "Invalid Request: bad id", status=400)

    method = msg.get("method")
    if method is None:
        # a client *response* (e.g. to a server-initiated request): no tool can be invoked by it
        if "id" in msg and ("result" in msg or "error" in msg):
            return await forward(request, msg, id_)
        return rpc_error(id_ if "id" in msg else None, INVALID_REQUEST, "Invalid Request: missing method", status=400)
    if not isinstance(method, str):
        return rpc_error(None, INVALID_REQUEST, "Invalid Request: method must be a string", status=400)

    is_notification = "id" not in msg

    # 3. method-level authorisation
    if method == "tools/call":
        params = msg.get("params")
        name = params.get("name") if isinstance(params, dict) else None
        if not isinstance(name, str):
            if is_notification:
                return Response(status_code=202)
            return rpc_error(id_, INVALID_PARAMS, "Invalid params: params.name must be a string")
        if is_admin_tool(name) and role != "admin":
            log.warning("blocked tools/call %s for role=%s", name, role)
            if is_notification:
                return Response(status_code=202)  # notifications never get a reply, but still must not run
            return rpc_error(id_, UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call", data={"required_role": "admin"})
        log.info("tools/call %s allowed for role=%s", name, role)

    # tools/list and every other method just pass through
    return await forward(request, msg, id_)


async def healthz(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


def create_app(downstream_url: str, secret: str, client: httpx.AsyncClient | None = None) -> Starlette:
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=60, write=10, pool=5))

    @asynccontextmanager
    async def lifespan(app: Starlette):
        yield
        if own_client:
            await client.aclose()

    app = Starlette(
        routes=[Route("/mcp", handle_mcp, methods=["POST"]), Route("/healthz", healthz, methods=["GET"])],
        lifespan=lifespan,
    )
    app.state.client = client
    app.state.downstream_url = downstream_url
    app.state.secret = secret
    return app


def app_from_env() -> Starlette:
    secret = os.environ.get("GATEWAY_JWT_SECRET")
    if not secret:
        log.warning("GATEWAY_JWT_SECRET not set, using the insecure dev secret")
        secret = "dev-secret-change-me"
    return create_app(os.environ.get("DOWNSTREAM_URL", "http://127.0.0.1:9000/mcp"), secret)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app_from_env(), host="127.0.0.1", port=int(os.environ.get("PORT", "8080")), log_level="info")
