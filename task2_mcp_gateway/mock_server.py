"""Tiny fake downstream MCP server (streamable HTTP, POST only) for testing the gateway.

It records every request it receives in app.state.calls, so tests can check that
blocked requests really never reached it.
"""
from __future__ import annotations

import json

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

TOOLS = [
    {"name": "get_status", "description": "Public status check", "inputSchema": {"type": "object"}},
    {"name": "admin_reset_key", "description": "Rotate an API key", "inputSchema": {"type": "object"}},
    {"name": "admin_delete_user", "description": "Delete a user", "inputSchema": {"type": "object"}},
    {"name": "stream_demo", "description": "Replies as an SSE stream", "inputSchema": {"type": "object"}},
]


def _ok(id_, result):
    return JSONResponse({"jsonrpc": "2.0", "id": id_, "result": result})


def _err(id_, code, msg):
    return JSONResponse({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": msg}})


async def mcp(request: Request):
    msg = await request.json()
    request.app.state.calls.append({"body": msg, "headers": dict(request.headers)})
    method, id_ = msg.get("method"), msg.get("id")

    if id_ is None:  # notification: accept and say nothing
        return Response(status_code=202)
    if method == "initialize":
        return _ok(id_, {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mock-downstream", "version": "0.1"},
        })
    if method == "ping":
        return _ok(id_, {})
    if method == "tools/list":
        return _ok(id_, {"tools": TOOLS})
    if method == "tools/call":
        name = msg["params"]["name"]
        if name == "stream_demo":
            async def events():
                yield b"event: message\ndata: " + json.dumps(
                    {"jsonrpc": "2.0", "id": id_, "result": {"content": [{"type": "text", "text": "chunk"}]}}
                ).encode() + b"\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        if name not in {t["name"] for t in TOOLS}:
            return _err(id_, -32602, f"Unknown tool: {name}")
        return _ok(id_, {"content": [{"type": "text", "text": f"{name} executed"}]})
    return _err(id_, -32601, "Method not found")


def create_mock_app() -> Starlette:
    app = Starlette(routes=[Route("/mcp", mcp, methods=["POST"])])
    app.state.calls = []
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_mock_app(), host="127.0.0.1", port=9000, log_level="info")
