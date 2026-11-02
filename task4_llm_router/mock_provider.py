"""Fake OpenAI-style provider (non-streaming) whose behaviour is driven by app.state.script.

  status        return this HTTP status with a body that contains a fake stack trace
  hang          seconds to sleep before answering (simulates a stuck provider)
  invalid_json  answer 200 with a body that is not JSON
  usage_total   total_tokens to report in usage (omit the usage block if None)
  content       assistant text
  name          echoed in the answer so tests can tell which provider served it
"""
from __future__ import annotations

import asyncio

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route


async def completions(request: Request):
    st = request.app.state
    body = await request.json()
    st.requests.append({"body": body, "headers": dict(request.headers)})
    st.inflight += 1
    try:
        sc = st.script
        if sc.get("hang"):
            await asyncio.sleep(sc["hang"])
        if sc.get("status"):
            return JSONResponse(
                {"error": {"message": "Traceback (most recent call last): File '/srv/provider/internal.py', line 88"}},
                status_code=sc["status"],
            )
        if sc.get("invalid_json"):
            return Response("<html>upstream proxy error</html>", media_type="application/json")
        out = {
            "id": "chatcmpl-mock", "object": "chat.completion", "model": body.get("model"),
            "served_by": sc.get("name", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": sc.get("content", "hello")},
                         "finish_reason": "stop"}],
        }
        if sc.get("usage_total") is not None:
            out["usage"] = {"total_tokens": sc["usage_total"]}
        return JSONResponse(out)
    finally:
        st.inflight -= 1


async def set_script(request: Request):
    """Control endpoint for manual demos: POST a JSON script to change behaviour, e.g. {"hang": 10}."""
    request.app.state.script = await request.json()
    return JSONResponse({"ok": True})


def create_mock_app(name: str = "mock") -> Starlette:
    app = Starlette(routes=[Route("/v1/chat/completions", completions, methods=["POST"]),
                         Route("/_script", set_script, methods=["POST"])])
    app.state.script = {"name": name}
    app.state.requests = []
    app.state.inflight = 0
    return app


if __name__ == "__main__":
    import sys

    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9100
    uvicorn.run(create_mock_app(f"mock-{port}"), host="127.0.0.1", port=port, log_level="info")
