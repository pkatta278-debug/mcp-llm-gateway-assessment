"""Shared test helper: run an ASGI app as a real HTTP server on a free port."""
import asyncio
import contextlib

import uvicorn


@contextlib.asynccontextmanager
async def serve(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="off",
                                          timeout_graceful_shutdown=0.2))  # do not wait for deliberately hung mock handlers
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task
