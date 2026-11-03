"""Task 4 - LLM gateway endpoint with token rate limiting and model failover.

POST /v1/chat/completions   (OpenAI-compatible, non-streaming)
    Authorization: Bearer <tenant api key>

Run (see README for the full demo):
    TENANT_KEYS=demo-key:50000 PRIMARY_URL=http://127.0.0.1:9100/v1/chat/completions \
    SECONDARY_URL=http://127.0.0.1:9101/v1/chat/completions python gateway.py

Config (env):
    TENANT_KEYS         "key1:50000,key2"   api keys, optional per-key tokens/minute (default 50000)
    PRIMARY_URL / PRIMARY_API_KEY / PRIMARY_MODEL / PRIMARY_TIMEOUT_MS (default 3000)
    SECONDARY_URL / SECONDARY_API_KEY / SECONDARY_MODEL / SECONDARY_TIMEOUT_MS (default 10000)
    DB_PATH             sqlite file for rate limit state (default ./ratelimit.db)
    PORT                default 8082
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from rate_limiter import DEFAULT_LIMIT, SlidingWindowRateLimiter
from router import GatewayError, ModelRouter, Provider, new_request_id

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s [llm-router] %(message)s")
log = logging.getLogger("llm-router")
logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx logs full provider URLs at INFO; keep them out of logs

MAX_BODY_BYTES = 1_000_000


def hash_key(api_key: str) -> str:
    """Tenant identity inside the gateway. The raw key is never stored or logged."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def error_response(err: GatewayError, request_id: str) -> JSONResponse:
    headers = {"X-Request-Id": request_id, **err.headers}
    return JSONResponse(err.body(request_id), status_code=err.status, headers=headers)


async def read_limited(request: Request) -> bytes:
    chunks, total = [], 0
    async for c in request.stream():
        total += len(c)
        if total > MAX_BODY_BYTES:
            raise GatewayError(413, "invalid_request", "Request body must be under 1 MB")
        chunks.append(c)
    return b"".join(chunks)


async def chat_completions(request: Request) -> Response:
    request_id = new_request_id()
    st = request.app.state
    try:
        auth = request.headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        key_hash = hash_key(token.strip()) if scheme.lower() == "bearer" and token.strip() else None
        if key_hash is None or key_hash not in st.tenants:
            raise GatewayError(401, "unauthorized", "Missing or invalid API key", {"WWW-Authenticate": "Bearer"})

        try:
            body = json.loads(await read_limited(request))
        except ValueError:
            raise GatewayError(400, "invalid_request", "Request body must be valid JSON") from None

        result = await st.router.complete(key_hash, body, limit=st.tenants[key_hash], request_id=request_id)
        return JSONResponse(
            result.data,
            headers={
                "X-Request-Id": request_id,
                "X-Gateway-Failover": "1" if result.failed_over else "0",
                "X-RateLimit-Limit-Tokens": str(result.limit),
                "X-RateLimit-Remaining-Tokens": str(max(0, result.limit - result.used)),
            },
        )
    except GatewayError as err:
        return error_response(err, request_id)
    except Exception:
        # A bug or a DB failure. Full detail goes to the log; the client gets nothing internal.
        log.exception("[%s] unhandled error", request_id)
        return error_response(GatewayError(500, "internal_error", "The gateway could not process the request"), request_id)


def parse_tenants(spec: str) -> dict[str, int]:
    """'keyA:100000,keyB' -> {sha256(keyA): 100000, sha256(keyB): 50000}"""
    out: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        key, _, lim = part.partition(":")
        out[hash_key(key)] = int(lim) if lim else DEFAULT_LIMIT
    return out


def create_app(router: ModelRouter, tenants: dict[str, int]) -> Starlette:
    """tenants maps sha256(api key) -> tokens per minute."""
    app = Starlette(routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])])
    app.state.router = router
    app.state.tenants = tenants
    return app


def providers_from_env() -> list[Provider]:
    out = []
    for prefix, default_ms in (("PRIMARY", 3000), ("SECONDARY", 10000)):
        url = os.environ.get(f"{prefix}_URL")
        if not url:
            continue
        out.append(Provider(
            name=prefix.lower(), url=url,
            api_key=os.environ.get(f"{prefix}_API_KEY"), model=os.environ.get(f"{prefix}_MODEL"),
            timeout_s=int(os.environ.get(f"{prefix}_TIMEOUT_MS", default_ms)) / 1000,
        ))
    return out


if __name__ == "__main__":
    import uvicorn

    providers = providers_from_env()
    if not providers:
        sys.exit("set PRIMARY_URL (and SECONDARY_URL for failover)")
    limiter = SlidingWindowRateLimiter(os.environ.get("DB_PATH", "ratelimit.db"))
    tenants = parse_tenants(os.environ.get("TENANT_KEYS", "demo-key"))
    uvicorn.run(create_app(ModelRouter(providers, limiter), tenants),
                host="127.0.0.1", port=int(os.environ.get("PORT", "8082")), log_level="info")
