"""Task 4 - model router: token rate limit -> primary provider -> automatic failover -> clean errors.

Request life cycle (ModelRouter.complete):

    1. estimate the token cost (prompt chars/4 + max_tokens) and RESERVE it in SQLite
         over budget            -> 429 rate_limit_exceeded (+ Retry-After), providers are never called
    2. call providers in order, each with its own hard deadline (primary: 3000 ms)
         429 / timeout / 5xx / connection error / garbage body  -> next provider
         400 / 413 / 422 (the request itself is bad)             -> stop, tell the client, no failover
    3. success  -> replace the estimate with the provider's real usage, return the body
       failure  -> release the reservation (nobody was served, nobody pays), return a sanitised error

Nothing from a provider failure (status text, body, stack trace, URL, model name) ever reaches the client.
It is logged with a request id, and the client gets the same id so support can find the log line.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
from dataclasses import dataclass, field
from typing import Any

import httpx

from rate_limiter import Decision, SlidingWindowRateLimiter

log = logging.getLogger("llm-router")

DEFAULT_MAX_TOKENS = 1024        # reserved for completion when the client gives no max_tokens
PER_MESSAGE_OVERHEAD = 4         # role / separators, same rule of thumb OpenAI documents
CLIENT_ERROR_STATUSES = {400, 413, 422}


# ---------------------------------------------------------------------- errors
class GatewayError(Exception):
    """Anything that should turn into the standard error body. Messages here are written for clients."""

    def __init__(self, status: int, type_: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status, self.type, self.message, self.headers = status, type_, message, headers or {}

    def body(self, request_id: str) -> dict[str, Any]:
        # The ONE error shape of this gateway. Exactly these three keys, always.
        return {"error": {"type": self.type, "message": self.message, "request_id": request_id}}


class UpstreamFailure(Exception):
    """Internal only: why one provider attempt failed. Never serialised to a client."""

    def __init__(self, kind: str, detail: str = "", client_error: bool = False):
        super().__init__(f"{kind} {detail}".strip())
        self.kind, self.detail, self.client_error = kind, detail, client_error


# ---------------------------------------------------------------------- config / results
@dataclass(frozen=True)
class Provider:
    name: str                    # logical name, used in logs only
    url: str                     # full chat-completions URL
    api_key: str | None = None
    model: str | None = None     # overrides the client's model (providers name models differently)
    timeout_s: float = 10.0


@dataclass
class RouterResult:
    data: dict[str, Any]
    provider: str
    failed_over: bool
    limit: int
    used: int
    attempts: list[str] = field(default_factory=list)


def new_request_id() -> str:
    return "gw-" + secrets.token_hex(8)


# ---------------------------------------------------------------------- token estimation
def estimate_tokens(body: dict[str, Any], default_max_tokens: int = DEFAULT_MAX_TOKENS) -> int:
    """Cheap upper-ish estimate used for the reservation: ceil(prompt chars / 4) + completion budget."""
    chars = 0
    for msg in body["messages"]:
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif content is not None:  # multi-part content (images, tool calls...): count its JSON size
            chars += len(json.dumps(content))
        chars += PER_MESSAGE_OVERHEAD * 4
    prompt = math.ceil(chars / 4)
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        max_tokens = default_max_tokens
    return prompt + max_tokens


def validate_body(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise GatewayError(400, "invalid_request", "Body must be a JSON object")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) for m in msgs):
        raise GatewayError(400, "invalid_request", "messages must be a non-empty array of objects")
    if body.get("stream") is True:
        raise GatewayError(400, "invalid_request", "Streaming is not supported on this endpoint")
    return body


# ---------------------------------------------------------------------- router
class ModelRouter:
    def __init__(
        self,
        providers: list[Provider],
        limiter: SlidingWindowRateLimiter,
        client: httpx.AsyncClient | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self.limiter = limiter
        # No httpx-level timeout: a single hard deadline per attempt (asyncio.timeout) covers connect,
        # send, wait AND reading the body. httpx's own timeouts are per-phase, so a provider that
        # trickles bytes could otherwise outlive the 3000 ms budget.
        self.client = client or httpx.AsyncClient(
            timeout=None, limits=httpx.Limits(max_connections=200, max_keepalive_connections=50)
        )
        self.default_max_tokens = default_max_tokens

    # ---- one provider attempt
    async def _call(self, provider: Provider, body: dict[str, Any]) -> dict[str, Any]:
        payload = dict(body)
        if provider.model:
            payload["model"] = provider.model
        headers = {"content-type": "application/json"}
        if provider.api_key:
            headers["authorization"] = f"Bearer {provider.api_key}"
        try:
            async with asyncio.timeout(provider.timeout_s):
                resp = await self.client.post(provider.url, content=json.dumps(payload).encode(), headers=headers)
        except (TimeoutError, httpx.TimeoutException):
            raise UpstreamFailure("timeout", f"> {provider.timeout_s}s") from None
        except httpx.HTTPError as exc:  # connection refused, DNS, reset, protocol error...
            raise UpstreamFailure("unavailable", type(exc).__name__) from None

        status = resp.status_code
        if status == 429:
            raise UpstreamFailure("rate_limited", "429")
        if status in CLIENT_ERROR_STATUSES:
            raise UpstreamFailure("client_error", str(status), client_error=True)
        if status != 200:  # 5xx, but also 401/403/404 = our provider config is wrong -> try the backup
            raise UpstreamFailure("error", str(status))
        try:
            data = resp.json()
        except ValueError:
            raise UpstreamFailure("bad_response", "not json") from None
        if not isinstance(data, dict):
            raise UpstreamFailure("bad_response", "not an object")
        return data

    @staticmethod
    def _actual_tokens(data: dict[str, Any], estimate: int) -> int:
        usage = data.get("usage")
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
            return total
        return estimate  # provider reported nothing: keep what we reserved (safe direction)

    # ---- full request
    async def complete(
        self, key_hash: str, body: dict[str, Any], limit: int | None = None, request_id: str | None = None
    ) -> RouterResult:
        request_id = request_id or new_request_id()
        body = validate_body(body)
        cost = estimate_tokens(body, self.default_max_tokens)

        decision: Decision = await self.limiter.reserve(key_hash, cost, limit)
        if decision.too_large:
            raise GatewayError(
                400, "request_too_large",
                f"Estimated size ({cost} tokens) exceeds the per-minute token limit ({decision.limit}). "
                "Shorten the prompt or lower max_tokens.",
            )
        if not decision.allowed:
            log.info("[%s] rate limited tenant=%s used=%s cost=%s", request_id, key_hash[:8], decision.used, cost)
            raise GatewayError(
                429, "rate_limit_exceeded",
                f"Token rate limit of {decision.limit} per minute reached. Retry in {decision.retry_after_s}s.",
                headers={"Retry-After": str(decision.retry_after_s)},
            )

        settled = False
        try:
            attempts: list[str] = []
            data, served_by = None, None
            for provider in self.providers:
                try:
                    data = await self._call(provider, body)
                    served_by = provider.name
                    break
                except UpstreamFailure as f:
                    attempts.append(f"{provider.name}:{f.kind}")
                    log.warning("[%s] provider=%s failed: %s", request_id, provider.name, f)
                    if f.client_error:  # our request is the problem; another provider will not fix it
                        raise GatewayError(400, "invalid_request", "The request was rejected as invalid") from None
            if data is None:
                raise GatewayError(
                    503, "providers_unavailable",
                    "No model provider could serve the request right now. Please retry shortly.",
                    headers={"Retry-After": "5"},
                )

            actual = self._actual_tokens(data, cost)
            used = await asyncio.shield(self.limiter.settle(decision.reservation_id, actual, key_hash))
            settled = True
            if len(attempts) > 0:
                log.info("[%s] served by %s after %s", request_id, served_by, attempts)
            return RouterResult(data, served_by, failed_over=bool(attempts), limit=decision.limit,
                                used=used, attempts=attempts)
        finally:
            if not settled:
                # Also runs on CancelledError (client went away): never leak a reservation.
                # shield() lets the DB cleanup finish even while this task is being cancelled.
                try:
                    await asyncio.shield(self.limiter.release(decision.reservation_id))
                except Exception:
                    log.exception("[%s] could not release reservation %s", request_id, decision.reservation_id)

    async def aclose(self) -> None:
        await self.client.aclose()
