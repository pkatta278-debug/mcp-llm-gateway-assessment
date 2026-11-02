"""Bearer token handling for the gateway.

Tokens are plain HS256 JWTs with a "role" claim. I wrote the verify step by hand
(hmac + base64) so the gateway has no extra dependency and every check is visible.

    python auth.py mint --role admin     # prints a token for manual testing
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import time

KNOWN_ROLES = {"admin", "viewer"}


class AuthError(Exception):
    """Token missing/invalid. The message is safe to log, NOT sent to the client."""


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def mint_token(secret: str, role: str, sub: str = "demo-user", ttl: int = 3600) -> str:
    header = _b64e(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    claims = {"sub": sub, "role": role, "exp": int(time.time()) + ttl}
    payload = _b64e(json.dumps(claims, separators=(",", ":")).encode())
    sig = hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64e(sig)}"


def verify_token(secret: str, token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("not a JWT")
    h_b64, p_b64, s_b64 = parts
    try:
        header = json.loads(_b64d(h_b64))
        claims = json.loads(_b64d(p_b64))
        given_sig = _b64d(s_b64)
    except Exception:
        raise AuthError("undecodable token") from None

    # Pin the algorithm. Never trust the token to tell us how to verify it
    # (that's how the classic alg=none bypass works).
    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise AuthError("unsupported alg")
    if not isinstance(claims, dict):
        raise AuthError("bad claims")

    expected = hmac.new(secret.encode(), f"{h_b64}.{p_b64}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, given_sig):
        raise AuthError("bad signature")

    exp = claims.get("exp")
    if exp is not None:
        if isinstance(exp, bool) or not isinstance(exp, (int, float)) or exp < time.time():
            raise AuthError("expired or bad exp")
    return claims


def role_from_header(secret: str, authorization: str | None) -> str:
    """Return the role for an Authorization header value, or raise AuthError."""
    if not authorization:
        raise AuthError("no Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("not a Bearer token")
    role = verify_token(secret, token.strip()).get("role")
    # Fail closed: a validly signed token with a weird role is just "viewer".
    return role if isinstance(role, str) and role in KNOWN_ROLES else "viewer"


if __name__ == "__main__":
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mint"])
    ap.add_argument("--role", default="viewer")
    ap.add_argument("--ttl", type=int, default=3600)
    a = ap.parse_args()
    print(mint_token(os.environ.get("GATEWAY_JWT_SECRET", "dev-secret-change-me"), a.role, ttl=a.ttl))
