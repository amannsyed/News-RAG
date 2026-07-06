from __future__ import annotations

import os
import time
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request


@dataclass
class TokenBucket:
    tokens: float
    updated_at: float


_buckets: dict[str, TokenBucket] = {}


def require_auth(authorization: str | None = Header(default=None)) -> str:
    expected = os.getenv("RAG_API_TOKEN", "").strip()
    if not expected:
        # Allow explicit opt-in to unauthenticated access via RAG_ALLOW_ANONYMOUS=true.
        # Without this flag the API refuses all requests when no token is configured,
        # preventing an accidental open deployment.
        if os.getenv("RAG_ALLOW_ANONYMOUS", "").strip().lower() not in {"1", "true", "yes"}:
            raise HTTPException(
                status_code=503,
                detail="RAG_API_TOKEN is not configured. Set it or set RAG_ALLOW_ANONYMOUS=true to permit unauthenticated access.",
            )
        return "anonymous"
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="invalid bearer token")
    return "token-user"


def check_rate_limit(request: Request, identity: str) -> None:
    rate_per_minute = int(os.getenv("RAG_RATE_LIMIT_PER_MINUTE", "60"))
    burst = int(os.getenv("RAG_RATE_LIMIT_BURST", "10"))
    now = time.monotonic()
    key = identity if identity != "anonymous" else (request.client.host if request.client else "anonymous")
    bucket = _buckets.get(key)
    refill_rate = rate_per_minute / 60.0
    if bucket is None:
        bucket = TokenBucket(tokens=float(burst), updated_at=now)
    elapsed = max(0.0, now - bucket.updated_at)
    bucket.tokens = min(float(burst), bucket.tokens + elapsed * refill_rate)
    bucket.updated_at = now
    if bucket.tokens < 1.0:
        _buckets[key] = bucket
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    bucket.tokens -= 1.0
    _buckets[key] = bucket
