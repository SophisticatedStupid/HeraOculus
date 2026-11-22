"""API-key authentication and rate limiting for the HeraOculus backend.

Authentication
--------------
When ``HERAOCULUS_API_KEY`` is set, every ``/api/v1/*`` route except
``/api/v1/health`` requires an ``X-API-Key`` header compared with
``secrets.compare_digest``.  When the variable is unset the backend runs in
dev mode and all requests are allowed (one startup warning is logged).

Rate limiting
-------------
A shared :class:`slowapi.Limiter` is configured with tiered limits and
``slowapi.middleware.SlowAPIMiddleware`` must be installed on the app
(see ``app.main``) so the default limit covers every route, including
undecorated ones.

Limits
~~~~~~
``LIMIT_LIGHT`` (60/minute)   default, covers undecorated routes such as /health
``LIMIT_UPSTREAM`` (10/minute) routes that call external services using the
                                backend's own credentials
``LIMIT_HEAVY`` (5/minute)    CPU-heavy compute and file-upload routes

The limiter keys on the real client IP.  Behind a trusted reverse proxy
(e.g. Vercel) the ``X-Forwarded-For`` header carries the client IP and may be
trusted, but only when ``TRUST_PROXY_HEADERS=true``; otherwise the socket
peer address is used so clients cannot spoof ``X-Forwarded-For`` to dodge
rate limits.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status
from slowapi import Limiter
from slowapi.middleware import (
    SlowAPIMiddleware,
    _should_exempt,
    sync_check_limits,
)
from starlette.applications import Starlette
from starlette.routing import Match
from starlette.types import Scope

logger = logging.getLogger("heraoculus.security")

API_KEY_HEADER = "X-API-Key"

# Tiered rate limits (limit strings parsed by slowapi/limits).
LIMIT_LIGHT = "60/minute"  # default: read-only / health checks
LIMIT_UPSTREAM = "10/minute"  # routes proxying external services (our credentials)
LIMIT_HEAVY = "5/minute"  # CPU-heavy compute and file uploads


def _trust_proxy_headers() -> bool:
    """Return True only when the app runs behind a trusted reverse proxy."""
    return os.getenv("TRUST_PROXY_HEADERS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_client_ip(request: Request) -> str:
    """Return the client IP used for rate-limit buckets.

    When ``TRUST_PROXY_HEADERS`` is set, the real client IP is read from
    ``X-Forwarded-For`` (leftmost entry).  Our trusted reverse proxy
    (Vercel) populates that header from the TCP connection and overwrites
    any client-supplied value, so it cannot be spoofed.  Without the env
    var the socket peer address is used, because an untrusted
    ``X-Forwarded-For`` is trivial to forge and would let clients evade
    rate limits by inventing a new IP per request.
    """
    if _trust_proxy_headers():
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
            if client_ip:
                return client_ip
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


# Shared limiter instance used by main.py (exception handler) and routers (route limits).
# headers_enabled stays off: slowapi's wrapper injects headers on the raw
# endpoint return value, which FastAPI dict endpoints don't provide.
limiter = Limiter(key_func=get_client_ip, default_limits=[LIMIT_LIGHT])

TRUST_PROXY_HEADERS = _trust_proxy_headers()
logger.info(
    "Rate limiting configured: default=%s upstream=%s heavy=%s; "
    "TRUST_PROXY_HEADERS=%s",
    LIMIT_LIGHT,
    LIMIT_UPSTREAM,
    LIMIT_HEAVY,
    TRUST_PROXY_HEADERS,
)


def _find_fastapi_route_handler(routes, scope: Scope):
    """Resolve the endpoint for ``scope``, descending into routers registered
    via FastAPI ``include_router``.

    slowapi's built-in helper only inspects top-level ``app.routes`` and
    skips ``_IncludedRouter`` wrappers (they have no ``endpoint``), so the
    stock middleware would exempt every router-registered route from the
    default limit.
    """
    for route in routes:
        match, _ = route.matches(scope)
        if match != Match.FULL:
            continue
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None:
            return endpoint
        sub_routes = getattr(route, "routes", None)
        if sub_routes is None:
            original_router = getattr(route, "original_router", None)
            sub_routes = (
                getattr(original_router, "routes", None)
                if original_router is not None
                else None
            )
        if sub_routes:
            handler = _find_fastapi_route_handler(sub_routes, scope)
            if handler is not None:
                return handler
    return None


class RateLimitMiddleware(SlowAPIMiddleware):
    """SlowAPI middleware that finds route handlers through FastAPI's
    ``include_router`` nesting, so default limits cover every route."""

    async def dispatch(self, request, call_next):
        app: Starlette = request.app
        limiter: Limiter = app.state.limiter

        if not limiter.enabled:
            return await call_next(request)

        handler = _find_fastapi_route_handler(app.routes, request.scope)
        if _should_exempt(limiter, handler):
            return await call_next(request)

        error_response, should_inject_headers = sync_check_limits(
            limiter, request, handler, app
        )
        if error_response is not None:
            return error_response

        response = await call_next(request)
        if should_inject_headers:
            response = limiter._inject_headers(response, request.state.view_rate_limit)
        return response

def _configured_api_key() -> Optional[str]:
    """Return the configured API key, or None when dev mode is active.

    An unset, empty, or whitespace-only ``HERAOCULUS_API_KEY`` all mean dev
    mode: the key is not configured and requests are allowed without a header.
    This prevents a blank value (e.g. ``HERAOCULUS_API_KEY=""``) from silently
    enabling auth with an empty key.
    """
    value = os.getenv("HERAOCULUS_API_KEY")
    if value is None or not value.strip():
        return None
    return value.strip()


# Log exactly once at startup whether auth is enabled.
if _configured_api_key() is None:
    logger.warning(
        "HERAOCULUS_API_KEY is not set: running in DEV mode without API-key "
        "authentication. Set HERAOCULUS_API_KEY to require X-API-Key on /api/v1/* "
        "(except /api/v1/health)."
    )


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """FastAPI dependency: enforce X-API-Key when configured, allow all in dev.

    Reads ``HERAOCULUS_API_KEY`` per request so tests can flip it via monkeypatch
    without re-importing the application.
    """
    expected = _configured_api_key()
    if expected is None:
        # Dev mode: no key configured, allow all requests.
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
