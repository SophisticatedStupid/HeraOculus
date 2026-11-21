"""API-key authentication for the Geo-Iris backend.

When ``GEOIRIS_API_KEY`` is set, every ``/api/v1/*`` route except
``/api/v1/health`` requires an ``X-API-Key`` header compared with
``secrets.compare_digest``.  When the variable is unset the backend runs in
dev mode and all requests are allowed (one startup warning is logged).
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger("geo_iris.security")

API_KEY_HEADER = "X-API-Key"

# Shared limiter instance used by main.py (exception handler) and routers (route limits).
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# Log exactly once at startup whether auth is enabled.
if os.getenv("GEOIRIS_API_KEY") is None:
    logger.warning(
        "GEOIRIS_API_KEY is not set: running in DEV mode without API-key "
        "authentication. Set GEOIRIS_API_KEY to require X-API-Key on /api/v1/* "
        "(except /api/v1/health)."
    )


def _configured_api_key() -> Optional[str]:
    """Return the configured API key, or None when dev mode is active."""
    return os.getenv("GEOIRIS_API_KEY")


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """FastAPI dependency: enforce X-API-Key when configured, allow all in dev.

    Reads ``GEOIRIS_API_KEY`` per request so tests can flip it via monkeypatch
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
