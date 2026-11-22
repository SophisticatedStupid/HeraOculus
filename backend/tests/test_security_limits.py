"""Security hardening tests: API-key auth, rate limiting, and input caps."""

from __future__ import annotations

from collections import Counter

import pytest
from fastapi.testclient import TestClient
from slowapi.middleware import SlowAPIMiddleware

from app.main import app
from app.security import RateLimitMiddleware, limiter

client = TestClient(app)

_API_KEY = "test_heraoculus_key_123"

_SPATIAL_FDR_BODY = {
    "region_name": "T",
    "center_lat": 0.0,
    "center_lon": 0.0,
    "grid_rows": 5,
    "grid_cols": 5,
    "n_years": 8,
    "alpha": 0.05,
    "correction_method": "by",
}


@pytest.fixture(autouse=True)
def _isolate_rate_limits():
    """Reset slowapi storage around each test so rate-limit counters never leak
    between this file and the rest of the test suite."""
    limiter.reset()
    yield
    limiter.reset()


def _headers(with_key: bool = True, key: str = _API_KEY) -> dict:
    return {"X-API-Key": key} if with_key else {}


def _iter_route_endpoints(routes):
    """Yield every endpoint registered on ``app.routes``, descending into
    routers added via FastAPI ``include_router``."""
    for route in routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None:
            yield endpoint
            continue
        sub_routes = getattr(route, "routes", None)
        if sub_routes is None:
            original_router = getattr(route, "original_router", None)
            sub_routes = (
                getattr(original_router, "routes", None)
                if original_router is not None
                else None
            )
        if sub_routes:
            yield from _iter_route_endpoints(sub_routes)


# ---------------------------------------------------------------------------
# API-key auth
# ---------------------------------------------------------------------------


def test_401_without_api_key_when_configured(monkeypatch):
    """When HERAOCULUS_API_KEY is set, /api/v1/* (except health) requires X-API-Key."""
    monkeypatch.setenv("HERAOCULUS_API_KEY", _API_KEY)
    resp = client.post("/api/v1/stats/sarima")
    assert resp.status_code == 401, resp.text


def test_health_exempt_from_api_key(monkeypatch):
    """Health stays reachable without a key even when auth is configured."""
    monkeypatch.setenv("HERAOCULUS_API_KEY", _API_KEY)
    resp = client.get("/api/v1/health", headers=_headers(with_key=False))
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "ok"


def test_valid_api_key_allows_request(monkeypatch):
    """A correct X-API-Key passes auth when HERAOCULUS_API_KEY is configured."""
    monkeypatch.setenv("HERAOCULUS_API_KEY", _API_KEY)
    resp = client.post(
        "/api/v1/stats/sarima",
        headers=_headers(with_key=True),
        json={
            "dates": [f"2020-{m:02d}" for m in range(1, 25)],
            "values": [float(m) for m in range(24)],
            "forecast_steps": 3,
        },
    )
    assert resp.status_code != 401, resp.text


def test_dev_mode_allows_without_key(monkeypatch):
    """When HERAOCULUS_API_KEY is unset, requests are allowed (dev mode)."""
    monkeypatch.delenv("HERAOCULUS_API_KEY", raising=False)
    resp = client.post("/api/v1/stats/sarima")
    assert resp.status_code != 401, resp.text


def test_blank_api_key_keeps_dev_mode(monkeypatch):
    """A blank HERAOCULUS_API_KEY ("") behaves like unset: dev mode, no 401."""
    monkeypatch.setenv("HERAOCULUS_API_KEY", "")
    resp = client.post("/api/v1/stats/sarima")
    assert resp.status_code != 401, resp.text


def test_whitespace_api_key_keeps_dev_mode(monkeypatch):
    """A whitespace-only HERAOCULUS_API_KEY behaves like unset: dev mode."""
    monkeypatch.setenv("HERAOCULUS_API_KEY", "   ")
    resp = client.post("/api/v1/stats/sarima")
    assert resp.status_code != 401, resp.text


def test_real_key_requires_header_and_allows_with_it(monkeypatch):
    """A real configured key returns 401 without the header and 200 with it."""
    monkeypatch.setenv("HERAOCULUS_API_KEY", _API_KEY)
    # No header -> 401.
    resp = client.post("/api/v1/stats/sarima")
    assert resp.status_code == 401, resp.text
    # Correct header -> allowed past auth (not 401).
    resp = client.post(
        "/api/v1/stats/sarima",
        headers=_headers(with_key=True),
        json={
            "dates": [f"2020-{m:02d}" for m in range(1, 25)],
            "values": [float(m) for m in range(24)],
            "forecast_steps": 3,
        },
    )
    assert resp.status_code != 401, resp.text


def test_blank_cors_origins_denies_all_and_does_not_crash(monkeypatch):
    """A blank CORS_ORIGINS yields an empty allow-list (deny-all), not allow-all,
    and the app still serves requests without crashing."""
    import importlib

    import app.config as config_module

    monkeypatch.setenv("CORS_ORIGINS", "")
    importlib.reload(config_module)
    assert config_module.settings.cors_origins == [], config_module.settings.cors_origins

    # Whitespace-only is also treated as empty.
    monkeypatch.setenv("CORS_ORIGINS", "   ")
    importlib.reload(config_module)
    assert config_module.settings.cors_origins == [], config_module.settings.cors_origins

    # The app still serves requests (no crash).
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_429_after_exceeding_rate_limit(monkeypatch):
    """Exceeding the per-IP rate limit returns 429."""
    monkeypatch.delenv("HERAOCULUS_API_KEY", raising=False)
    responses = [
        client.post("/api/v1/stats/spatial-fdr", json=_SPATIAL_FDR_BODY)
        for _ in range(6)
    ]
    codes = [r.status_code for r in responses]
    # At least one request is allowed and at least one is rate-limited.
    assert 200 in codes, codes
    assert 429 in codes, codes


def test_default_limit_applies_to_undecorated_route(monkeypatch):
    """Undecorated routes (/api/v1/health) fall back to the default 60/minute."""
    monkeypatch.delenv("HERAOCULUS_API_KEY", raising=False)
    responses = [client.get("/api/v1/health") for _ in range(61)]
    codes = [r.status_code for r in responses]
    assert 200 in codes, codes
    assert responses[-1].status_code == 429, codes
    last = responses[-1]
    assert "Retry-After" in last.headers, last.headers
    assert last.json() == {"error": "Rate limit exceeded: 60 per 1 minute"}, last.text


def test_every_registered_route_has_a_rate_limit():
    """Every route in app.routes carries a limit (explicit or via default).

    Fails if default limits or the middleware are removed, or if a new route
    is added without a limit, so coverage can't silently regress.
    """
    assert limiter._default_limits, "default limits must cover undecorated routes"
    assert any(
        issubclass(m.cls, SlowAPIMiddleware) for m in app.user_middleware
    ), "SlowAPIMiddleware must be installed so default limits are enforced"

    for endpoint in _iter_route_endpoints(app.routes):
        name = f"{endpoint.__module__}.{endpoint.__name__}"
        assert name in limiter._route_limits or limiter._default_limits, (
            f"route {name} has no rate limit"
        )


def test_429_response_is_consistent_json(monkeypatch):
    """429 responses are JSON with a Retry-After header on any limited route."""
    monkeypatch.delenv("HERAOCULUS_API_KEY", raising=False)
    for _ in range(5):
        client.post("/api/v1/stats/spatial-fdr", json=_SPATIAL_FDR_BODY)
    resp = client.post("/api/v1/stats/spatial-fdr", json=_SPATIAL_FDR_BODY)
    assert resp.status_code == 429, resp.text
    assert resp.headers.get("content-type", "").startswith("application/json")
    body = resp.json()
    assert "error" in body
    assert resp.headers.get("Retry-After") is not None, resp.headers


# ---------------------------------------------------------------------------
# Client IP keying (proxy header trust)
# ---------------------------------------------------------------------------


def test_spoofed_forwarded_for_does_not_change_bucket_when_untrusted(monkeypatch):
    """Without TRUST_PROXY_HEADERS, X-Forwarded-For is ignored for keying."""
    monkeypatch.delenv("HERAOCULUS_API_KEY", raising=False)
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    headers = {"X-Forwarded-For": "1.2.3.4"}
    for _ in range(5):
        resp = client.post(
            "/api/v1/stats/spatial-fdr", json=_SPATIAL_FDR_BODY, headers=headers
        )
        assert resp.status_code == 200, resp.text
    # A different spoofed value must NOT allocate a fresh bucket.
    resp = client.post(
        "/api/v1/stats/spatial-fdr",
        json=_SPATIAL_FDR_BODY,
        headers={"X-Forwarded-For": "9.9.9.9"},
    )
    assert resp.status_code == 429, resp.text


def test_forwarded_for_creates_separate_buckets_when_trusted(monkeypatch):
    """With TRUST_PROXY_HEADERS, distinct X-Forwarded-For IPs get distinct buckets."""
    monkeypatch.delenv("HERAOCULUS_API_KEY", raising=False)
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    ip_a_headers = {"X-Forwarded-For": "203.0.113.10"}
    ip_b_headers = {"X-Forwarded-For": "203.0.113.20"}

    for _ in range(5):
        resp = client.post(
            "/api/v1/stats/spatial-fdr", json=_SPATIAL_FDR_BODY, headers=ip_a_headers
        )
        assert resp.status_code == 200, resp.text
    # Bucket A is exhausted.
    resp = client.post(
        "/api/v1/stats/spatial-fdr", json=_SPATIAL_FDR_BODY, headers=ip_a_headers
    )
    assert resp.status_code == 429, resp.text
    # Bucket B is independent and still fresh.
    resp = client.post(
        "/api/v1/stats/spatial-fdr", json=_SPATIAL_FDR_BODY, headers=ip_b_headers
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Input caps
# ---------------------------------------------------------------------------


def test_422_for_oversized_inputs(monkeypatch):
    """Oversized request lists are rejected with 422."""
    monkeypatch.delenv("HERAOCULUS_API_KEY", raising=False)
    oversized = [str(i) for i in range(31)]
    resp = client.post(
        "/api/v1/earthdata/granules",
        json={
            "short_name": "AIRS3STM",
            "bbox": [-10.0, 40.0, 20.0, 60.0],
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
            "max_results": 31,
        },
    )
    assert resp.status_code == 422, resp.text
