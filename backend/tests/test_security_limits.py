"""Security hardening tests: API-key auth, rate limiting, and input caps."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.security import limiter

client = TestClient(app)

_API_KEY = "test_heraoculus_key_123"


@pytest.fixture(autouse=True)
def _isolate_rate_limits():
    """Reset slowapi storage around each test so rate-limit counters never leak
    between this file and the rest of the test suite."""
    limiter.reset()
    yield
    limiter.reset()


def _headers(with_key: bool = True, key: str = _API_KEY) -> dict:
    return {"X-API-Key": key} if with_key else {}


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


def test_429_after_exceeding_rate_limit(monkeypatch):
    """Exceeding the per-IP rate limit returns 429."""
    monkeypatch.delenv("HERAOCULUS_API_KEY", raising=False)
    body = {
        "region_name": "T",
        "center_lat": 0.0,
        "center_lon": 0.0,
        "grid_rows": 5,
        "grid_cols": 5,
        "n_years": 8,
        "alpha": 0.05,
        "correction_method": "by",
    }
    responses = [
        client.post("/api/v1/stats/spatial-fdr", json=body) for _ in range(6)
    ]
    codes = [r.status_code for r in responses]
    # At least one request is allowed and at least one is rate-limited.
    assert 200 in codes, codes
    assert 429 in codes, codes


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
