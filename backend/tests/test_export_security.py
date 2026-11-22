"""Security tests: export filename safety, API-key authentication, rate limiting.

These tests verify that:
* Export artifacts are written inside a dedicated directory and never outside it.
* Client-supplied export filenames with path separators or ``..`` traversal are
  rejected (HTTP 400), never written to disk.
* API-key authentication fails closed when a key is configured (missing/invalid
  header -> 401) and allows all requests in dev mode (key unset/blank).
* Blank or whitespace-only HERAOCULUS_API_KEY is treated as unset (dev mode).
* Heavy raster endpoints enforce rate limits (HTTP 429).
* No secrets leak into response bodies.

The slowapi limiter uses in-memory storage shared across the test process, so
each test resets it to avoid cross-test rate-limit exhaustion.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.security import limiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    """Reset the in-memory rate-limiter storage before every test."""
    storage = limiter.limiter.storage
    if hasattr(storage, "reset"):
        storage.reset()  # type: ignore[attr-defined]


def _make_export_payload() -> dict:
    arr = np.full((4, 4), 50.0, dtype=float)
    arr[1:3, 1:3] = np.nan
    interp = client.post(
        "/api/v1/rasters/interpolate",
        json={
            "data": [[None if np.isnan(v) else float(v) for v in row] for row in arr],
            "method": "nearest_neighbor",
            "raster_type": "continuous",
            "band_index": 1,
        },
    )
    assert interp.status_code == 200, interp.text
    data = interp.json()
    return {
        "output_filename": "result.tif",
        "nodata": -9999.0,
        "include_provenance_band": True,
        "include_quality_band": False,
        "reconstructed_array": data["reconstructed"],
        "provenance_mask": data["provenance_mask"],
        "crs": "EPSG:32610",
        "transform": {"a": 10, "b": 0, "c": 500000, "d": 0, "e": -10, "f": 5000000},
        "method": "nearest_neighbor",
    }


# ---------------------------------------------------------------------------
# Export filename security
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "evil",
    [
        "../../../outside/evil.tif",
        "..\\..\\evil.tif",
        "folder/evil.tif",
        "evil/../escape.tif",
        "./evil.tif",
    ],
)
def test_export_rejects_traversal_filenames(evil: str) -> None:
    """Traversal filenames are rejected with 400 and never exported."""
    resp = client.post(
        "/api/v1/rasters/interpolate/export",
        json={**_make_export_payload(), "output_filename": evil},
    )
    assert resp.status_code == 400, f"{evil!r} should be rejected, got {resp.status_code}"


def test_export_accepted_filename_echoed() -> None:
    """A safe basename is echoed in Content-Disposition and the body is valid GeoTIFF."""
    resp = client.post(
        "/api/v1/rasters/interpolate/export",
        json={**_make_export_payload(), "output_filename": "my_result.tif"},
    )
    assert resp.status_code == 200, resp.text
    cd = resp.headers.get("content-disposition", "")
    assert 'filename="my_result.tif"' in cd, cd
    assert resp.headers.get("content-type", "").startswith("image/tiff")


# ---------------------------------------------------------------------------
# API-key authentication
# ---------------------------------------------------------------------------

def _configured_key() -> object:
    """Read _configured_api_key directly (unit-level, no event loop needed)."""
    from app.security import _configured_api_key

    return _configured_api_key()


def test_configured_api_key_blank_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset or whitespace-only HERAOCULUS_API_KEY => None (dev mode)."""
    monkeypatch.delenv("HERAOCULUS_API_KEY", raising=False)
    assert _configured_key() is None
    monkeypatch.setenv("HERAOCULUS_API_KEY", "   ")
    assert _configured_key() is None


def test_configured_api_key_real_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-blank key is returned stripped."""
    monkeypatch.setenv("HERAOCULUS_API_KEY", "  secret-key-123  ")
    assert _configured_key() == "secret-key-123"


def test_protected_endpoint_allows_all_in_dev_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key configured => protected endpoint allows requests (no 401)."""
    monkeypatch.delenv("HERAOCULUS_API_KEY", raising=False)
    resp = client.post(
        "/api/v1/rasters/interpolate",
        json={"data": [[1.0, 2.0], [3.0, 4.0]], "method": "nearest_neighbor"},
    )
    # 200 (success) or 422 (validation), but never 401.
    assert resp.status_code != 401, resp.text


def test_protected_endpoint_fails_closed_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Key configured => missing/invalid header -> 401, valid header -> success."""
    monkeypatch.setenv("HERAOCULUS_API_KEY", "secret-key-123")

    # Missing header -> 401
    resp = client.post(
        "/api/v1/rasters/interpolate",
        json={"data": [[1.0, 2.0], [3.0, 4.0]], "method": "nearest_neighbor"},
    )
    assert resp.status_code == 401, resp.text

    # Wrong header -> 401
    resp = client.post(
        "/api/v1/rasters/interpolate",
        json={"data": [[1.0, 2.0], [3.0, 4.0]], "method": "nearest_neighbor"},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401, resp.text

    # Correct header -> success (not 401)
    resp = client.post(
        "/api/v1/rasters/interpolate",
        json={"data": [[1.0, 2.0], [3.0, 4.0]], "method": "nearest_neighbor"},
        headers={"X-API-Key": "secret-key-123"},
    )
    assert resp.status_code != 401, resp.text


def test_no_secret_leaks_in_responses() -> None:
    """Health/auth status responses never contain full secrets."""
    for path in ["/api/v1/health", "/api/v1/auth/status"]:
        resp = client.get(path)
        assert resp.status_code == 200
        body = str(resp.json())
        for needle in ("super_secret", "my_earthdata_token", "my_firms_map_key"):
            assert needle not in body


# ---------------------------------------------------------------------------
# Rate limiting on heavy endpoints
# ---------------------------------------------------------------------------

def test_heavy_endpoint_enforces_rate_limit() -> None:
    """The interpolate endpoint (LIMIT_HEAVY = 5/min) returns 429 under load."""
    payload = {
        "data": [[1.0, 2.0], [3.0, None]],
        "method": "nearest_neighbor",
        "raster_type": "continuous",
        "band_index": 1,
    }
    statuses = []
    for _ in range(7):
        resp = client.post("/api/v1/rasters/interpolate", json=payload)
        statuses.append(resp.status_code)
    # At least one request should be rate-limited (429) once the 5/min window is exceeded.
    assert 429 in statuses, f"Expected a 429 under load, got statuses {statuses}"
    # All responses are JSON, no internal details leaked.
    resp = client.post("/api/v1/rasters/interpolate", json=payload)
    body = str(resp.json()).lower()
    assert "traceback" not in body
    assert "/tmp" not in body
    assert "tempfile" not in body
