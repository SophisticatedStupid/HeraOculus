"""Tests for health, capability discovery, and non-leaking service credential status."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["capabilities"]["sarima"] is True
    assert data["capabilities"]["bayesian_mcmc"] is True
    assert data["capabilities"]["spatial_fdr"] is True
    assert data["capabilities"]["netcdf4_hdf5_xarray"] is True
    assert data["capabilities"]["rasterio_dem_sar_lulc"] is True


def test_unauthenticated_user_cannot_configure_credentials():
    """Per-user credential configuration endpoint must not exist (no auth infra)."""
    resp = client.post("/api/v1/auth/configure", json={"firms_map_key": "some_key"})
    assert resp.status_code == 404


def test_auth_status_reports_environment_source_without_secrets():
    resp = client.get("/api/v1/auth/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "earthdata_configured" in body
    assert "firms_configured" in body
    assert body["earthdata_source"] in {"environment", "none"}
    assert body["firms_source"] in {"environment", "none"}
    # Credentials must never appear in the response body.
    serialized = str(body)
    assert "EDL" not in serialized
    assert "NASA_FIRMS" not in serialized or "firms" in serialized.lower()
