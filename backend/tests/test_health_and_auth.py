"""Tests for health, capability discovery, and non-leaking service credential status."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.services.auth_store import AuthStore

client = TestClient(app)

_HINT_FIELDS = (
    "earthdata_username_hint",
    "earthdata_token_hint",
    "firms_key_hint",
    "cds_key_hint",
)


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


def test_auth_status_has_no_hint_fields(monkeypatch):
    """The /auth/status body must contain no *_hint fields that could leak
    partial secrets."""
    monkeypatch.setenv("EARTHDATA_TOKEN", "edl_tok_abc123def456")
    monkeypatch.setenv("EARTHDATA_USERNAME", "edl_user_xyz789uvw")
    monkeypatch.setenv("NASA_FIRMS_MAP_KEY", "firms_key_qwerty123")
    monkeypatch.setenv("CDS_API_KEY", "cds_key_asdfgh")

    # Reconstruct the store so it picks up the patched env, then swap it into
    # the router module so the endpoint uses it.
    store = AuthStore()
    monkeypatch.setattr("app.routers.health.auth_store", store)

    # Raw status() must not contain hint fields.
    raw = store.status()
    for field in _HINT_FIELDS:
        assert field not in raw, field

    # Serialized response must not contain hint fields or any secret substring.
    resp = client.get("/api/v1/auth/status")
    assert resp.status_code == 200
    body = resp.json()
    serialized = str(body)
    for field in _HINT_FIELDS:
        assert field not in serialized, field

    secrets = (
        "edl_tok_abc123def456",
        "edl_user_xyz789uvw",
        "firms_key_qwerty123",
        "cds_key_asdfgh",
    )
    for secret in secrets:
        assert secret not in serialized, secret

    # Booleans and source labels are still reported.
    assert body["earthdata_configured"] is True
    assert body["earthdata_source"] == "environment"
    assert body["firms_configured"] is True
    assert body["firms_source"] == "environment"
    assert body["cds_configured"] is True
    assert body["cds_source"] == "environment"
