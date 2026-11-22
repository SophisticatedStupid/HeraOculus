"""Tests for the gridded-climate disk cache under a read-only filesystem.

When the cache directory cannot be created or written to, the service must
fall back to a temp directory, and failing that disable caching entirely —
never crash the app or fail a request.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.services import gridded_climate_service as gcs

client = TestClient(app)


def _disable_cache(monkeypatch):
    """Simulate a read-only filesystem by making cache resolution return None."""
    monkeypatch.setattr(gcs, "_CACHE_DIR", None)
    monkeypatch.setattr(gcs, "_CACHE_DISABLED", True)


def test_cache_disabled_when_read_only(monkeypatch):
    """Simulating a read-only path leaves cache get/set as no-ops and the app
    still imports and serves requests."""
    _disable_cache(monkeypatch)

    # Cache read/write must be no-ops, not raise.
    assert gcs._cache_get("chirps", "any_payload") is None
    gcs._cache_set("chirps", "any_payload", b"some data")  # must not raise

    # The app still imports and serves a request.
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_resolve_cache_dir_returns_none_when_read_only(monkeypatch):
    """_resolve_cache_dir returns None and disables caching when every
    candidate location is unwritable."""
    # Force every candidate to fail by monkeypatching Path.mkdir to raise.
    def fake_mkdir(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(gcs.Path, "mkdir", fake_mkdir)
    # Reset module state so _resolve_cache_dir re-evaluates candidates.
    monkeypatch.setattr(gcs, "_CACHE_DIR", None)
    monkeypatch.setattr(gcs, "_CACHE_DISABLED", False)

    assert gcs._resolve_cache_dir() is None
    assert gcs._CACHE_DISABLED is True

    # get/set remain no-ops.
    assert gcs._cache_get("chirps", "p") is None
    gcs._cache_set("chirps", "p", b"data")
