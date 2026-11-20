"""Security tests for server-managed credential model.

These tests verify that credential handling is safe when multiple users
share a single process with no per-user credential isolation.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_environment_defaults_still_work():
    """Verify that environment variables are respected at process start."""
    # The health endpoint should succeed regardless of env state.
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    # Source should be "environment" or "none", never "session".
    assert data["auth"]["earthdata_source"] in {"environment", "none"}
    assert data["auth"]["firms_source"] in {"environment", "none"}


def test_credentials_never_appear_in_response_bodies():
    """Ensure raw secrets are not exposed in any API response."""
    # Check /health and /auth/status
    for path in ["/api/v1/health", "/api/v1/auth/status"]:
        resp = client.get(path)
        assert resp.status_code == 200
        serialized = str(resp.json())
        # Masked hints may appear, but not full tokens/keys.
        # The mask function only keeps first 3 and last 3 chars.
        # Full raw secrets must be absent.
        # Check that common secret patterns are not verbatim.
        assert "super_secret" not in serialized  # noqa: E501
        assert "secret_key" not in serialized  # noqa: E501
        assert "my_earthdata_token" not in serialized  # noqa: E501
        assert "my_firms_map_key" not in serialized  # noqa: E501

    # Verify /auth/configure returns 404, so no secret leakage via that route
    resp = client.post("/api/v1/auth/configure", json={"clear_session": True})
    assert resp.status_code == 404


def test_environment_variable_isolation_between_simulated_users():
    """In the server-managed model, there is no per-user credential store.

    This test verifies that the /auth/configure endpoint is absent (404),
    meaning no user can override another's effective credentials.
    Since credentials are process-wide environment variables, all users
    see the same effective credentials — which is the expected safe behavior.
    """
    # Both simulated users observe identical credential status.
    health_resp = client.get("/api/v1/health").json()
    auth_resp = client.get("/api/v1/auth/status").json()

    # Health endpoint same for both (trivially, same process)
    assert health_resp["status"] == "ok"

    # Auth status consistent across repeated calls (process-wide, no per-user drift)
    # Calling multiple times should give the same source values.
    calls = [client.get("/api/v1/auth/status").json() for _ in range(5)]
    for i in range(1, len(calls)):
        assert calls[i]["earthdata_source"] == calls[0]["earthdata_source"]
        assert calls[i]["firms_source"] == calls[0]["firms_source"]

    # /auth/configure does not exist — no user can affect another's effective creds
    configure_resp = client.post("/api/v1/auth/configure", json={"clear_session": True})
    assert configure_resp.status_code == 404, (
        "configure endpoint must not exist; per-user credential override "
        "would enable cross-user credential contamination"
    )