"""Read-only service credential resolver backed by environment variables only.

Geo-Iris uses shared service credentials for NASA Earthdata / FIRMS proxies.
There is no per-user credential store: credentials are sourced exclusively from
environment variables (EARTHDATA_TOKEN, EARTHDATA_USERNAME, EARTHDATA_PASSWORD,
NASA_FIRMS_MAP_KEY). No endpoint accepts or mutates user-provided secrets.

This design eliminates cross-user credential contamination and avoids the need
for an authentication system. Credentials are configured at deployment time
via environment variables or a secure secret manager (e.g., AWS Secrets Manager,
HashiCorp Vault, Kubernetes secrets) and apply process-wide.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CredentialSnapshot:
    """Immutable snapshot of credentials resolved at process start."""
    earthdata_token: Optional[str] = None
    earthdata_username: Optional[str] = None
    earthdata_password: Optional[str] = None
    firms_map_key: Optional[str] = None


def _mask_secret(value: Optional[str]) -> Optional[str]:
    """Return a masked hint for logging/status without exposing the secret."""
    if not value:
        return None
    cleaned = value.strip()
    if len(cleaned) <= 6:
        return "*" * len(cleaned)
    return f"{cleaned[:3]}...{cleaned[-3:]}"


class AuthStore:
    """Thread-safe read-only holder resolving credentials from environment variables.

    All credentials are loaded once at initialization from environment variables.
    There is no runtime mutation, no per-session overrides, and no cross-user
    sharing concerns because credentials are process-wide by design.
    """

    def __init__(self) -> None:
        self._snapshot = CredentialSnapshot(
            earthdata_token=os.getenv("EARTHDATA_TOKEN") or None,
            earthdata_username=os.getenv("EARTHDATA_USERNAME") or None,
            earthdata_password=os.getenv("EARTHDATA_PASSWORD") or None,
            firms_map_key=os.getenv("NASA_FIRMS_MAP_KEY") or None,
        )

    def get_firms_map_key(self) -> Optional[str]:
        return self._snapshot.firms_map_key

    def get_earthdata_token(self) -> Optional[str]:
        return self._snapshot.earthdata_token

    def get_earthdata_credentials(self) -> tuple[Optional[str], Optional[str]]:
        return self._snapshot.earthdata_username, self._snapshot.earthdata_password

    def has_earthdata_auth(self) -> bool:
        token = self.get_earthdata_token()
        user, pw = self.get_earthdata_credentials()
        return bool(token or (user and pw))

    def status(self) -> dict:
        edl_token = self._snapshot.earthdata_token
        edl_user = self._snapshot.earthdata_username
        edl_pass = self._snapshot.earthdata_password
        firms_key = self._snapshot.firms_map_key

        edl_source = (
            "environment"
            if (edl_token or (edl_user and edl_pass))
            else "none"
        )
        firms_source = (
            "environment"
            if firms_key
            else "none"
        )

        return {
            "earthdata_configured": bool(edl_token or (edl_user and edl_pass)),
            "earthdata_source": edl_source,
            "earthdata_username_hint": _mask_secret(edl_user),
            "earthdata_token_hint": _mask_secret(edl_token),
            "firms_configured": bool(firms_key),
            "firms_source": firms_source,
            "firms_key_hint": _mask_secret(firms_key),
            "keyless_feeds_available": [
                "NASA EONET v3 Wildfires & Events",
                "NASA ASF SAR Granule Search",
                "NASA CMR Discovery",
                "Local NetCDF4/HDF5/GeoTIFF Upload Analytics",
            ],
        }


auth_store = AuthStore()