"""Read-only service credential resolver backed by environment variables only.

HeraOculus uses shared service credentials for NASA Earthdata / FIRMS proxies.
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


def _env_optional(name: str) -> Optional[str]:
    """Return an env var stripped of surrounding whitespace, or None when unset,
    empty, or whitespace-only.  Blank credentials behave like unset so a
    deployment that ships ``VAR=""`` does not accidentally hold an empty secret."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


@dataclass(frozen=True)
class CredentialSnapshot:
    """Immutable snapshot of credentials resolved at process start."""
    earthdata_token: Optional[str] = None
    earthdata_username: Optional[str] = None
    earthdata_password: Optional[str] = None
    firms_map_key: Optional[str] = None
    cds_api_key: Optional[str] = None
    cdsapi_key: Optional[str] = None


class AuthStore:
    """Thread-safe read-only holder resolving credentials from environment variables.

    All credentials are loaded once at initialization from environment variables.
    There is no runtime mutation, no per-session overrides, and no cross-user
    sharing concerns because credentials are process-wide by design.
    """

    def __init__(self) -> None:
        self._snapshot = CredentialSnapshot(
            earthdata_token=_env_optional("EARTHDATA_TOKEN"),
            earthdata_username=_env_optional("EARTHDATA_USERNAME"),
            earthdata_password=_env_optional("EARTHDATA_PASSWORD"),
            firms_map_key=_env_optional("NASA_FIRMS_MAP_KEY"),
            cds_api_key=_env_optional("CDS_API_KEY"),
            cdsapi_key=_env_optional("CDSAPI_KEY"),
        )

    def get_firms_map_key(self) -> Optional[str]:
        return self._snapshot.firms_map_key

    def get_earthdata_token(self) -> Optional[str]:
        return self._snapshot.earthdata_token

    def get_earthdata_credentials(self) -> tuple[Optional[str], Optional[str]]:
        return self._snapshot.earthdata_username, self._snapshot.earthdata_password

    def get_cds_credentials(self) -> Optional[str]:
        """Return a CDS API key if configured (CDS_API_KEY or CDSAPI_KEY)."""
        return self._snapshot.cds_api_key or self._snapshot.cdsapi_key

    def has_earthdata_auth(self) -> bool:
        token = self.get_earthdata_token()
        user, pw = self.get_earthdata_credentials()
        return bool(token or (user and pw))

    def status(self) -> dict:
        edl_token = self._snapshot.earthdata_token
        edl_user = self._snapshot.earthdata_username
        edl_pass = self._snapshot.earthdata_password
        firms_key = self._snapshot.firms_map_key
        cds_key = self.get_cds_credentials()

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
        cds_source = "environment" if cds_key else "none"

        return {
            "earthdata_configured": bool(edl_token or (edl_user and edl_pass)),
            "earthdata_source": edl_source,
            "firms_configured": bool(firms_key),
            "firms_source": firms_source,
            "cds_configured": bool(cds_key),
            "cds_source": cds_source,
            "keyless_feeds_available": [
                "NASA EONET v3 Wildfires & Events",
                "NASA ASF SAR Granule Search",
                "NASA CMR Discovery",
                "Local NetCDF4/HDF5/GeoTIFF Upload Analytics",
            ],
        }


auth_store = AuthStore()