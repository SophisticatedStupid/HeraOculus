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


@dataclass(frozen=True)
class CredentialSnapshot:
    """Immutable snapshot of credentials resolved at process start."""
    earthdata_token: Optional[str] = None
    earthdata_username: Optional[str] = None
    earthdata_password: Optional[str] = None
    firms_map_key: Optional[str] = None
    cds_api_key: Optional[str] = None
    cdsapi_key: Optional[str] = None
    destine_pat: Optional[str] = None


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
            cds_api_key=os.getenv("CDS_API_KEY") or None,
            cdsapi_key=os.getenv("CDSAPI_KEY") or None,
            destine_pat=os.getenv("DESTINE_PAT") or None,
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

    def get_destine_pat(self) -> Optional[str]:
        return self._snapshot.destine_pat

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
        destine_pat = self.get_destine_pat()

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
        destine_source = "environment" if destine_pat else "none"

        return {
            "earthdata_configured": bool(edl_token or (edl_user and edl_pass)),
            "earthdata_source": edl_source,
            "firms_configured": bool(firms_key),
            "firms_source": firms_source,
            "cds_configured": bool(cds_key),
            "cds_source": cds_source,
            "destine_configured": bool(destine_pat),
            "destine_source": destine_source,
            "keyless_feeds_available": [
                "NASA EONET v3 Wildfires & Events",
                "NASA ASF SAR Granule Search",
                "NASA CMR Discovery",
                "Local NetCDF4/HDF5/GeoTIFF Upload Analytics",
            ],
        }


auth_store = AuthStore()