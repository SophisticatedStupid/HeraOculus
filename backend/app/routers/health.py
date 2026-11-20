"""Health check and service credential status routes.

Credential configuration is server-managed via environment variables only
(EARTHDATA_TOKEN, EARTHDATA_USERNAME, EARTHDATA_PASSWORD, NASA_FIRMS_MAP_KEY).
There is no public per-user credential configuration endpoint because this
deployment has no authentication infrastructure.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..schemas import AuthStatusResponse, HealthResponse
from ..services.auth_store import auth_store

router = APIRouter(prefix="/api/v1", tags=["Health & Auth"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.version,
        capabilities={
            "sarima": True,
            "bayesian_mcmc": True,
            "extreme_value_gev_gpd": True,
            "spatial_fdr": True,
            "netcdf4_hdf5_xarray": True,
            "rasterio_dem_sar_lulc": True,
            "firms_eonet_proxy": True,
            "asf_sar_search": True,
            "appeears_cmr_proxy": True,
        },
        auth=AuthStatusResponse(**auth_store.status()),
    )


@router.get("/auth/status", response_model=AuthStatusResponse)
def get_auth_status() -> AuthStatusResponse:
    return AuthStatusResponse(**auth_store.status())
