"""Gridded climate / reanalysis rainfall routes.

Endpoints
---------
POST /api/v1/gridded/{dataset}

Returns a normalized monthly precipitation time series for a point (nearest
cell) or area-mean (mean of valid cells in a bounding box) from a public
gridded product.

Datasets
--------
* ``chirps``   — CHIRPS v3 (Climate Hazards Center), keyless, 0.05°, mm/month
* ``imerg``    — GPM IMERG Final V07B (GES DISC), NASA Earthdata login, 0.1°
* ``era5-land``— ERA5-Land (Copernicus CDS / ARCO Zarr mirror), CDS key, 0.1°

Missing credentials or an out-of-coverage query yield an explicit
``status: "unavailable"`` response — never synthetic data.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Path

from ..schemas import GriddedSeriesRequest, GriddedSeriesResponse
from ..security import require_api_key
from ..services.auth_store import auth_store
from ..services.gridded_climate_service import (
    GriddedUnavailable,
    GriddedUpstreamError,
    fetch_chirps_monthly,
)

router = APIRouter(
    prefix="/api/v1/gridded",
    tags=["Gridded Climate / Reanalysis Rainfall"],
    dependencies=[Depends(require_api_key)],
)


def _require_credentials(configured: bool, dataset: str, what: str) -> None:
    """Raise 400 (configured as unavailable) when a dataset needs credentials
    that are not set in the environment.  The router returns a clean
    'unavailable' series rather than a 5xx so callers can degrade gracefully."""
    if not configured:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{dataset.upper()} requires {what} credentials which are not "
                f"configured on this backend. Set them via environment variables "
                f"and redeploy, or choose a keyless dataset."
            ),
        )


@router.post("/{dataset}", response_model=GriddedSeriesResponse)
def gridded_series(
    dataset: str = Path(
        ...,
        description="Gridded dataset: chirps | imerg | era5-land",
    ),
    payload: GriddedSeriesRequest = ...,
) -> GriddedSeriesResponse:
    """Return a normalized monthly precipitation series."""
    if dataset not in ("chirps", "imerg", "era5-land"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown dataset '{dataset}'. "
                f"Allowed: {', '.join(['chirps', 'imerg', 'era5-land'])}."
            ),
        )

    # Resolve start / end year/month from validated 'YYYY-MM' strings.
    sy, sm = [int(x) for x in payload.start_date.split("-")]
    ey, em = [int(x) for x in payload.end_date.split("-")]

    if dataset == "chirps":
        # Keyless — no credentials required.
        try:
            series = fetch_chirps_monthly(
                lat=payload.lat,
                lon=payload.lon,
                start_year=sy,
                start_month=sm,
                end_year=ey,
                end_month=em,
                area_mean=payload.area_mean,
                bbox=tuple(payload.bbox) if payload.bbox else None,
            )
        except GriddedUnavailable as exc:
            return GriddedSeriesResponse(
                status="unavailable",
                message=str(exc),
                series=None,
            )
        except GriddedUpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return GriddedSeriesResponse(
            status="available",
            message=(
                f"Retrieved {series.valid_count} valid month(s) of CHIRPS v3 "
                f"({series.missing_count} missing) for {payload.start_date}.."
                f"{payload.end_date}."
            ),
            series=series.to_dict(),
        )

    # IMERG and ERA5-Land require credentials; check before touching upstream.
    if dataset == "imerg":
        _require_credentials(
            auth_store.has_earthdata_auth(), "imerg", "NASA Earthdata"
        )
        raise HTTPException(
            status_code=501,
            detail="GPM IMERG connector not yet implemented in this build.",
        )
    if dataset == "era5-land":
        _require_credentials(
            bool(os.getenv("CDS_API_KEY")) or bool(os.getenv("CDSAPI_KEY")),
            "era5-land",
            "Copernicus CDS API",
        )
        raise HTTPException(
            status_code=501,
            detail="ERA5-Land connector not yet implemented in this build.",
        )

    raise HTTPException(status_code=400, detail=f"Unknown dataset '{dataset}'.")
