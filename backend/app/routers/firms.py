"""NASA FIRMS & EONET active wildfire routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..schemas import FirmsQueryRequest
from ..security import limiter, require_api_key
from ..services.firms_service import query_firms_fires

router = APIRouter(
    prefix="/api/v1/firms",
    tags=["Wildfire & FIRMS"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/fires")
@limiter.limit("5/minute")
def get_active_fires(request: Request, payload: FirmsQueryRequest) -> dict:
    return query_firms_fires(
        west=payload.west,
        south=payload.south,
        east=payload.east,
        north=payload.north,
        source=payload.source,
        day_range=payload.day_range,
        region_label=payload.region_label or "Selected Bounding Box",
        allow_eonet_fallback=payload.allow_eonet_fallback,
    )
