"""STAC satellite imagery catalog search routes.

Proxy searches against public STAC catalogs (Earth Search by Element 84 by
default; Microsoft Planetary Computer optional via ``STAC_PROVIDER``) and
return normalized, provider-agnostic item metadata.

This is a catalog/metadata-only endpoint: it discovers available scenes and
their Cloud-Optimized GeoTIFF asset URLs, but never downloads pixel data or
fabricates synthetic results.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..schemas import StacSearchRequest, StacSearchResponse
from ..security import LIMIT_UPSTREAM, limiter, require_api_key
from ..services.stac_service import search_items

router = APIRouter(
    prefix="/api/v1/stac",
    tags=["STAC Imagery Catalog"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/search", response_model=StacSearchResponse)
@limiter.limit(LIMIT_UPSTREAM)
def stac_search(request: Request, payload: StacSearchRequest) -> StacSearchResponse:
    """Search a STAC catalog for satellite imagery scenes.

    Request body (JSON)::

        {
          "collections": ["sentinel-2-l2a"],
          "bbox": [-122.5, 37.0, -122.0, 37.5],
          "start_date": "2024-06-01T00:00:00Z",
          "end_date": "2024-06-10T23:59:59Z",
          "max_cloud_cover": 20,
          "limit": 10
        }

    Returns normalized items with ``id``, ``datetime``, ``cloud_cover``,
    ``bbox``, ``platform``, ``thumbnail`` and a dict of canonical asset hrefs
    (``red``/``nir``/``swir``/``scl``/``blue``/``green``).
    """
    try:
        result = search_items(
            collections=payload.collections,
            bbox=payload.bbox,
            start_date=payload.start_date,
            end_date=payload.end_date,
            max_cloud_cover=payload.max_cloud_cover,
            limit=payload.limit,
            page_token=payload.page_token,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=502,
            detail=f"STAC search failed: {exc}",
        ) from exc

    return StacSearchResponse(
        provider=result["provider"],
        items=result["items"],
        context=result["context"],
        next_token=result.get("next_token"),
    )
