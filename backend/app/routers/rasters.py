"""SAR Change Detection, DEM Differencing (DoD), LULC Transitions & NASA ASF routes."""

from __future__ import annotations

import asyncio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..schemas import (
    AsfSearchRequest,
    DemDiffPresetRequest,
    LulcPresetRequest,
    SarChangePresetRequest,
)
from ..services.raster_service import (
    compute_dem_differencing,
    compute_dem_differencing_from_paths,
    compute_lulc_from_paths,
    compute_lulc_transitions,
    compute_sar_change_detection,
    compute_sar_change_from_paths,
    search_asf_sar_granules,
)
from .utils import cleanup_temp_file, read_checked_upload, validate_upload_signature

router = APIRouter(prefix="/api/v1/rasters", tags=["SAR, DEM & LULC Rasters"])

# Processing timeout for raster uploads (seconds).  Prevents a single large
# upload from monopolizing worker threads indefinitely.
RASTER_PROCESS_TIMEOUT = 120.0


async def _run_with_timeout(coro, timeout: float = RASTER_PROCESS_TIMEOUT):
    """Run an async coroutine with a hard wall-clock timeout."""
    return await asyncio.wait_for(coro, timeout=timeout)


@router.post("/dem-diff")
def dem_diff_preset(payload: DemDiffPresetRequest) -> dict:
    return compute_dem_differencing(
        scenario=payload.scenario,
        lod_threshold_m=payload.lod_threshold_m,
        pixel_size_m=payload.pixel_size_m,
    )


@router.post("/dem-diff/upload")
async def dem_diff_upload(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    lod_threshold_m: float = Form(default=0.75),
    pixel_grid_only: bool = Form(default=False),
) -> dict:
    b1_path, n1 = await read_checked_upload(before_file)
    b2_path, n2 = await read_checked_upload(after_file)
    validate_upload_signature(b1_path, "geotiff")
    validate_upload_signature(b2_path, "geotiff")
    try:
        return await _run_with_timeout(
            asyncio.to_thread(
                compute_dem_differencing_from_paths,
                b1_path, n1, b2_path, n2,
                lod_threshold_m=lod_threshold_m,
                pixel_grid_only=pixel_grid_only,
            )
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="DEM differencing processing timed out."
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to compute DEM DoD: {exc}") from exc
    finally:
        cleanup_temp_file(b1_path)
        cleanup_temp_file(b2_path)


@router.post("/sar-change")
def sar_change_preset(payload: SarChangePresetRequest) -> dict:
    return compute_sar_change_detection(
        scenario=payload.scenario,
        speckle_filter=payload.speckle_filter,
        window_size=payload.window_size,
        threshold_db=payload.threshold_db,
        threshold_mode=payload.threshold_mode,
        pixel_size_m=payload.pixel_size_m,
    )


@router.post("/sar-change/upload")
async def sar_change_upload(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    speckle_filter: str = Form(default="lee"),
    window_size: int = Form(default=5),
    threshold_db: float = Form(default=3.0),
    threshold_mode: str = Form(default="fixed"),
    pixel_grid_only: bool = Form(default=False),
) -> dict:
    b1_path, n1 = await read_checked_upload(before_file)
    b2_path, n2 = await read_checked_upload(after_file)
    validate_upload_signature(b1_path, "geotiff")
    validate_upload_signature(b2_path, "geotiff")
    filt = speckle_filter if speckle_filter in {"none", "lee", "mean"} else "lee"
    mode = threshold_mode if threshold_mode in {"fixed", "otsu"} else "fixed"
    try:
        return await _run_with_timeout(
            asyncio.to_thread(
                compute_sar_change_from_paths,
                b1_path, n1, b2_path, n2,
                speckle_filter=filt,  # type: ignore[arg-type]
                window_size=window_size,
                threshold_db=threshold_db,
                threshold_mode=mode,  # type: ignore[arg-type]
                pixel_grid_only=pixel_grid_only,
            )
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="SAR change processing timed out."
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to compute SAR change: {exc}") from exc
    finally:
        cleanup_temp_file(b1_path)
        cleanup_temp_file(b2_path)


@router.post("/asf-search")
def asf_sar_search(payload: AsfSearchRequest) -> dict:
    return search_asf_sar_granules(
        lat=payload.lat,
        lon=payload.lon,
        radius_deg=payload.radius_deg,
        platform=payload.platform,
        start_date=payload.start_date,
        end_date=payload.end_date,
        max_results=payload.max_results,
    )


@router.post("/lulc-transitions")
def lulc_preset(payload: LulcPresetRequest) -> dict:
    return compute_lulc_transitions(scenario=payload.scenario)


@router.post("/lulc-transitions/upload")
async def lulc_upload(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    pixel_grid_only: bool = Form(default=False),
) -> dict:
    b1_path, n1 = await read_checked_upload(before_file)
    b2_path, n2 = await read_checked_upload(after_file)
    validate_upload_signature(b1_path, "geotiff")
    validate_upload_signature(b2_path, "geotiff")
    try:
        return await _run_with_timeout(
            asyncio.to_thread(
                compute_lulc_from_paths,
                b1_path, n1, b2_path, n2, pixel_grid_only=pixel_grid_only
            )
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="LULC transition processing timed out."
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to compute LULC matrix: {exc}") from exc
    finally:
        cleanup_temp_file(b1_path)
        cleanup_temp_file(b2_path)
