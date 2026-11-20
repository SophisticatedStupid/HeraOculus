"""Earthdata, AppEEARS, Vertical Sounding Profiles, Trace Gases & NetCDF4/HDF5 routes."""

from __future__ import annotations

import asyncio
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..schemas import (
    AppeearsPointRequest,
    EarthdataGranuleRequest,
    TraceGasRequest,
    VerticalProfileRequest,
)
from ..services.earthdata_service import (
    extract_netcdf_subset,
    get_trace_gas_analytics,
    get_vertical_pressure_profile,
    inspect_netcdf_file,
    proxy_appeears_point_task,
    search_earthdata_granules,
)
from .utils import cleanup_temp_file, read_checked_upload, validate_upload_signature

router = APIRouter(prefix="/api/v1/earthdata", tags=["Earthdata & NetCDF4"])

# Processing timeout for NetCDF/HDF5 uploads (seconds).
NETCDF_PROCESS_TIMEOUT = 120.0


async def _run_with_timeout(coro, timeout: float = NETCDF_PROCESS_TIMEOUT):
    """Run an async coroutine with a hard wall-clock timeout."""
    return await asyncio.wait_for(coro, timeout=timeout)


@router.post("/granules")
def search_granules(payload: EarthdataGranuleRequest) -> dict:
    return search_earthdata_granules(
        short_name=payload.short_name,
        bbox=payload.bbox,
        start_date=payload.start_date,
        end_date=payload.end_date,
        max_results=payload.max_results,
    )


@router.post("/appeears-point")
def submit_appeears_point(payload: AppeearsPointRequest) -> dict:
    return proxy_appeears_point_task(
        product=payload.product,
        layer=payload.layer,
        lat=payload.lat,
        lon=payload.lon,
        start_date=payload.start_date,
        end_date=payload.end_date,
        task_name=payload.task_name,
    )


@router.post("/vertical-profile")
def vertical_profile(payload: VerticalProfileRequest) -> dict:
    return get_vertical_pressure_profile(
        lat=payload.lat,
        lon=payload.lon,
        start_year=payload.start_year,
        end_year=payload.end_year,
        product=payload.product,
    )


@router.post("/trace-gases")
def trace_gases(payload: TraceGasRequest) -> dict:
    return get_trace_gas_analytics(
        gas=payload.gas,
        lat=payload.lat,
        lon=payload.lon,
        region_label=payload.region_label or f"({payload.lat:.1f}, {payload.lon:.1f})",
        start_year=payload.start_year,
        end_year=payload.end_year,
    )


@router.post("/netcdf/inspect")
async def inspect_netcdf(file: UploadFile = File(...)) -> dict:
    path, filename = await read_checked_upload(file)
    validate_upload_signature(path, "netcdf_hdf5")
    try:
        return await _run_with_timeout(
            asyncio.to_thread(inspect_netcdf_file, path, filename)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="NetCDF inspect timed out."
        ) from exc
    finally:
        cleanup_temp_file(path)


@router.post("/netcdf/extract")
async def extract_netcdf(
    file: UploadFile = File(...),
    variable: Optional[str] = Form(default=None),
    target_lat: Optional[float] = Form(default=None),
    target_lon: Optional[float] = Form(default=None),
    spatial_mean: bool = Form(default=False),
) -> dict:
    path, filename = await read_checked_upload(file)
    validate_upload_signature(path, "netcdf_hdf5")
    try:
        return await _run_with_timeout(
            asyncio.to_thread(
                extract_netcdf_subset,
                path,
                filename,
                variable=variable,
                target_lat=target_lat,
                target_lon=target_lon,
                spatial_mean=spatial_mean,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="NetCDF extract timed out."
        ) from exc
    finally:
        cleanup_temp_file(path)
