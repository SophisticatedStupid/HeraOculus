"""SAR Change Detection, DEM Differencing (DoD), LULC Transitions,
NASA ASF, and Raster Interpolation & Gap-Filling routes."""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from rasterio.transform import Affine

from ..security import limiter, require_api_key

from ..schemas import (
    AsfSearchRequest,
    CompatibleMethodsResponse,
    DemDiffPresetRequest,
    ExportRequest,
    InterpolateRequest,
    InterpolateResponse,
    LulcPresetRequest,
    RasterInspectRequest,
    RasterInspectResponse,
    SarChangePresetRequest,
    ValidateRequest,
    ValidateResponse,
)
from ..services.interpolation_provenance_service import (
    build_provenance_mask,
    compute_quality_raster,
    save_provenance_geotiff,
)
from ..services.interpolation_validation_service import (
    get_compatible_methods,
    validate_interpolation,
)
from ..services.raster_interpolation_service import (
    INTERPOLATION_METHODS,
    InterpolationResult,
    run_interpolation,
    run_temporal_interpolation,
    TemporalInterpolationResult,
)
from ..services.raster_inspection_service import (
    RasterInspectionResult,
    inspect_raster_from_bytes,
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
from .utils import (
    cleanup_temp_file,
    read_checked_upload,
    run_heavy_task,
    validate_upload_signature,
)

router = APIRouter(
    prefix="/api/v1/rasters",
    tags=["SAR, DEM & LULC Rasters"],
    dependencies=[Depends(require_api_key)],
)

# Processing timeout for raster uploads (seconds).  Prevents a single large
# upload from monopolizing worker threads indefinitely.
RASTER_PROCESS_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# Raster Inspection
# ---------------------------------------------------------------------------

@router.post("/inspect")
async def inspect_raster(
    file: UploadFile = File(...),
    band_index: int = Form(default=1, ge=1),
    user_confirmed_type: Optional[str] = Form(default=None),
) -> dict:
    """Inspect an uploaded raster for NoData and missing-data diagnostics.

    Returns dimensions, CRS, transform, valid/missing counts, per-band stats,
    connected missing-region analysis, and distance-to-valid statistics.
    """
    path, name = await read_checked_upload(file)
    validate_upload_signature(path, "geotiff")
    try:
        result: RasterInspectionResult = await asyncio.to_thread(
            inspect_raster_from_bytes,
            raw_bytes=open(path, "rb").read(),
            filename=name,
            band_index=band_index,
            user_confirmed_type=user_confirmed_type if user_confirmed_type else None,
        )
        return _inspection_result_to_dict(result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to inspect raster: {exc}") from exc
    finally:
        cleanup_temp_file(path)


def _inspection_result_to_dict(result: RasterInspectionResult) -> dict:
    return {
        "width": result.width,
        "height": result.height,
        "bands": result.bands,
        "crs": result.crs,
        "transform": result.transform,
        "pixel_size_x": result.pixel_size_x,
        "pixel_size_y": result.pixel_size_y,
        "extent": result.extent,
        "data_type": result.data_type,
        "nodata_value": result.nodata_value,
        "has_alpha_band": result.has_alpha_band,
        "band_index": result.band_index,
        "valid_pixel_count": result.valid_pixel_count,
        "missing_pixel_count": result.missing_pixel_count,
        "missing_percentage": result.missing_percentage,
        "valid_percentage": result.valid_percentage,
        "band_stats": result.band_stats,
        "raster_type": result.raster_type,
        "missing_region_count": result.missing_region_count,
        "missing_region_stats": [
            {
                "id": s.id,
                "pixel_count": s.pixel_count,
                "area_map_units": s.area_map_units,
                "bbox": s.bbox,
                "centroid": s.centroid,
                "is_edge": s.is_edge,
                "surrounded_by_valid": s.surrounded_by_valid,
                "min_distance_to_valid": s.min_distance_to_valid,
                "mean_distance_to_valid": s.mean_distance_to_valid,
            }
            for s in result.missing_region_stats
        ],
        "largest_missing_region_pixels": result.largest_missing_region_pixels,
        "average_missing_region_size": result.average_missing_region_size,
        "missing_region_size_distribution": result.missing_region_size_distribution,
        "isolated_missing_pixels": result.isolated_missing_pixels,
        "edge_gaps": result.edge_gaps,
        "interior_gaps": result.interior_gaps,
        "distance_to_valid_stats": result.distance_to_valid_stats,
        "missing_spatial_distribution": result.missing_spatial_distribution,
        "processing_duration_ms": result.processing_duration_ms,
        "inspected_at": result.inspected_at,
    }


# ---------------------------------------------------------------------------
# Compatible Methods
# ---------------------------------------------------------------------------

@router.get("/interpolate/compatible-methods")
def compatible_methods(
    raster_type: str = "continuous",
) -> CompatibleMethodsResponse:
    """Return interpolation methods compatible with a given raster type."""
    methods = get_compatible_methods(raster_type)
    return CompatibleMethodsResponse(raster_type=raster_type, methods=methods)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

@router.post("/interpolate")
def interpolate_raster(payload: InterpolateRequest) -> dict:
    """Interpolate missing values in a raster band.

    Accepts a 2D array as nested lists with metadata (transform, CRS, nodata).
    Returns reconstructed array, provenance mask, quality raster, and statistics.
    """
    import numpy as np

    try:
        array = np.array(payload.data, dtype=float)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid `data` array: {exc}") from exc

    is_temporal = payload.method.startswith("temporal_")

    if is_temporal:
        if array.ndim != 3:
            raise HTTPException(
                status_code=400,
                detail=f"Temporal interpolation requires a 3D cube (T, H, W), got {array.ndim}D.",
            )
    else:
        if array.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Spatial interpolation requires a 2D array, got {array.ndim}D.",
            )

    try:
        if is_temporal:
            if payload.times is None:
                raise ValueError("`times` is required for temporal interpolation methods.")
            times = np.array(payload.times, dtype=float)
            result: TemporalInterpolationResult = run_temporal_interpolation(
                cube=array,
                method=payload.method,
                times=times,
                max_gap_steps=payload.max_gap_steps,
                allow_extrapolation=payload.allow_extrapolation,
                band_index=payload.band_index,
            )
            return _temporal_result_to_dict(result)
        else:
            result: InterpolationResult = run_interpolation(
                array=array,
                method=payload.method,
                band_index=payload.band_index,
                raster_type=payload.raster_type,
                max_distance_pixels=payload.max_distance_pixels,
                geographic_crs=payload.geographic_crs,
                pixel_size_x=payload.pixel_size_x,
                pixel_size_y=payload.pixel_size_y,
                neighborhood=payload.neighborhood,
                edge_handling=payload.edge_handling,
                clip_to_observed_range=payload.clip_to_observed_range,
                window_size=payload.window_size,
                min_valid_neighbors=payload.min_valid_neighbors,
                neighborhood_shape=payload.neighborhood_shape,
                power=payload.power,
                search_radius_pixels=payload.search_radius_pixels,
                max_neighbors=payload.max_neighbors,
                min_neighbors=payload.min_neighbors,
                variogram_model=payload.variogram_model,
                variogram_nugget=payload.variogram_nugget,
                variogram_sill=payload.variogram_sill,
                variogram_range=payload.variogram_range,
            )
            return _interpolation_result_to_dict(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Interpolation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Export endpoint
# ---------------------------------------------------------------------------

@router.post("/interpolate/export")
def export_interpolation(payload: ExportRequest) -> dict:
    """Export reconstructed raster as GeoTIFF with provenance and quality bands.

    Accepts a reconstructed array and provenance mask, writes a GeoTIFF with
    three bands: reconstructed values, provenance mask, quality score.
    Does not overwrite any source file.
    """
    import numpy as np

    output_dir = tempfile.gettempdir()
    output_path = os.path.join(output_dir, payload.output_filename)

    # Convert transform dict to Affine
    affine_transform = None
    if payload.transform:
        try:
            affine_transform = Affine(
                payload.transform["a"], payload.transform["b"], payload.transform["c"],
                payload.transform["d"], payload.transform["e"], payload.transform["f"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid transform: {exc}"
            ) from exc

    try:
        save_provenance_geotiff(
            output_path=output_path,
            reconstructed_array=np.array(payload.reconstructed_array, dtype=float),
            provenance_mask=np.array(payload.provenance_mask, dtype=object),
            quality_raster=np.array(payload.quality_mask, dtype=float) if payload.quality_mask else None,
            crs=payload.crs,
            transform=affine_transform,
            nodata=payload.nodata,
            input_metadata={},
            processing_metadata={"method": payload.method, "export": True},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc

    return {
        "output_path": output_path,
        "filename": payload.output_filename,
        "bands": 3,
        "crs": str(payload.crs) if payload.crs else None,
        "include_provenance_band": payload.include_provenance_band,
        "include_quality_band": payload.include_quality_band,
    }


# ---------------------------------------------------------------------------
# Existing helper functions
# ---------------------------------------------------------------------------


def _array_from_payload(payload: InterpolateRequest) -> np.ndarray:
    """Extract the 2D array from the request payload.

    The frontend sends the array as `data` field (nested lists).
    """
    import numpy as np
    if payload.data is None:
        raise ValueError("`data` (2D array as nested lists) is required.")
    arr = np.array(payload.data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"`data` must be a 2D array, got {arr.ndim}D.")
    return arr


def _interpolation_result_to_dict(result: InterpolationResult) -> dict:
    import numpy as np

    def to_list(arr: Optional[np.ndarray]) -> Optional[list]:
        if arr is None:
            return None
        out = np.asarray(arr, dtype=object)
        result_list = []
        for row in out:
            result_row = []
            for v in row:
                if isinstance(v, float) and (np.isnan(v) if isinstance(v, float) else False):
                    result_row.append(None)
                elif isinstance(v, np.floating) and np.isnan(v):
                    result_row.append(None)
                else:
                    result_row.append(v.item() if hasattr(v, 'item') else v)
            result_list.append(result_row)
        return result_list

    return {
        "method": result.method,
        "band_index": result.band_index,
        "raster_type": result.raster_type,
        "config": result.config,
        "observed_pixel_count": result.observed_pixel_count,
        "interpolated_pixel_count": result.interpolated_pixel_count,
        "remaining_nodata_count": result.remaining_nodata_count,
        "excluded_pixel_count": result.excluded_pixel_count,
        "processing_duration_ms": result.processing_duration_ms,
        "provenance_mask": to_list(result.provenance_mask),
        "reconstructed": to_list(result.reconstructed),
        "quality_mask": to_list(result.quality_mask),
        "distance_to_observation": to_list(result.distance_to_observation),
        "kriging_variance": to_list(result.quality_mask),  # quality_mask IS kriging variance for kriging
        "variogram_fit": result.config.get("variogram_fit"),
    }


def _temporal_result_to_dict(result: TemporalInterpolationResult) -> dict:
    import numpy as np

    def to_list(arr: Optional[np.ndarray]) -> Optional[list]:
        if arr is None:
            return None
        if arr.ndim == 3:
            return [[[None if (isinstance(v, (float, np.floating)) and np.isnan(v)) else v.item() if hasattr(v, 'item') else v for v in row] for row in slice_2d] for slice_2d in arr]
        out = np.asarray(arr, dtype=object)
        result_list = []
        for row in out:
            result_row = []
            for v in row:
                if isinstance(v, float) and np.isnan(v):
                    result_row.append(None)
                elif isinstance(v, np.floating) and np.isnan(v):
                    result_row.append(None)
                else:
                    result_row.append(v.item() if hasattr(v, 'item') else v)
            result_list.append(result_row)
        return result_list

    return {
        "method": result.method,
        "band_index": 1,
        "raster_type": "temporal",
        "config": result.config,
        "observed_pixel_count": result.valid_observation_count,
        "interpolated_pixel_count": result.interpolated_count,
        "remaining_nodata_count": result.remaining_nodata_count,
        "excluded_pixel_count": result.excluded_count,
        "processing_duration_ms": result.processing_duration_ms,
        "provenance_mask": to_list(result.provenance_cube),
        "reconstructed": to_list(result.reconstructed_cube),
        "quality_mask": None,
        "distance_to_observation": None,
        "kriging_variance": None,
        "variogram_fit": None,
        "extrapolated_pixels": result.extrapolated_pixels,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@router.post("/interpolate/validate")
def validate_interpolation_endpoint(payload: ValidateRequest) -> dict:
    """Cross-validate an interpolation method using withheld observations.

    Runs random holdout or spatial block validation and returns error metrics
    (MAE, RMSE, bias, R² when appropriate) plus per-pixel residuals.
    """
    import numpy as np

    array = np.array(payload.data, dtype=float)
    if array.ndim != 2:
        raise HTTPException(status_code=400, detail="`data` must be a 2D array.")

    try:
        result = validate_interpolation(
            array=array,
            method=payload.method,
            raster_type=payload.raster_type,
            validation_strategy=payload.validation_strategy,
            sample_fraction=payload.sample_fraction,
            n_blocks=payload.n_blocks,
            block_index=payload.block_index,
            seed=payload.seed,
            max_distance_pixels=payload.max_distance_pixels,
            neighborhood=payload.neighborhood,
            edge_handling=payload.edge_handling,
            clip_to_observed_range=payload.clip_to_observed_range,
            window_size=payload.window_size,
            min_valid_neighbors=payload.min_valid_neighbors,
            neighborhood_shape=payload.neighborhood_shape,
            power=payload.power,
            search_radius_pixels=payload.search_radius_pixels,
            max_neighbors=payload.max_neighbors,
            min_neighbors=payload.min_neighbors,
            variogram_model=payload.variogram_model,
            variogram_nugget=payload.variogram_nugget,
            variogram_sill=payload.variogram_sill,
            variogram_range=payload.variogram_range,
        )
        return _validation_result_to_dict(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}") from exc


def _validation_result_to_dict(result) -> dict:
    return {
        "method": result.method,
        "config": result.config,
        "validation_strategy": result.validation_strategy,
        "validation_sample_count": result.validation_sample_count,
        "n_excluded_from_validation": result.n_excluded_from_validation,
        "mae": result.mae,
        "rmse": result.rmse,
        "bias": result.bias,
        "median_absolute_error": result.median_absolute_error,
        "mape": result.mape,
        "r_squared": result.r_squared,
        "reconstruction_percentage": result.reconstruction_percentage,
        "remaining_nodata_percentage": result.remaining_nodata_percentage,
        "excluded_percentage": result.excluded_percentage,
        "processing_duration_ms": result.processing_duration_ms,
        "warnings": result.warnings,
        "limitations": result.limitations,
        "validation_predicted": result.validation_predicted,
        "validation_observed": result.validation_observed,
        "validation_residuals": result.validation_residuals,
        "validation_coords": result.validation_coords,
    }


# ---------------------------------------------------------------------------
# Existing routes continue below (unchanged)
# ---------------------------------------------------------------------------

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
        return await run_heavy_task(
            compute_dem_differencing_from_paths,
            (b1_path, n1, b2_path, n2),
            kwargs={
                "lod_threshold_m": lod_threshold_m,
                "pixel_grid_only": pixel_grid_only,
            },
            timeout=RASTER_PROCESS_TIMEOUT,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        return await run_heavy_task(
            compute_sar_change_from_paths,
            (b1_path, n1, b2_path, n2),
            kwargs={
                "speckle_filter": filt,
                "window_size": window_size,
                "threshold_db": threshold_db,
                "threshold_mode": mode,
                "pixel_grid_only": pixel_grid_only,
            },
            timeout=RASTER_PROCESS_TIMEOUT,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        return await run_heavy_task(
            compute_lulc_from_paths,
            (b1_path, n1, b2_path, n2),
            kwargs={"pixel_grid_only": pixel_grid_only},
            timeout=RASTER_PROCESS_TIMEOUT,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to compute LULC matrix: {exc}") from exc
    finally:
        cleanup_temp_file(b1_path)
        cleanup_temp_file(b2_path)
