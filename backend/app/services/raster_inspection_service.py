"""Raster inspection and NoData diagnostics service.

Detects, characterizes, and reports missing data regions in geospatial rasters.
Produces a missing-data diagnostic report including connected-component analysis,
spatial distribution, and distance-to-valid-observation metrics.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage


@dataclass
class MissingRegionStats:
    """Statistics for a single connected missing-data region."""
    id: int
    pixel_count: int
    area_map_units: float
    bbox: tuple[int, int, int, int]  # (row_min, row_max, col_min, col_max)
    centroid: tuple[float, float]  # (row, col)
    is_edge: bool
    surrounded_by_valid: bool
    min_distance_to_valid: float
    mean_distance_to_valid: float


@dataclass
class RasterInspectionResult:
    """Complete raster inspection and missing-data diagnostic report."""
    # Basic metadata
    width: int
    height: int
    bands: int
    crs: Optional[str]
    transform: Optional[dict]
    pixel_size_x: Optional[float]
    pixel_size_y: Optional[float]
    extent: Optional[tuple[float, float, float, float]]  # (west, south, east, north)
    data_type: str
    nodata_value: Optional[float]
    has_alpha_band: bool
    band_index: int

    # Valid / missing counts
    valid_pixel_count: int
    missing_pixel_count: int
    missing_percentage: float
    valid_percentage: float

    # Per-band statistics
    band_stats: list[dict]

    # Raster classification
    raster_type: Literal["continuous", "categorical", "temporal", "unknown"]

    # Missing region diagnostics
    missing_region_count: int
    missing_region_stats: list[MissingRegionStats]
    largest_missing_region_pixels: int
    average_missing_region_size: float
    missing_region_size_distribution: dict  # "isolated": count, "small_cluster": count, "large_gap": count, etc.
    isolated_missing_pixels: int
    edge_gaps: int
    interior_gaps: int

    # Spatial distribution
    distance_to_valid_stats: dict  # min, max, mean, median of distance from missing to nearest valid
    missing_spatial_distribution: dict  # quadrant distribution

    # Processing metadata
    processing_duration_ms: float
    inspected_at: str


def _classify_raster_type(array: np.ndarray, user_confirmed: Optional[str] = None) -> str:
    """Classify raster as continuous, categorical, temporal, or unknown.

    Does not infer semantics from dtype alone; uses user confirmation when provided,
    otherwise falls back to dtype-based heuristic (integer -> categorical, float -> continuous).
    """
    if user_confirmed in {"continuous", "categorical", "temporal", "unknown"}:
        return user_confirmed
    if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        return "categorical"
    return "continuous"


def _connected_components(
    mask: np.ndarray,
) -> tuple[int, list[tuple[int, int, int, int]], list[np.ndarray]]:
    """Label connected missing regions using 8-connectivity.

    Returns (num_features, slices_list, labeled_array).
    """
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_features = ndimage.label(mask, structure=structure)
    slices = ndimage.find_objects(labeled)
    return num_features, slices, labeled


def _is_edge_region(
    labeled: np.ndarray,
    region_id: int,
    shape: tuple[int, int],
) -> bool:
    """Check if a labeled region touches the raster boundary."""
    rows, cols = np.where(labeled == region_id)
    H, W = shape
    return bool(
        np.any(rows == 0)
        or np.any(rows == H - 1)
        or np.any(cols == 0)
        or np.any(cols == W - 1)
    )


def _is_surrounded_by_valid(
    labeled: np.ndarray,
    region_id: int,
    valid_mask: np.ndarray,
) -> bool:
    """Check if a missing region is completely surrounded by valid pixels."""
    rows, cols = np.where(labeled == region_id)
    if len(rows) == 0:
        return False
    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())
    # Check the 1-pixel dilated boundary around the region's bounding box
    pad = 1
    r0 = max(0, r_min - pad)
    r1 = min(labeled.shape[0], r_max + pad + 1)
    c0 = max(0, c_min - pad)
    c1 = min(labeled.shape[1], c_max + pad + 1)
    boundary_rows = list(range(r0, r1))
    boundary_cols = list(range(c0, c1))
    # Any valid pixel immediately adjacent (including diagonals) to the region
    for r in boundary_rows:
        for c in boundary_cols:
            if labeled[r, c] != region_id and valid_mask[r, c]:
                return True
    return False


def _compute_distance_to_valid(
    missing_mask: np.ndarray,
    valid_mask: np.ndarray,
    pixel_size_x: Optional[float],
    pixel_size_y: Optional[float],
) -> dict:
    """Compute distance from each missing pixel to nearest valid observation."""
    if not np.any(missing_mask) or not np.any(valid_mask):
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "count": 0,
        }
    # Distance transform: distance from each missing pixel to nearest valid pixel
    # ndimage.distance_transform_edt computes distance to nearest True (non-zero) pixel
    distances = ndimage.distance_transform_edt(~valid_mask)
    missing_distances = distances[missing_mask]
    if missing_distances.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "count": 0,
        }
    # Convert pixel distances to map units if resolution is available
    if pixel_size_x and pixel_size_y:
        # Use average pixel size for isotropic approximation
        avg_ps = (pixel_size_x + pixel_size_y) / 2.0
        missing_distances = missing_distances * avg_ps
    return {
        "min": round(float(np.min(missing_distances)), 4),
        "max": round(float(np.max(missing_distances)), 4),
        "mean": round(float(np.mean(missing_distances)), 4),
        "median": round(float(np.median(missing_distances)), 4),
        "count": int(missing_distances.size),
    }


def _spatial_distribution_quadrants(
    missing_mask: np.ndarray,
) -> dict:
    """Distribute missing pixels across four spatial quadrants."""
    H, W = missing_mask.shape
    if H == 0 or W == 0:
        return {"nw": 0, "ne": 0, "sw": 0, "se": 0, "center": 0}
    mid_r, mid_c = H // 2, W // 2
    nw = int(np.sum(missing_mask[:mid_r, :mid_c]))
    ne = int(np.sum(missing_mask[:mid_r, mid_c:]))
    sw = int(np.sum(missing_mask[mid_r:, :mid_c]))
    se = int(np.sum(missing_mask[mid_r:, mid_c:]))
    center = int(np.sum(missing_mask)) - (nw + ne + sw + se)
    return {"nw": nw, "ne": ne, "sw": sw, "se": se, "center": center}


def inspect_raster(
    *,
    array: np.ndarray,
    crs: Optional[str] = None,
    transform=None,
    nodata: Optional[float] = None,
    band_index: int = 1,
    user_confirmed_type: Optional[str] = None,
    pixel_size_x: Optional[float] = None,
    pixel_size_y: Optional[float] = None,
    extent: Optional[tuple[float, float, float, float]] = None,
) -> RasterInspectionResult:
    """Inspect a raster array and produce a complete missing-data diagnostic report.

    Args:
        array: 2D numpy array (float64 with NaN for missing).
        crs: Coordinate reference system string.
        transform: rasterio Affine transform.
        nodata: NoData value from source raster.
        band_index: 1-based band number being inspected.
        user_confirmed_type: User-confirmed raster type (continuous/categorical/temporal/unknown).
        pixel_size_x, pixel_size_y: Pixel dimensions in map units.
        extent: (west, south, east, north) bounding box.

    Returns:
        RasterInspectionResult with full diagnostics.
    """
    import time

    start = time.perf_counter()

    H, W = array.shape
    valid_mask = np.isfinite(array)
    missing_mask = ~valid_mask

    valid_count = int(np.sum(valid_mask))
    missing_count = int(np.sum(missing_mask))
    total = H * W
    missing_pct = round(100.0 * missing_count / total, 4) if total > 0 else 0.0
    valid_pct = round(100.0 * valid_count / total, 4) if total > 0 else 0.0

    # Per-band statistics (single band passed in)
    finite_vals = array[valid_mask]
    if finite_vals.size > 0:
        band_stat = {
            "band": band_index,
            "min": round(float(np.min(finite_vals)), 6),
            "max": round(float(np.max(finite_vals)), 6),
            "mean": round(float(np.mean(finite_vals)), 6),
            "std": round(float(np.std(finite_vals)), 6),
            "valid_count": valid_count,
            "missing_count": missing_count,
        }
    else:
        band_stat = {
            "band": band_index,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "valid_count": 0,
            "missing_count": missing_count,
        }

    raster_type = _classify_raster_type(array, user_confirmed_type)

    # Connected component analysis of missing regions
    num_features, slices, labeled = _connected_components(missing_mask)

    region_stats: list[MissingRegionStats] = []
    isolated_count = 0
    small_cluster_count = 0
    large_gap_count = 0
    edge_gap_count = 0
    interior_gap_count = 0
    entirely_missing_count = 0
    largest_region_pixels = 0
    region_sizes = []

    # Distance from missing pixels to nearest valid
    dist_stats = _compute_distance_to_valid(
        missing_mask, valid_mask, pixel_size_x, pixel_size_y
    )

    # Compute per-pixel distances for region-level stats
    if np.any(missing_mask) and np.any(valid_mask):
        all_distances = ndimage.distance_transform_edt(~valid_mask)
    else:
        all_distances = np.zeros_like(array, dtype=float)

    for i in range(1, num_features + 1):
        region_slice = slices[i - 1]
        if region_slice is None:
            continue
        region_mask = (labeled == i)
        pixel_count = int(np.sum(region_mask))
        region_sizes.append(pixel_count)
        if pixel_count > largest_region_pixels:
            largest_region_pixels = pixel_count

        rows, cols = np.where(region_mask)
        r_min, r_max = int(rows.min()), int(rows.max())
        c_min, c_max = int(cols.min()), int(cols.max())
        centroid = (float(np.mean(rows)), float(np.mean(cols)))

        is_edge = _is_edge_region(labeled, i, (H, W))
        surrounded = _is_surrounded_by_valid(labeled, i, valid_mask)

        region_distances = all_distances[region_mask]
        if region_distances.size > 0:
            min_dist = round(float(np.min(region_distances)), 4)
            mean_dist = round(float(np.mean(region_distances)), 4)
        else:
            min_dist = 0.0
            mean_dist = 0.0

        # Convert to map units
        if pixel_size_x and pixel_size_y:
            avg_ps = (pixel_size_x + pixel_size_y) / 2.0
            min_dist = round(min_dist * avg_ps, 4)
            mean_dist = round(mean_dist * avg_ps, 4)

        region_stats.append(
            MissingRegionStats(
                id=i,
                pixel_count=pixel_count,
                area_map_units=round(float(pixel_count * (pixel_size_x or 1.0) * (pixel_size_y or 1.0)), 4),
                bbox=(r_min, r_max, c_min, c_max),
                centroid=centroid,
                is_edge=is_edge,
                surrounded_by_valid=surrounded,
                min_distance_to_valid=min_dist,
                mean_distance_to_valid=mean_dist,
            )
        )

        # Classify region type
        if pixel_count == 1:
            isolated_count += 1
        elif pixel_count <= 9:  # 3x3 or smaller
            small_cluster_count += 1
        else:
            large_gap_count += 1

        if is_edge:
            edge_gap_count += 1
        else:
            interior_gap_count += 1

        if pixel_count == total:
            entirely_missing_count += 1

    avg_region_size = round(float(np.mean(region_sizes)), 4) if region_sizes else 0.0

    size_distribution = {
        "isolated_pixels": isolated_count,
        "small_clusters": small_cluster_count,
        "large_gaps": large_gap_count,
        "entirely_missing": entirely_missing_count,
    }

    spatial_dist = _spatial_distribution_quadrants(missing_mask)

    # Transform to dict for JSON serialization
    transform_dict = None
    if transform is not None:
        try:
            transform_dict = {
                "a": float(transform.a),
                "b": float(transform.b),
                "c": float(transform.c),
                "d": float(transform.d),
                "e": float(transform.e),
                "f": float(transform.f),
            }
        except Exception:
            transform_dict = None

    # Data type string
    dtype_str = str(array.dtype)
    if np.issubdtype(array.dtype, np.integer):
        dtype_str = "integer"
    elif np.issubdtype(array.dtype, np.floating):
        dtype_str = "float"
    elif np.issubdtype(array.dtype, np.bool_):
        dtype_str = "boolean"

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    from datetime import datetime, timezone

    return RasterInspectionResult(
        width=W,
        height=H,
        bands=1,
        crs=crs,
        transform=transform_dict,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        extent=extent,
        data_type=dtype_str,
        nodata_value=nodata,
        has_alpha_band=False,
        band_index=band_index,
        valid_pixel_count=valid_count,
        missing_pixel_count=missing_count,
        missing_percentage=missing_pct,
        valid_percentage=valid_pct,
        band_stats=[band_stat],
        raster_type=raster_type,
        missing_region_count=num_features,
        missing_region_stats=region_stats,
        largest_missing_region_pixels=largest_region_pixels,
        average_missing_region_size=avg_region_size,
        missing_region_size_distribution=size_distribution,
        isolated_missing_pixels=isolated_count,
        edge_gaps=edge_gap_count,
        interior_gaps=interior_gap_count,
        distance_to_valid_stats=dist_stats,
        missing_spatial_distribution=spatial_dist,
        processing_duration_ms=duration_ms,
        inspected_at=datetime.now(timezone.utc).isoformat(),
    )


def inspect_raster_from_file(
    *,
    file_path: str,
    band_index: int = 1,
    user_confirmed_type: Optional[str] = None,
) -> RasterInspectionResult:
    """Inspect a raster file on disk."""
    import os
    import time

    start = time.perf_counter()

    with rasterio.open(file_path) as src:
        arr = src.read(band_index).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        crs = str(src.crs) if src.crs else None
        transform = src.transform
        width = src.width
        height = src.height
        res = src.res
        count = src.count
        has_alpha = src.count > 1 and src.profile.get("photometric") == "alpha"

        # Compute extent from transform
        west = transform.c
        north = transform.f
        pixel_w, pixel_h = res
        east = west + width * pixel_w
        south = north - height * abs(pixel_h)
        extent = (west, south, east, north)

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    result = inspect_raster(
        array=arr,
        crs=crs,
        transform=transform,
        nodata=nodata,
        band_index=band_index,
        user_confirmed_type=user_confirmed_type,
        pixel_size_x=pixel_w,
        pixel_size_y=abs(pixel_h),
        extent=extent,
    )
    # Override with multi-band info
    result.bands = count
    result.has_alpha_band = has_alpha
    result.processing_duration_ms = duration_ms
    return result


def inspect_raster_from_bytes(
    *,
    raw_bytes: bytes,
    filename: str,
    band_index: int = 1,
    user_confirmed_type: Optional[str] = None,
) -> RasterInspectionResult:
    """Inspect raster from in-memory bytes via a temporary file."""
    import os as _os

    suffix = Path(filename).suffix.lower() or ".tif"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with _os.fdopen(fd, "wb") as handle:
            handle.write(raw_bytes)
        return inspect_raster_from_file(
            file_path=tmp_path,
            band_index=band_index,
            user_confirmed_type=user_confirmed_type,
        )
    finally:
        try:
            _os.remove(tmp_path)
        except OSError:
            pass
