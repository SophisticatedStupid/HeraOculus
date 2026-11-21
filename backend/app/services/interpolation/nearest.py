"""Nearest neighbor interpolation family.

Assigns each missing pixel the value of its closest valid neighbor.
Supports both continuous and categorical rasters; categorical values are
preserved exactly (no blending).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import (
    InterpolationResult,
    _ensure_float_array,
    _nearest_valid_indices,
    _valid_mask,
)


def interpolate_nearest_neighbor(
    array: np.ndarray,
    *,
    max_distance_pixels: Optional[float] = None,
    raster_type: str = "continuous",
    band_index: int = 1,
) -> InterpolationResult:
    """Nearest-neighbor interpolation.

    Assigns each missing pixel the value of its closest valid neighbor.
    Supports both continuous and categorical rasters. Categorical values are
    preserved exactly (no blending).

    Args:
        array: 2D float64 array, NaN = missing.
        max_distance_pixels: Max search radius in pixels. None = unlimited.
        raster_type: 'continuous' or 'categorical'.
        band_index: 1-based band number.

    Returns:
        InterpolationResult with reconstructed array and provenance mask.
    """
    start = time.perf_counter()
    arr = _ensure_float_array(array)
    valid = _valid_mask(arr)
    missing = ~valid

    H, W = arr.shape
    reconstructed = arr.copy()
    provenance = np.full((H, W), "nodata", dtype=object)
    provenance[valid] = "observed"

    valid_coords = np.argwhere(valid)  # (N, 2) [row, col]
    missing_coords = np.argwhere(missing)  # (M, 2) [row, col]

    interpolated_count = 0
    excluded_count = 0

    if missing_coords.shape[0] > 0 and valid_coords.shape[0] > 0:
        nearest_idx = _nearest_valid_indices(
            missing_coords, valid_coords, max_distance_pixels
        )
        for i, (r, c) in enumerate(missing_coords):
            idx = nearest_idx[i]
            if idx >= 0:
                vr, vc = valid_coords[idx]
                reconstructed[r, c] = arr[vr, vc]
                provenance[r, c] = "interpolated"
                interpolated_count += 1
            else:
                provenance[r, c] = "excluded"
                excluded_count += 1
    elif missing_coords.shape[0] > 0:
        # No valid pixels at all — all missing are excluded
        excluded_count = int(missing_coords.shape[0])
        provenance[missing] = "excluded"

    observed_count = int(np.sum(valid))
    remaining_nodata = int(np.sum((provenance == "nodata") | (provenance == "excluded")))
    # Actually: nodata means still missing and not interpolated/excluded
    remaining_nodata = int(np.sum(missing & (provenance != "interpolated")))

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return InterpolationResult(
        reconstructed=reconstructed,
        provenance_mask=provenance,
        quality_mask=None,
        method="nearest_neighbor",
        band_index=band_index,
        raster_type=raster_type,
        config={"max_distance_pixels": max_distance_pixels},
        observed_pixel_count=observed_count,
        interpolated_pixel_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_pixel_count=excluded_count,
        processing_duration_ms=duration_ms,
    )
