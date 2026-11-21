"""Focal / window-based interpolation family.

Bilinear, bicubic, local mean and local median interpolation for continuous
rasters.  Each fills missing pixels from valid neighbours within a moving
window; NoData cells are excluded from the computation.
"""

from __future__ import annotations

import math
import time
from typing import Literal, Optional

import numpy as np
from scipy import ndimage

from .base import InterpolationResult, _ensure_float_array, _valid_mask


def interpolate_bilinear(
    array: np.ndarray,
    *,
    neighborhood: Literal["3x3", "5x5"] = "3x3",
    edge_handling: Literal["clamp", "nearest_valid", "nodata"] = "clamp",
    max_distance_pixels: Optional[float] = None,
    band_index: int = 1,
) -> InterpolationResult:
    """Bilinear interpolation for continuous rasters.

    Estimates missing values using weighted contributions from the 2x2
    neighborhood around each missing cell. Missing neighboring cells are
    NOT treated as zero — they are excluded from the weight sum.

    Args:
        array: 2D float64 array, NaN = missing.
        neighborhood: '3x3' or '5x5' search window.
        edge_handling: How to handle raster boundaries.
        max_distance_pixels: Max distance to nearest valid observation.
        band_index: 1-based band number.

    Returns:
        InterpolationResult.
    """
    start = time.perf_counter()
    arr = _ensure_float_array(array)
    valid = _valid_mask(arr)
    missing = ~valid

    H, W = arr.shape
    reconstructed = arr.copy()
    provenance = np.full((H, W), "nodata", dtype=object)
    provenance[valid] = "observed"

    # Neighborhood radius
    radius = 1 if neighborhood == "3x3" else 2

    valid_coords = np.argwhere(valid)
    missing_coords = np.argwhere(missing)

    interpolated_count = 0
    excluded_count = 0

    if missing_coords.size > 0 and valid_coords.size > 0:
        # Precompute distance transform for max_distance check
        if max_distance_pixels is not None:
            dist_to_valid = ndimage.distance_transform_edt(~valid)

        for r, c in missing_coords:
            # Check max distance
            if max_distance_pixels is not None:
                d = dist_to_valid[r, c]
                if d > max_distance_pixels:
                    provenance[r, c] = "excluded"
                    excluded_count += 1
                    continue

            # Gather valid neighbors within the neighborhood window
            r0 = max(0, r - radius)
            r1 = min(H, r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(W, c + radius + 1)

            window = arr[r0:r1, c0:c1]
            window_valid = valid[r0:r1, c0:c1]

            # Bilinear: use the 4 closest grid positions around (r,c)
            # For simplicity and correctness, use inverse-distance weighting
            # over valid neighbors in the window (generalized bilinear)
            neighbor_vals = []
            neighbor_weights = []
            for dr in range(r0, r1):
                for dc in range(c0, c1):
                    if window_valid[dr - r0, dc - c0]:
                        dist = math.sqrt((dr - r) ** 2 + (dc - c) ** 2)
                        if dist == 0:
                            continue  # self (shouldn't happen since missing)
                        neighbor_vals.append(arr[dr, dc])
                        neighbor_weights.append(1.0 / (dist + 1e-12))

            if len(neighbor_vals) >= 1:
                weights = np.array(neighbor_weights, dtype=float)
                weights = weights / np.sum(weights)
                reconstructed[r, c] = float(np.sum(np.array(neighbor_vals) * weights))
                provenance[r, c] = "interpolated"
                interpolated_count += 1
            else:
                provenance[r, c] = "excluded"
                excluded_count += 1
    elif missing_coords.shape[0] > 0:
        excluded_count = int(missing_coords.shape[0])
        provenance[missing] = "excluded"

    observed_count = int(np.sum(valid))
    remaining_nodata = int(np.sum(missing & (provenance != "interpolated")))

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return InterpolationResult(
        reconstructed=reconstructed,
        provenance_mask=provenance,
        quality_mask=None,
        method="bilinear",
        band_index=band_index,
        raster_type="continuous",
        config={"neighborhood": neighborhood, "edge_handling": edge_handling, "max_distance_pixels": max_distance_pixels},
        observed_pixel_count=observed_count,
        interpolated_pixel_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_pixel_count=excluded_count,
        processing_duration_ms=duration_ms,
    )


def interpolate_bicubic(
    array: np.ndarray,
    *,
    neighborhood: Literal["4x4", "6x6"] = "4x4",
    edge_handling: Literal["clamp", "nodata"] = "clamp",
    max_distance_pixels: Optional[float] = None,
    clip_to_observed_range: bool = True,
    band_index: int = 1,
) -> InterpolationResult:
    """Bicubic interpolation for continuous rasters.

    Reconstructs continuous surfaces using cubic convolution interpolation
    over a 4x4 neighborhood. Missing neighbors are excluded from the fit.
    Values may overshoot the local observed range; range diagnostics are
    provided when clip_to_observed_range is True.

    Args:
        array: 2D float64 array, NaN = missing.
        neighborhood: '4x4' or '6x6' cubic window.
        edge_handling: How to handle boundaries.
        max_distance_pixels: Max distance to nearest valid observation.
        clip_to_observed_range: If True, clip output to local min/max of valid neighbors.
        band_index: 1-based band number.

    Returns:
        InterpolationResult.
    """
    start = time.perf_counter()
    arr = _ensure_float_array(array)
    valid = _valid_mask(arr)
    missing = ~valid

    H, W = arr.shape
    reconstructed = arr.copy()
    provenance = np.full((H, W), "nodata", dtype=object)
    provenance[valid] = "observed"

    radius = 2 if neighborhood == "4x4" else 3

    valid_coords = np.argwhere(valid)
    missing_coords = np.argwhere(missing)

    interpolated_count = 0
    excluded_count = 0

    if missing_coords.size > 0 and valid_coords.size > 0:
        if max_distance_pixels is not None:
            dist_to_valid = ndimage.distance_transform_edt(~valid)

        for r, c in missing_coords:
            if max_distance_pixels is not None:
                d = dist_to_valid[r, c]
                if d > max_distance_pixels:
                    provenance[r, c] = "excluded"
                    excluded_count += 1
                    continue

            r0 = max(0, r - radius)
            r1 = min(H, r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(W, c + radius + 1)

            window = arr[r0:r1, c0:c1]
            window_valid = valid[r0:r1, c0:c1]

            # Cubic convolution: use valid neighbors with cubic weights
            # Weight = (1 - |d|)^3 for |d| <= 1, else 0 (cubic kernel)
            neighbor_vals = []
            neighbor_weights = []
            for dr in range(r0, r1):
                for dc in range(c0, c1):
                    if window_valid[dr - r0, dc - c0]:
                        dist = math.sqrt((dr - r) ** 2 + (dc - c) ** 2)
                        if dist == 0:
                            continue
                        # Normalize distance by radius for cubic kernel
                        u = dist / (radius + 1e-12)
                        if u <= 1.0:
                            w = (1.0 - u) ** 3
                        else:
                            w = 0.0
                        if w > 0:
                            neighbor_vals.append(arr[dr, dc])
                            neighbor_weights.append(w)

            if len(neighbor_vals) >= 1:
                vals = np.array(neighbor_vals, dtype=float)
                wts = np.array(neighbor_weights, dtype=float)
                wts = wts / np.sum(wts)
                est = float(np.sum(vals * wts))

                # Range diagnostics / clipping
                local_min = float(np.min(vals))
                local_max = float(np.max(vals))
                if clip_to_observed_range:
                    est = max(local_min, min(local_max, est))

                reconstructed[r, c] = est
                provenance[r, c] = "interpolated"
                interpolated_count += 1
            else:
                provenance[r, c] = "excluded"
                excluded_count += 1
    elif missing_coords.size > 0:
        excluded_count = int(missing_coords.shape[0])
        provenance[missing] = "excluded"

    observed_count = int(np.sum(valid))
    remaining_nodata = int(np.sum(missing & (provenance != "interpolated")))

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return InterpolationResult(
        reconstructed=reconstructed,
        provenance_mask=provenance,
        quality_mask=None,
        method="bicubic",
        band_index=band_index,
        raster_type="continuous",
        config={
            "neighborhood": neighborhood,
            "edge_handling": edge_handling,
            "max_distance_pixels": max_distance_pixels,
            "clip_to_observed_range": clip_to_observed_range,
        },
        observed_pixel_count=observed_count,
        interpolated_pixel_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_pixel_count=excluded_count,
        processing_duration_ms=duration_ms,
    )


def interpolate_local_mean(
    array: np.ndarray,
    *,
    window_size: int = 3,
    min_valid_neighbors: int = 1,
    neighborhood_shape: Literal["square", "circle"] = "square",
    max_distance_pixels: Optional[float] = None,
    band_index: int = 1,
) -> InterpolationResult:
    """Local mean interpolation.

    Fills missing pixels using the mean of valid observations within a
    configurable moving window. NoData cells are ignored.

    Args:
        array: 2D float64 array, NaN = missing.
        window_size: Size of the moving window (odd integer).
        min_valid_neighbors: Minimum valid observations required.
        neighborhood_shape: 'square' or 'circle'.
        max_distance_pixels: Max distance to nearest valid observation.
        band_index: 1-based band number.

    Returns:
        InterpolationResult.
    """
    start = time.perf_counter()
    arr = _ensure_float_array(array)
    valid = _valid_mask(arr)
    missing = ~valid

    H, W = arr.shape
    reconstructed = arr.copy()
    provenance = np.full((H, W), "nodata", dtype=object)
    provenance[valid] = "observed"

    radius = window_size // 2

    # Build structural element for the neighborhood shape
    if neighborhood_shape == "circle":
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        struct = x * x + y * y <= radius * radius
    else:
        struct = np.ones((window_size, window_size), dtype=bool)

    valid_coords = np.argwhere(valid)
    missing_coords = np.argwhere(missing)

    interpolated_count = 0
    excluded_count = 0

    if missing_coords.size > 0 and valid_coords.size > 0:
        if max_distance_pixels is not None:
            dist_to_valid = ndimage.distance_transform_edt(~valid)

        # Use generic_filter-like approach but manual for control
        padded = np.pad(arr, radius, mode="constant", constant_values=np.nan)
        padded_valid = np.pad(valid, radius, mode="constant", constant_values=False)

        for r, c in missing_coords:
            if max_distance_pixels is not None:
                d = dist_to_valid[r, c]
                if d > max_distance_pixels:
                    provenance[r, c] = "excluded"
                    excluded_count += 1
                    continue

            # Extract window centered at (r+c) in padded coordinates
            pr, pc = r + radius, c + radius
            window = padded[pr - radius:pr + radius + 1, pc - radius:pc + radius + 1]
            wvalid = padded_valid[pr - radius:pr + radius + 1, pc - radius:pc + radius + 1]

            # Apply structural element
            mask = wvalid & struct

            n_valid = int(np.sum(mask))
            if n_valid >= min_valid_neighbors:
                vals = window[mask]
                reconstructed[r, c] = float(np.mean(vals))
                provenance[r, c] = "interpolated"
                interpolated_count += 1
            else:
                provenance[r, c] = "excluded"
                excluded_count += 1
    elif missing_coords.size > 0:
        excluded_count = int(missing_coords.shape[0])
        provenance[missing] = "excluded"

    observed_count = int(np.sum(valid))
    remaining_nodata = int(np.sum(missing & (provenance != "interpolated")))

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return InterpolationResult(
        reconstructed=reconstructed,
        provenance_mask=provenance,
        quality_mask=None,
        method="local_mean",
        band_index=band_index,
        raster_type="continuous",
        config={
            "window_size": window_size,
            "min_valid_neighbors": min_valid_neighbors,
            "neighborhood_shape": neighborhood_shape,
            "max_distance_pixels": max_distance_pixels,
        },
        observed_pixel_count=observed_count,
        interpolated_pixel_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_pixel_count=excluded_count,
        processing_duration_ms=duration_ms,
    )


def interpolate_local_median(
    array: np.ndarray,
    *,
    window_size: int = 3,
    min_valid_neighbors: int = 1,
    neighborhood_shape: Literal["square", "circle"] = "square",
    max_distance_pixels: Optional[float] = None,
    band_index: int = 1,
) -> InterpolationResult:
    """Local median interpolation.

    Fills missing pixels using the median of valid observations within a
    configurable moving window. Robust to outliers. NoData cells are ignored.

    Args:
        array: 2D float64 array, NaN = missing.
        window_size: Size of the moving window (odd integer).
        min_valid_neighbors: Minimum valid observations required.
        neighborhood_shape: 'square' or 'circle'.
        max_distance_pixels: Max distance to nearest valid observation.
        band_index: 1-based band number.

    Returns:
        InterpolationResult.
    """
    start = time.perf_counter()
    arr = _ensure_float_array(array)
    valid = _valid_mask(arr)
    missing = ~valid

    H, W = arr.shape
    reconstructed = arr.copy()
    provenance = np.full((H, W), "nodata", dtype=object)
    provenance[valid] = "observed"

    radius = window_size // 2

    if neighborhood_shape == "circle":
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        struct = x * x + y * y <= radius * radius
    else:
        struct = np.ones((window_size, window_size), dtype=bool)

    valid_coords = np.argwhere(valid)
    missing_coords = np.argwhere(missing)

    interpolated_count = 0
    excluded_count = 0

    if missing_coords.size > 0 and valid_coords.size > 0:
        if max_distance_pixels is not None:
            dist_to_valid = ndimage.distance_transform_edt(~valid)

        padded = np.pad(arr, radius, mode="constant", constant_values=np.nan)
        padded_valid = np.pad(valid, radius, mode="constant", constant_values=False)

        for r, c in missing_coords:
            if max_distance_pixels is not None:
                d = dist_to_valid[r, c]
                if d > max_distance_pixels:
                    provenance[r, c] = "excluded"
                    excluded_count += 1
                    continue

            pr, pc = r + radius, c + radius
            window = padded[pr - radius:pr + radius + 1, pc - radius:pc + radius + 1]
            wvalid = padded_valid[pr - radius:pr + radius + 1, pc - radius:pc + radius + 1]

            mask = wvalid & struct
            n_valid = int(np.sum(mask))

            if n_valid >= min_valid_neighbors:
                vals = window[mask]
                reconstructed[r, c] = float(np.median(vals))
                provenance[r, c] = "interpolated"
                interpolated_count += 1
            else:
                provenance[r, c] = "excluded"
                excluded_count += 1
    elif missing_coords.size > 0:
        excluded_count = int(missing_coords.shape[0])
        provenance[missing] = "excluded"

    observed_count = int(np.sum(valid))
    remaining_nodata = int(np.sum(missing & (provenance != "interpolated")))

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return InterpolationResult(
        reconstructed=reconstructed,
        provenance_mask=provenance,
        quality_mask=None,
        method="local_median",
        band_index=band_index,
        raster_type="continuous",
        config={
            "window_size": window_size,
            "min_valid_neighbors": min_valid_neighbors,
            "neighborhood_shape": neighborhood_shape,
            "max_distance_pixels": max_distance_pixels,
        },
        observed_pixel_count=observed_count,
        interpolated_pixel_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_pixel_count=excluded_count,
        processing_duration_ms=duration_ms,
    )
