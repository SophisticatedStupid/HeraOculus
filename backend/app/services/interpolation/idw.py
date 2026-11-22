"""Inverse Distance Weighting (IDW) interpolation.

Z(x) = sum(w_i * z_i) / sum(w_i), where w_i = 1 / d_i^p.

Distances are computed in *map units*:

* Projected CRS: Euclidean distance between pixel coordinates (the coordinates
  are already in a linear unit such as metres).
* Geographic CRS: great-circle (haversine) distance between pixel centroids,
  expressed in metres.  Pixel centroids are placed relative to the raster
  centre using ``pixel_size_x`` / ``pixel_size_y`` (degrees per pixel); this
  preserves the latitude dependence of the longitude scale (``cos(lat)``) and
  never mixes degrees with metres.  The absolute longitude/latitude offset is
  arbitrary (the raster is placed relative to its centre), but relative
  great-circle distances between pixels are computed correctly on the sphere.
  True absolute geodesics require the raster geographic transform origin, which
  is not available to the interpolator.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from .base import InterpolationResult, _ensure_float_array, _valid_mask


# Mean Earth radius in metres.
_EARTH_RADIUS_M = 6371000.0


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points (degrees) in metres."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    h = min(1.0, h)
    return 2.0 * _EARTH_RADIUS_M * math.asin(math.sqrt(h))


def interpolate_idw(
    array: np.ndarray,
    *,
    power: float = 2.0,
    search_radius_pixels: Optional[float] = None,
    max_neighbors: int = 12,
    min_neighbors: int = 1,
    geographic_crs: bool = False,
    pixel_size_x: Optional[float] = None,
    pixel_size_y: Optional[float] = None,
    band_index: int = 1,
) -> InterpolationResult:
    """Inverse Distance Weighting (IDW) interpolation.

    Z(x) = sum(w_i * z_i) / sum(w_i), where w_i = 1 / d_i^p.

    Args:
        array: 2D float64 array, NaN = missing.
        power: IDW power parameter p.
        search_radius_pixels: Max search radius in pixels.
        max_neighbors: Maximum number of neighbors to use.
        min_neighbors: Minimum valid neighbors required (else NoData).
        geographic_crs: If True, use great-circle (haversine) distance in
            metres; otherwise Euclidean distance in pixel coordinates.
        pixel_size_x, pixel_size_y: Pixel dimensions.  For a geographic CRS
            these are degrees per pixel and are converted to metres via
            haversine; for a projected CRS they are map units per pixel.
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

    # Build coordinate grids (row, col).
    valid_coords = np.argwhere(valid)  # (N, 2) [row, col]
    missing_coords = np.argwhere(missing)  # (M, 2) [row, col]

    interpolated_count = 0
    excluded_count = 0
    distance_to_obs = np.full((H, W), np.nan, dtype=float)

    # Raster-centre anchor for geographic centroid positions (degrees).
    center_r = (H - 1) / 2.0
    center_c = (W - 1) / 2.0

    if missing_coords.size > 0 and valid_coords.size > 0:
        valid_vals = arr[valid]

        # Precompute pixel centroid geographic positions (degrees) when needed.
        if geographic_crs and pixel_size_x and pixel_size_y:
            valid_lat = (center_r - valid_coords[:, 0]) * pixel_size_y
            valid_lon = (valid_coords[:, 1] - center_c) * pixel_size_x

        for r, c in missing_coords:
            # Compute distances to all valid pixels.
            dr = valid_coords[:, 0] - r
            dc = valid_coords[:, 1] - c

            if geographic_crs and pixel_size_x and pixel_size_y:
                # Great-circle (haversine) distance in metres between centroids.
                miss_lat = (center_r - r) * pixel_size_y
                miss_lon = (c - center_c) * pixel_size_x
                dist = np.array(
                    [
                        _haversine_meters(miss_lat, miss_lon, float(vla), float(vlo))
                        for vla, vlo in zip(valid_lat, valid_lon)
                    ],
                    dtype=float,
                )
            else:
                dist = np.sqrt(dr * dr + dc * dc)

            # Apply search radius filter.
            if search_radius_pixels is not None:
                within_radius = dist <= search_radius_pixels
                if not np.any(within_radius):
                    provenance[r, c] = "excluded"
                    excluded_count += 1
                    continue
                dist = dist[within_radius]
                candidate_vals = valid_vals[within_radius]
                candidate_coords = valid_coords[within_radius]
            else:
                candidate_vals = valid_vals
                candidate_coords = valid_coords

            # Sort by distance, take max_neighbors.
            order = np.argsort(dist)
            dist = dist[order]
            candidate_vals = candidate_vals[order]

            if len(dist) > max_neighbors:
                dist = dist[:max_neighbors]
                candidate_vals = candidate_vals[:max_neighbors]

            if len(dist) < min_neighbors:
                provenance[r, c] = "excluded"
                excluded_count += 1
                continue

            # Handle zero-distance observations (duplicate coordinates).
            zero_mask = dist <= 1e-12
            if np.any(zero_mask):
                reconstructed[r, c] = float(candidate_vals[zero_mask][0])
                distance_to_obs[r, c] = 0.0
            else:
                # IDW weights: w_i = 1 / d_i^p
                weights = 1.0 / np.power(dist, power)
                weight_sum = float(np.sum(weights))
                if weight_sum <= 0:
                    provenance[r, c] = "excluded"
                    excluded_count += 1
                    continue
                reconstructed[r, c] = float(
                    np.sum(candidate_vals * weights) / weight_sum
                )
                distance_to_obs[r, c] = float(dist[0])

            provenance[r, c] = "interpolated"
            interpolated_count += 1
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
        method="idw",
        band_index=band_index,
        raster_type="continuous",
        config={
            "power": power,
            "search_radius_pixels": search_radius_pixels,
            "max_neighbors": max_neighbors,
            "min_neighbors": min_neighbors,
            "geographic_crs": geographic_crs,
        },
        observed_pixel_count=observed_count,
        interpolated_pixel_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_pixel_count=excluded_count,
        processing_duration_ms=duration_ms,
        distance_to_observation=distance_to_obs,
    )
