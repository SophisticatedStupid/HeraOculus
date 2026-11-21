"""Shared types and low-level helpers for the raster interpolation engine.

This module holds the data structures and small utilities that are shared
across all interpolation method families.  It has no external dependencies
beyond NumPy/SciPy and the standard library.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata  # noqa: F401  (re-exported for backwards compatibility)


@dataclass
class InterpolationResult:
    """Result of a single-band interpolation operation."""
    # Output arrays (same shape as input)
    reconstructed: np.ndarray  # float64, NaN where still missing
    provenance_mask: np.ndarray  # 'observed' | 'interpolated' | 'nodata' | 'excluded'
    quality_mask: Optional[np.ndarray]  # per-pixel quality diagnostic (0-1), None if unsupported

    # Method & configuration echo
    method: str
    band_index: int
    raster_type: str
    config: dict

    # Statistics
    observed_pixel_count: int
    interpolated_pixel_count: int
    remaining_nodata_count: int
    excluded_pixel_count: int
    processing_duration_ms: float

    # Per-pixel distance to nearest observation (for IDW/NN diagnostics)
    distance_to_observation: Optional[np.ndarray] = None


@dataclass
class TemporalInterpolationResult:
    """Result of temporal interpolation on a raster time series cube."""
    reconstructed_cube: np.ndarray  # (T, H, W) float64, NaN = still missing
    provenance_cube: np.ndarray  # (T, H, W) object: 'observed'|'interpolated'|'excluded'|'nodata'
    method: str
    config: dict
    valid_observation_count: int
    interpolated_count: int
    remaining_nodata_count: int
    excluded_count: int
    processing_duration_ms: float
    extrapolated_pixels: list[dict]  # pixels where extrapolation occurred


def _ensure_float_array(array: np.ndarray) -> np.ndarray:
    """Convert input to float64, mapping NaN/inf consistently."""
    arr = np.asarray(array, dtype=np.float64)
    return arr


def _valid_mask(array: np.ndarray) -> np.ndarray:
    """Return True where pixel is finite (observed)."""
    return np.isfinite(array)


def _nearest_valid_indices(
    missing_coords: np.ndarray,
    valid_coords: np.ndarray,
    max_distance_pixels: Optional[float] = None,
) -> np.ndarray:
    """For each missing coordinate, find index of nearest valid coordinate.

    Returns array of indices into valid_coords, or -1 if no valid neighbor
    within max_distance_pixels.
    """
    if valid_coords.size == 0:
        return np.full(missing_coords.shape[0], -1, dtype=int)
    if missing_coords.size == 0:
        return np.empty(0, dtype=int)

    # Compute pairwise squared Euclidean distances
    # missing: (M, 2), valid: (V, 2)
    diff = missing_coords[:, None, :] - valid_coords[None, :, :]  # (M, V, 2)
    dist_sq = np.sum(diff * diff, axis=2)  # (M, V)
    nearest_idx = np.argmin(dist_sq, axis=1)  # (M,)
    nearest_dist = np.sqrt(dist_sq[np.arange(missing_coords.shape[0]), nearest_idx])

    if max_distance_pixels is not None:
        nearest_idx = np.where(nearest_dist <= max_distance_pixels, nearest_idx, -1)

    return nearest_idx
