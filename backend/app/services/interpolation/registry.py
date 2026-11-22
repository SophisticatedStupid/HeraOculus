"""Method registry and dispatch entry points for raster interpolation.

Holds ``INTERPOLATION_METHODS`` (method metadata) and the public
``run_interpolation`` / ``run_temporal_interpolation`` dispatchers that route
to the concrete algorithm implementations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import InterpolationResult, TemporalInterpolationResult
from .focal import (
    interpolate_bicubic,
    interpolate_bilinear,
    interpolate_local_mean,
    interpolate_local_median,
)
from .idw import interpolate_idw
from .kriging import interpolate_ordinary_kriging
from .nearest import interpolate_nearest_neighbor
from .temporal import interpolate_temporal


INTERPOLATION_METHODS: dict[str, dict] = {
    "nearest_neighbor": {
        "label": "Nearest Neighbor",
        "description": "Assign each missing pixel the value of its closest valid neighbor. Preserves categorical class labels exactly.",
        "supported_raster_types": ["continuous", "categorical"],
        "supports_max_distance": True,
        "supports_categorical": True,
        "default_config": {"max_distance_pixels": None},
    },
    "bilinear": {
        "label": "Bilinear",
        "description": "Estimate continuous values using weighted contributions from neighboring cells. Missing neighbors are excluded from the weight sum, not treated as zero.",
        "supported_raster_types": ["continuous"],
        "supports_max_distance": True,
        "supports_categorical": False,
        "default_config": {"neighborhood": "3x3", "edge_handling": "clamp", "max_distance_pixels": None},
    },
    "bicubic": {
        "label": "Bicubic",
        "description": "Reconstruct continuous surfaces using cubic convolution interpolation. Values may overshoot the local observed range; range diagnostics provided.",
        "supported_raster_types": ["continuous"],
        "supports_max_distance": True,
        "supports_categorical": False,
        "default_config": {"neighborhood": "4x4", "edge_handling": "clamp", "max_distance_pixels": None, "clip_to_observed_range": True},
    },
    "local_mean": {
        "label": "Local Mean",
        "description": "Fill small missing regions using the mean of valid observations within a moving window. Robust for small gaps.",
        "supported_raster_types": ["continuous"],
        "supports_max_distance": True,
        "supports_categorical": False,
        "default_config": {"window_size": 3, "min_valid_neighbors": 1, "neighborhood_shape": "square", "max_distance_pixels": None},
    },
    "local_median": {
        "label": "Local Median",
        "description": "Fill small missing regions using the median of valid observations within a moving window. Robust to outliers.",
        "supported_raster_types": ["continuous"],
        "supports_max_distance": True,
        "supports_categorical": False,
        "default_config": {"window_size": 3, "min_valid_neighbors": 1, "neighborhood_shape": "square", "max_distance_pixels": None},
    },
    "idw": {
        "label": "Inverse Distance Weighting (IDW)",
        "description": "Estimate missing values through distance-weighted neighboring observations. Configurable power, search radius, and neighbor limits. Uses geodesic-aware distance for geographic CRS.",
        "supported_raster_types": ["continuous"],
        "supports_max_distance": True,
        "supports_categorical": False,
        "default_config": {"power": 2.0, "search_radius_pixels": None, "max_neighbors": 12, "min_neighbors": 1, "geographic_crs": False},
    },
    "ordinary_kriging": {
        "label": "Ordinary Kriging",
        "description": "Estimate missing continuous values using spatial autocorrelation modeled by an empirical variogram. Produces predictions and kriging variance uncertainty raster.",
        "supported_raster_types": ["continuous"],
        "supports_max_distance": True,
        "supports_categorical": False,
        "default_config": {"variogram_model": "spherical", "search_radius_pixels": None, "max_neighbors": 12, "min_neighbors": 3},
    },
    # ── Temporal methods (operate on 3D cubes, not 2D arrays) ────────────
    "temporal_linear": {
        "label": "Linear Temporal Interpolation",
        "description": "Fill missing values using linear interpolation between valid observations at other dates. True interpolation only between bracketing observations.",
        "supported_raster_types": ["temporal"],
        "supports_max_distance": False,
        "supports_categorical": False,
        "default_config": {"max_gap_steps": None, "allow_extrapolation": False},
    },
    "temporal_nearest_date": {
        "label": "Nearest-Date Temporal Interpolation",
        "description": "Fill missing values using the observation from the nearest date in time.",
        "supported_raster_types": ["temporal"],
        "supports_max_distance": False,
        "supports_categorical": False,
        "default_config": {"max_gap_steps": None, "allow_extrapolation": False},
    },
    "temporal_forward_fill": {
        "label": "Forward Fill",
        "description": "Fill missing values using the last valid observation from an earlier date.",
        "supported_raster_types": ["temporal"],
        "supports_max_distance": False,
        "supports_categorical": False,
        "default_config": {"max_gap_steps": None, "allow_extrapolation": False},
    },
    "temporal_backward_fill": {
        "label": "Backward Fill",
        "description": "Fill missing values using the first valid observation from a later date.",
        "supported_raster_types": ["temporal"],
        "supports_max_distance": False,
        "supports_categorical": False,
        "default_config": {"max_gap_steps": None, "allow_extrapolation": False},
    },
    "temporal_local_mean": {
        "label": "Local Temporal Mean",
        "description": "Fill missing values using the mean of all valid observations across the time series at each pixel.",
        "supported_raster_types": ["temporal"],
        "supports_max_distance": False,
        "supports_categorical": False,
        "default_config": {"max_gap_steps": None, "allow_extrapolation": False},
    },
}


def run_interpolation(
    array: np.ndarray,
    *,
    method: str,
    band_index: int = 1,
    raster_type: str = "continuous",
    max_distance_pixels: Optional[float] = None,
    geographic_crs: bool = False,
    pixel_size_x: Optional[float] = None,
    pixel_size_y: Optional[float] = None,
    **kwargs,
) -> InterpolationResult:
    """Dispatch to the selected interpolation algorithm.

    Args:
        array: 2D float64 array, NaN = missing.
        method: Interpolation method name.
        band_index: 1-based band number.
        raster_type: 'continuous' or 'categorical'.
        max_distance_pixels: Max search radius in pixels.
        geographic_crs: Whether the raster uses geographic coordinates.
        pixel_size_x, pixel_size_y: Pixel dimensions in map units.
        **kwargs: Method-specific parameters.

    Returns:
        InterpolationResult.

    Raises:
        ValueError: If method is unknown or incompatible with raster_type.
    """
    if method not in INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown interpolation method '{method}'. "
            f"Available: {sorted(INTERPOLATION_METHODS.keys())}"
        )

    method_info = INTERPOLATION_METHODS[method]
    if raster_type not in method_info["supported_raster_types"]:
        raise ValueError(
            f"Method '{method}' is not compatible with raster type '{raster_type}'. "
            f"Supported types: {method_info['supported_raster_types']}"
        )

    if method == "nearest_neighbor":
        return interpolate_nearest_neighbor(
            array, max_distance_pixels=max_distance_pixels,
            raster_type=raster_type, band_index=band_index,
        )
    elif method == "bilinear":
        return interpolate_bilinear(
            array,
            neighborhood=kwargs.get("neighborhood") or "3x3",
            edge_handling=kwargs.get("edge_handling") or "clamp",
            max_distance_pixels=max_distance_pixels,
            band_index=band_index,
        )
    elif method == "bicubic":
        return interpolate_bicubic(
            array,
            neighborhood=kwargs.get("neighborhood") or "4x4",
            edge_handling=kwargs.get("edge_handling") or "clamp",
            max_distance_pixels=max_distance_pixels,
            clip_to_observed_range=(
                True
                if kwargs.get("clip_to_observed_range") is None
                else bool(kwargs["clip_to_observed_range"])
            ),
            band_index=band_index,
        )
    elif method == "local_mean":
        return interpolate_local_mean(
            array,
            window_size=kwargs.get("window_size") or 3,
            min_valid_neighbors=kwargs.get("min_valid_neighbors") or 1,
            neighborhood_shape=kwargs.get("neighborhood_shape") or "square",
            max_distance_pixels=max_distance_pixels,
            band_index=band_index,
        )
    elif method == "local_median":
        return interpolate_local_median(
            array,
            window_size=kwargs.get("window_size") or 3,
            min_valid_neighbors=kwargs.get("min_valid_neighbors") or 1,
            neighborhood_shape=kwargs.get("neighborhood_shape") or "square",
            max_distance_pixels=max_distance_pixels,
            band_index=band_index,
        )
    elif method == "idw":
        return interpolate_idw(
            array,
            power=kwargs.get("power") or 2.0,
            search_radius_pixels=kwargs.get("search_radius_pixels") if kwargs.get("search_radius_pixels") is not None else max_distance_pixels,
            max_neighbors=kwargs.get("max_neighbors") or 12,
            min_neighbors=kwargs.get("min_neighbors") or 1,
            geographic_crs=geographic_crs,
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
            band_index=band_index,
        )
    elif method == "ordinary_kriging":
        return interpolate_ordinary_kriging(
            array,
            variogram_model=kwargs.get("variogram_model") or "spherical",
            variogram_nugget=kwargs.get("variogram_nugget"),
            variogram_sill=kwargs.get("variogram_sill"),
            variogram_range=kwargs.get("variogram_range"),
            search_radius_pixels=kwargs.get("search_radius_pixels") if kwargs.get("search_radius_pixels") is not None else max_distance_pixels,
            max_neighbors=kwargs.get("max_neighbors") or 12,
            min_neighbors=kwargs.get("min_neighbors") or 3,
            band_index=band_index,
        )

    # Temporal methods are not dispatched through run_interpolation (3D cubes)
    if method.startswith("temporal_"):
        raise ValueError(
            f"Temporal method '{method}' must be called via interpolate_temporal() "
            f"with a 3D cube, not run_interpolation()."
        )

    raise ValueError(f"Method '{method}' not implemented.")


def run_temporal_interpolation(
    cube: np.ndarray,
    *,
    method: str,
    times: np.ndarray,
    max_gap_steps: Optional[int] = None,
    allow_extrapolation: bool = False,
    band_index: int = 1,
) -> TemporalInterpolationResult:
    """Dispatch to the selected temporal interpolation algorithm.

    Args:
        cube: 3D array (T, H, W), NaN = missing.
        method: Temporal method name (temporal_linear, temporal_nearest_date, etc.).
        times: 1D array of decimal-year time coordinates (length T).
        max_gap_steps: Max consecutive missing steps to fill.
        allow_extrapolation: If True, fill leading/trailing gaps.
        band_index: 1-based band number.

    Returns:
        TemporalInterpolationResult.
    """
    temporal_methods = {
        "temporal_linear": "linear",
        "temporal_nearest_date": "nearest_date",
        "temporal_forward_fill": "forward_fill",
        "temporal_backward_fill": "backward_fill",
        "temporal_local_mean": "local_temporal_mean",
    }

    if method not in temporal_methods:
        raise ValueError(
            f"Unknown temporal method '{method}'. "
            f"Available: {sorted(temporal_methods.keys())}"
        )

    inner_method = temporal_methods[method]
    return interpolate_temporal(
        cube,
        times=times,
        method=inner_method,
        max_gap_steps=max_gap_steps,
        allow_extrapolation=allow_extrapolation,
        band_index=band_index,
    )
