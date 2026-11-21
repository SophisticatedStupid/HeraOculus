"""Raster interpolation engine (backward-compatible re-export shim).

This module has been split into smaller modules under
``backend/app/services/interpolation/``:

- ``base.py``            shared types (``InterpolationResult``,
  ``TemporalInterpolationResult``) and low-level helpers
- ``nearest.py``         nearest neighbor interpolation
- ``focal.py``           bilinear, bicubic, local mean, local median
- ``idw.py``             inverse distance weighting
- ``kriging.py``         ordinary kriging + variogram helpers
- ``temporal.py``        temporal interpolation for 3D cubes
- ``registry.py``        method registry and ``run_*`` dispatchers

All public names are re-exported here so existing imports such as
``from app.services.raster_interpolation_service import ...`` keep working
unchanged.  No behavior has changed.
"""

from .interpolation import (  # noqa: F401
    INTERPOLATION_METHODS,
    InterpolationResult,
    TemporalInterpolationResult,
    _empirical_variogram,
    _ensure_float_array,
    _fit_variogram,
    _nearest_valid_indices,
    _ordinary_kriging_system,
    _temporal_interpolate_1d,
    _valid_mask,
    _VARIOGRAM_MODELS,
    interpolate_bicubic,
    interpolate_bilinear,
    interpolate_idw,
    interpolate_local_mean,
    interpolate_local_median,
    interpolate_nearest_neighbor,
    interpolate_ordinary_kriging,
    interpolate_temporal,
    run_interpolation,
    run_temporal_interpolation,
)

__all__ = [
    "INTERPOLATION_METHODS",
    "InterpolationResult",
    "TemporalInterpolationResult",
    "_empirical_variogram",
    "_ensure_float_array",
    "_fit_variogram",
    "_nearest_valid_indices",
    "_ordinary_kriging_system",
    "_temporal_interpolate_1d",
    "_valid_mask",
    "_VARIOGRAM_MODELS",
    "interpolate_bicubic",
    "interpolate_bilinear",
    "interpolate_idw",
    "interpolate_local_mean",
    "interpolate_local_median",
    "interpolate_nearest_neighbor",
    "interpolate_ordinary_kriging",
    "interpolate_temporal",
    "run_interpolation",
    "run_temporal_interpolation",
]
