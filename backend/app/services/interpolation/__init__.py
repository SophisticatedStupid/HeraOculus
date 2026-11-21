"""Raster interpolation engine.

Implements spatially-aware gap-filling algorithms:
- Nearest neighbor (continuous & categorical)
- Bilinear interpolation (continuous)
- Bicubic interpolation (continuous)
- Local mean interpolation (continuous)
- Local median interpolation (continuous)
- Inverse Distance Weighting (IDW) (continuous)
- Ordinary Kriging (continuous)
- Temporal interpolation (time-series cubes)

All algorithms preserve original valid pixels, respect NoData masks,
and never modify the source array.

This package splits the former monolithic ``raster_interpolation_service``
module into one module per interpolation method family plus a shared
base/types module.  All public names remain importable from here.
"""

from .base import (
    InterpolationResult,
    TemporalInterpolationResult,
    _ensure_float_array,
    _valid_mask,
    _nearest_valid_indices,
)
from .focal import (
    interpolate_bilinear,
    interpolate_bicubic,
    interpolate_local_mean,
    interpolate_local_median,
)
from .idw import interpolate_idw
from .kriging import (
    _empirical_variogram,
    _fit_variogram,
    _ordinary_kriging_system,
    _variogram_model_exponential,
    _variogram_model_gaussian,
    _variogram_model_linear,
    _variogram_model_spherical,
    _VARIOGRAM_MODELS,
    interpolate_ordinary_kriging,
)
from .nearest import interpolate_nearest_neighbor
from .registry import INTERPOLATION_METHODS, run_interpolation, run_temporal_interpolation
from .temporal import _temporal_interpolate_1d, interpolate_temporal

__all__ = [
    "InterpolationResult",
    "TemporalInterpolationResult",
    "_ensure_float_array",
    "_valid_mask",
    "_nearest_valid_indices",
    "interpolate_nearest_neighbor",
    "interpolate_bilinear",
    "interpolate_bicubic",
    "interpolate_local_mean",
    "interpolate_local_median",
    "interpolate_idw",
    "interpolate_ordinary_kriging",
    "_empirical_variogram",
    "_fit_variogram",
    "_ordinary_kriging_system",
    "_variogram_model_spherical",
    "_variogram_model_exponential",
    "_variogram_model_gaussian",
    "_variogram_model_linear",
    "_VARIOGRAM_MODELS",
    "interpolate_temporal",
    "_temporal_interpolate_1d",
    "INTERPOLATION_METHODS",
    "run_interpolation",
    "run_temporal_interpolation",
]
