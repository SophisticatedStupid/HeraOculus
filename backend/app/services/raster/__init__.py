"""SAR / DEM Landform Change Detection, LULC Transition Matrix & NASA ASF Proxy.

This package splits the former monolithic ``raster_service`` module into
domain-focused sub-modules:

- ``base.py``   shared raster I/O, alignment, pixel-area and rendering helpers
- ``dem.py``    DEM of Difference (DoD) change detection
- ``sar.py``    SAR speckle filtering, log-ratio change detection & ASF proxy
- ``lulc.py``   LULC transition matrix & Cohen's Kappa

All public (and private, test-referenced) names are re-exported here so
existing imports such as ``from app.services import raster_service as rs``
and ``from ..services.raster_service import ...`` keep working unchanged.
No behavior has changed.
"""

from .base import (
    _align_second_raster,
    _as_pyproj_crs,
    _compute_pixel_areas,
    _crs_equivalent,
    _determine_area_kind,
    _encode_png,
    _get_crs_linear_units_to_meters,
    _pixel_area_metadata,
    _pixel_grid_resize,
    _read_raster_bytes,
    _read_raster_file,
    _reproject_to_grid,
    _render_scalar_png,
    _validate_alignment,
)
from .dem import (
    _synthetic_dem_pair,
    compute_dem_differencing,
    compute_dem_differencing_from_files,
    compute_dem_differencing_from_paths,
)
from .lulc import (
    LULC_CLASS_LABELS,
    LULC_PALETTE,
    NODATA_COLOR,
    _render_categorical_png,
    _synthetic_lulc_pair,
    compute_lulc_from_files,
    compute_lulc_from_paths,
    compute_lulc_transitions,
)
from .sar import (
    _lee_speckle_filter,
    _mean_speckle_filter,
    _sar_result_all_invalid,
    _synthetic_sar_pair,
    compute_sar_change_detection,
    compute_sar_change_from_files,
    compute_sar_change_from_paths,
    search_asf_sar_granules,
)

__all__ = [
    # base
    "_align_second_raster",
    "_as_pyproj_crs",
    "_compute_pixel_areas",
    "_crs_equivalent",
    "_determine_area_kind",
    "_encode_png",
    "_get_crs_linear_units_to_meters",
    "_pixel_area_metadata",
    "_pixel_grid_resize",
    "_read_raster_bytes",
    "_read_raster_file",
    "_reproject_to_grid",
    "_render_scalar_png",
    "_validate_alignment",
    # dem
    "_synthetic_dem_pair",
    "compute_dem_differencing",
    "compute_dem_differencing_from_files",
    "compute_dem_differencing_from_paths",
    # lulc
    "LULC_CLASS_LABELS",
    "LULC_PALETTE",
    "NODATA_COLOR",
    "_render_categorical_png",
    "_synthetic_lulc_pair",
    "compute_lulc_from_files",
    "compute_lulc_from_paths",
    "compute_lulc_transitions",
    # sar
    "_lee_speckle_filter",
    "_mean_speckle_filter",
    "_sar_result_all_invalid",
    "_synthetic_sar_pair",
    "compute_sar_change_detection",
    "compute_sar_change_from_files",
    "compute_sar_change_from_paths",
    "search_asf_sar_granules",
]
