"""Comprehensive tests for the Raster Interpolation & Gap-Filling Laboratory.

Covers all interpolation algorithms, missing-data patterns, data types,
geospatial properties, scientific correctness, preservation guarantees,
and API contracts.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.raster_interpolation_service import (
    INTERPOLATION_METHODS,
    InterpolationResult,
    interpolate_bicubic,
    interpolate_bilinear,
    interpolate_idw,
    run_interpolation,
    run_temporal_interpolation,
)
from app.services.raster_inspection_service import (
    RasterInspectionResult,
    inspect_raster,
    inspect_raster_from_bytes,
    inspect_raster_from_file,
)
from app.services.interpolation.idw import _haversine_meters
from app.services.interpolation_validation_service import (
    validate_interpolation,
    get_compatible_methods,
)
from app.services.interpolation_provenance_service import (
    build_provenance_mask,
    compute_quality_raster,
    save_provenance_geotiff,
    provenance_mask_to_geotiff,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_array_with_hole(
    shape: tuple[int, int] = (10, 10),
    hole: tuple[slice, slice] = (slice(3, 7), slice(3, 7)),
    fill_value: float = 50.0,
) -> np.ndarray:
    """Create an array with a rectangular hole of NaNs."""
    arr = np.full(shape, fill_value, dtype=float)
    arr[hole] = np.nan
    return arr


def _make_geotiff_bytes(
    array: np.ndarray,
    *,
    crs: str | None = "EPSG:32610",
    transform=None,
    nodata: float | None = None,
    dtype: str | None = None,
) -> bytes:
    array = np.asarray(array)
    if transform is None:
        transform = from_origin(500000.0, 5000000.0, 10.0, 10.0)
    write_dtype = dtype or array.dtype.name
    fd, path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    try:
        with rasterio.open(
            path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
            count=1, dtype=write_dtype, crs=crs, transform=transform, nodata=nodata,
        ) as dst:
            dst.write(array.astype(write_dtype), 1)
        with open(path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Algorithm registry tests
# ---------------------------------------------------------------------------

def test_interpolation_methods_registry():
    """All advertised methods are present in the registry."""
    expected = {
        "nearest_neighbor", "bilinear", "bicubic", "local_mean",
        "local_median", "idw", "ordinary_kriging",
        "temporal_linear", "temporal_nearest_date",
        "temporal_forward_fill", "temporal_backward_fill",
        "temporal_local_mean",
    }
    assert set(INTERPOLATION_METHODS.keys()) == expected


def test_compatible_methods_continuous():
    """Continuous rasters get spatial methods, not temporal."""
    methods = [m for m in get_compatible_methods("continuous") if m["compatible"]]
    ids = {m["id"] for m in methods}
    assert "nearest_neighbor" in ids
    assert "bilinear" in ids
    assert "ordinary_kriging" in ids
    assert "temporal_linear" not in ids


def test_compatible_methods_categorical():
    """Categorical rasters only get nearest neighbor."""
    methods = [m for m in get_compatible_methods("categorical") if m["compatible"]]
    ids = {m["id"] for m in methods}
    assert ids == {"nearest_neighbor"}


def test_compatible_methods_temporal():
    """Temporal rasters get temporal methods."""
    methods = [m for m in get_compatible_methods("temporal") if m["compatible"]]
    ids = {m["id"] for m in methods}
    assert "temporal_linear" in ids
    assert "temporal_forward_fill" in ids
    assert "nearest_neighbor" not in ids


def test_method_incompatibility_raises():
    """Incompatible method + raster_type raises ValueError."""
    arr = _make_array_with_hole()
    with pytest.raises(ValueError, match="not compatible"):
        run_interpolation(arr, method="bilinear", raster_type="categorical")


def test_unknown_method_raises():
    """Unknown method raises ValueError."""
    arr = _make_array_with_hole()
    with pytest.raises(ValueError, match="Unknown interpolation method"):
        run_interpolation(arr, method="nonexistent_method", raster_type="continuous")


# ---------------------------------------------------------------------------
# Nearest neighbor tests
# ---------------------------------------------------------------------------

def test_nearest_neighbor_fills_hole():
    arr = _make_array_with_hole()
    r: InterpolationResult = run_interpolation(arr, method="nearest_neighbor", raster_type="continuous")
    assert r.interpolated_pixel_count == 16
    assert r.remaining_nodata_count == 0
    # All interpolated pixels should be 50.0 (neighbor value)
    for i in range(3, 7):
        for j in range(3, 7):
            assert r.reconstructed[i, j] == 50.0


def test_nearest_neighbor_preserves_valid():
    arr = _make_array_with_hole()
    arr[0, 0] = 99.0
    r = run_interpolation(arr, method="nearest_neighbor", raster_type="continuous")
    assert r.reconstructed[0, 0] == 99.0
    assert r.observed_pixel_count == 84


def test_nearest_neighbor_categorical():
    arr = np.full((10, 10), 1.0, dtype=float)
    arr[3:7, 3:7] = np.nan
    r = run_interpolation(arr, method="nearest_neighbor", raster_type="categorical")
    assert r.interpolated_pixel_count == 16
    assert r.raster_type == "categorical"


def test_nearest_neighbor_all_missing_excluded():
    arr = np.full((5, 5), np.nan, dtype=float)
    r = run_interpolation(arr, method="nearest_neighbor", raster_type="continuous")
    assert r.interpolated_pixel_count == 0
    assert r.excluded_pixel_count == 25
    assert r.remaining_nodata_count == 25


def test_nearest_neighbor_no_missing():
    arr = np.full((5, 5), 50.0, dtype=float)
    r = run_interpolation(arr, method="nearest_neighbor", raster_type="continuous")
    assert r.interpolated_pixel_count == 0
    assert r.remaining_nodata_count == 0


def test_nearest_neighbor_max_distance_excludes():
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="nearest_neighbor", raster_type="continuous", max_distance_pixels=0.5)
    assert r.interpolated_pixel_count == 0
    assert r.excluded_pixel_count == 16


# ---------------------------------------------------------------------------
# Bilinear tests
# ---------------------------------------------------------------------------

def test_bilinear_fills_hole_with_large_neighborhood():
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="bilinear", raster_type="continuous", neighborhood="5x5")
    assert r.interpolated_pixel_count == 16
    assert r.remaining_nodata_count == 0


def test_bilinear_small_neighborhood_leaves_corners():
    """3x3 neighborhood can't reach 4x4 hole corners."""
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="bilinear", raster_type="continuous", neighborhood="3x3")
    assert r.interpolated_pixel_count < 16  # corners left as NoData
    assert r.excluded_pixel_count > 0


def test_bilinear_continuous_only():
    arr = _make_array_with_hole()
    with pytest.raises(ValueError, match="not compatible"):
        run_interpolation(arr, method="bilinear", raster_type="categorical")


# ---------------------------------------------------------------------------
# Bicubic tests
# ---------------------------------------------------------------------------

def test_bicubic_fills_hole():
    """Bicubic fills a hole that is surrounded by valid observations.

    Compact-support bicubic interpolates from the immediate neighbourhood, so
    a hole whose cells all have valid neighbours is filled completely.
    """
    arr = _make_array_with_hole(shape=(6, 6), hole=(slice(2, 4), slice(2, 4)))
    r = run_interpolation(arr, method="bicubic", raster_type="continuous")
    assert r.interpolated_pixel_count == 4
    assert r.remaining_nodata_count == 0


def test_bicubic_4x4_not_5x5():
    """A requested 4x4 neighbourhood must be a genuine 4x4 grid, not 5x5.

    With a 4x4 stencil the four centre cells of a 4x4 hole have no valid
    neighbour inside the stencil, so they are correctly excluded rather than
    being filled by a silently-expanded 5x5 window.
    """
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="bicubic", raster_type="continuous", neighborhood="4x4")
    # Edge cells of the hole are reached by the 4x4 stencil; the four centre
    # cells ((4,4),(4,5),(5,4),(5,5)) are not.
    assert r.interpolated_pixel_count == 12
    assert r.excluded_pixel_count == 4
    for i, j in ((4, 4), (4, 5), (5, 4), (5, 5)):
        assert r.provenance_mask[i, j] == "excluded", (i, j)


def test_bicubic_clips_to_range():
    """Bicubic with clipping should not exceed local observed range."""
    arr = np.full((10, 10), 10.0, dtype=float)
    arr[0, 0] = 100.0  # outlier
    arr[4:6, 4:6] = np.nan
    r = run_interpolation(arr, method="bicubic", raster_type="continuous", clip_to_observed_range=True)
    # All interpolated values should be within [10, 100]
    for i in range(4, 6):
        for j in range(4, 6):
            v = r.reconstructed[i, j]
            assert 10.0 <= v <= 100.0, f"Value {v} outside range at ({i},{j})"


# ---------------------------------------------------------------------------
# Local mean tests
# ---------------------------------------------------------------------------

def test_local_mean_fills_small_hole():
    arr = _make_array_with_hole(shape=(6, 6), hole=(slice(2, 4), slice(2, 4)))
    r = run_interpolation(arr, method="local_mean", raster_type="continuous", window_size=3)
    assert r.interpolated_pixel_count == 4
    assert r.remaining_nodata_count == 0


def test_local_mean_min_neighbors_excludes():
    arr = _make_array_with_hole()
    # Window 3x3 around hole edge has neighbors, but center of 4x4 hole
    # with min_valid_neighbors=9 should exclude center pixels
    r = run_interpolation(arr, method="local_mean", raster_type="continuous", window_size=3, min_valid_neighbors=9)
    assert r.excluded_pixel_count > 0


# ---------------------------------------------------------------------------
# Local median tests
# ---------------------------------------------------------------------------

def test_local_median_fills_hole():
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="local_median", raster_type="continuous", window_size=5)
    assert r.interpolated_pixel_count == 16
    assert r.remaining_nodata_count == 0


# ---------------------------------------------------------------------------
# IDW tests
# ---------------------------------------------------------------------------

def test_idw_fills_hole():
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="idw", raster_type="continuous")
    assert r.interpolated_pixel_count == 16
    assert r.remaining_nodata_count == 0


def test_idw_power_parameter():
    arr = _make_array_with_hole()
    r1 = run_interpolation(arr, method="idw", raster_type="continuous", power=1.0)
    r2 = run_interpolation(arr, method="idw", raster_type="continuous", power=10.0)
    # Different powers should give different results (unless all neighbors equidistant)
    assert r1.interpolated_pixel_count == r2.interpolated_pixel_count == 16


def test_idw_min_neighbors_excludes():
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="idw", raster_type="continuous", min_neighbors=50)
    assert r.excluded_pixel_count == 16


# ---------------------------------------------------------------------------
# Ordinary kriging tests
# ---------------------------------------------------------------------------

def test_kriging_fills_hole():
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="ordinary_kriging", raster_type="continuous")
    assert r.interpolated_pixel_count == 16
    assert r.remaining_nodata_count == 0
    assert r.quality_mask is not None  # kriging variance


def test_kriging_variance_non_negative():
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="ordinary_kriging", raster_type="continuous")
    kv = r.quality_mask
    for i in range(3, 7):
        for j in range(3, 7):
            v = kv[i, j]
            if v is not None:
                assert v >= 0.0, f"Negative kriging variance at ({i},{j}): {v}"


def test_kriging_insufficient_observations():
    arr = np.full((3, 3), np.nan, dtype=float)
    arr[0, 0] = 50.0  # only 1 observation, need min_neighbors=3
    r = run_interpolation(arr, method="ordinary_kriging", raster_type="continuous")
    assert r.interpolated_pixel_count == 0


def test_kriging_variogram_models():
    arr = _make_array_with_hole()
    for model in ["spherical", "exponential", "gaussian", "linear"]:
        r = run_interpolation(arr, method="ordinary_kriging", raster_type="continuous", variogram_model=model)
        assert r.interpolated_pixel_count == 16, f"Model {model} failed"


# ---------------------------------------------------------------------------
# Temporal interpolation tests
# ---------------------------------------------------------------------------

def test_temporal_linear_interpolates_middle():
    cube = np.full((5, 10, 10), 50.0, dtype=float)
    cube[2, 3:7, 3:7] = np.nan  # missing at middle time step
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    r = run_temporal_interpolation(cube, method="temporal_linear", times=times)
    assert r.interpolated_count == 16
    assert r.remaining_nodata_count == 0
    # Value at t=2 should be average of t=1 and t=3 (both 50)
    for i in range(3, 7):
        for j in range(3, 7):
            assert r.reconstructed_cube[2, i, j] == 50.0


def test_temporal_forward_fill():
    cube = np.full((5, 10, 10), 50.0, dtype=float)
    cube[1, 3:7, 3:7] = np.nan  # missing at time step 1 (not leading)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # Forward fill uses t=0 value
    r = run_temporal_interpolation(cube, method="temporal_forward_fill", times=times, allow_extrapolation=True)
    assert r.interpolated_count == 16
    for i in range(3, 7):
        for j in range(3, 7):
            assert r.reconstructed_cube[1, i, j] == 50.0


def test_temporal_backward_fill():
    cube = np.full((5, 10, 10), 50.0, dtype=float)
    cube[3, 3:7, 3:7] = np.nan  # missing at time step 3 (not trailing)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    r = run_temporal_interpolation(cube, method="temporal_backward_fill", times=times, allow_extrapolation=True)
    assert r.interpolated_count == 16


def test_temporal_max_gap():
    cube = np.full((10, 5, 5), 50.0, dtype=float)
    cube[2:6, :, :] = np.nan  # 4 consecutive missing steps
    times = np.arange(10, dtype=float)
    # Max gap of 2 should not fill the 4-step gap
    r = run_temporal_interpolation(cube, method="temporal_linear", times=times, max_gap_steps=2)
    assert r.interpolated_count == 0
    # Max gap of 5 should fill it
    r = run_temporal_interpolation(cube, method="temporal_linear", times=times, max_gap_steps=5)
    assert r.interpolated_count > 0


def test_temporal_requires_times():
    cube = np.full((5, 5, 5), 50.0, dtype=float)
    with pytest.raises(ValueError, match=r"times.*required"):
        run_temporal_interpolation(cube, method="temporal_linear", times=None)


# ---------------------------------------------------------------------------
# Missing-data / preservation tests
# ---------------------------------------------------------------------------

def test_original_valid_pixels_unchanged():
    arr = np.full((10, 10), 50.0, dtype=float)
    arr[0, 0] = 99.0
    arr[9, 9] = 1.0
    arr[3:7, 3:7] = np.nan
    r = run_interpolation(arr, method="idw", raster_type="continuous")
    assert r.reconstructed[0, 0] == 99.0
    assert r.reconstructed[9, 9] == 1.0
    assert r.observed_pixel_count == 84


def test_provenance_mask_classes():
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="nearest_neighbor", raster_type="continuous")
    prov = r.provenance_mask
    # Valid pixels should be 'observed'
    assert prov[0, 0] == "observed"
    # Interpolated pixels should be 'interpolated'
    assert prov[4, 4] == "interpolated"
    # Count check
    observed = int(np.sum(prov == "observed"))
    interpolated = int(np.sum(prov == "interpolated"))
    assert observed == 84
    assert interpolated == 16


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_validation_returns_metrics():
    arr = np.full((30, 30), 50.0, dtype=float)
    arr[10:20, 10:20] = np.nan
    r = validate_interpolation(
        array=arr, method="nearest_neighbor", raster_type="continuous",
        validation_strategy="random_holdout", sample_fraction=0.1, seed=42,
    )
    assert r.validation_sample_count > 0
    if r.mae is not None:
        assert r.mae >= 0.0
    if r.rmse is not None:
        assert r.rmse >= 0.0


def test_validation_spatial_block():
    arr = np.full((30, 30), 50.0, dtype=float)
    arr[10:20, 10:20] = np.nan
    r = validate_interpolation(
        array=arr, method="idw", raster_type="continuous",
        validation_strategy="spatial_block", n_blocks=4, block_index=0, seed=42,
    )
    assert r.validation_strategy == "spatial_block"


def test_validation_insufficient_data():
    arr = np.full((3, 3), np.nan, dtype=float)  # no valid pixels
    r = validate_interpolation(array=arr, method="nearest_neighbor", raster_type="continuous")
    assert r.validation_sample_count == 0


# ---------------------------------------------------------------------------
# Raster inspection tests
# ---------------------------------------------------------------------------

def test_inspection_basic():
    arr = _make_array_with_hole()
    result: RasterInspectionResult = inspect_raster(array=arr, band_index=1)
    assert result.width == 10
    assert result.height == 10
    assert result.valid_pixel_count == 84
    assert result.missing_pixel_count == 16
    assert result.missing_percentage == 16.0
    assert result.valid_percentage == 84.0
    assert result.raster_type == "continuous"


def test_inspection_categorical():
    arr = np.full((10, 10), 1, dtype=float)
    arr[3:7, 3:7] = np.nan
    result = inspect_raster(array=arr, band_index=1, user_confirmed_type="categorical")
    assert result.raster_type == "categorical"


def test_inspection_all_missing():
    arr = np.full((5, 5), np.nan, dtype=float)
    result = inspect_raster(array=arr, band_index=1)
    assert result.missing_pixel_count == 25
    assert result.missing_percentage == 100.0
    assert result.valid_pixel_count == 0


def test_inspection_no_missing():
    arr = np.full((5, 5), 50.0, dtype=float)
    result = inspect_raster(array=arr, band_index=1)
    assert result.missing_pixel_count == 0
    assert result.missing_percentage == 0.0


def test_inspection_from_file():
    arr = _make_array_with_hole()
    b = _make_geotiff_bytes(arr, nodata=-9999.0)
    result = inspect_raster_from_bytes(raw_bytes=b, filename="test.tif", band_index=1)
    assert result.width == 10
    assert result.height == 10
    assert result.crs == "EPSG:32610"


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------

def test_build_provenance_mask():
    orig = _make_array_with_hole()
    recon = orig.copy()
    recon[np.isnan(orig)] = 50.0
    interp_mask = np.isnan(orig)
    pm = build_provenance_mask(orig, recon, interp_mask)
    assert pm.counts["observed"] == 84
    assert pm.counts["interpolated"] == 16


def test_compute_quality_raster():
    prov = np.full((5, 5), "observed", dtype=object)
    prov[2, 2] = "interpolated"
    q = compute_quality_raster(prov)
    assert q[0, 0] == 1.0  # observed = highest quality
    assert not np.isnan(q[2, 2])  # interpolated has quality score
    assert 0.0 < q[2, 2] <= 1.0


def test_save_provenance_geotiff(tmp_path):
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="nearest_neighbor", raster_type="continuous")
    out = tmp_path / "out.tif"
    save_provenance_geotiff(
        output_path=str(out),
        reconstructed_array=r.reconstructed,
        provenance_mask=r.provenance_mask,
        quality_raster=r.quality_mask,
        crs="EPSG:32610",
        transform=from_origin(500000.0, 5000000.0, 10.0, 10.0),
        input_metadata={},
        processing_metadata={"method": "nearest_neighbor"},
    )
    assert out.exists()
    with rasterio.open(out) as src:
        assert src.count == 3  # reconstructed + provenance + quality
        assert src.crs.to_epsg() == 32610


# ---------------------------------------------------------------------------
# API-level tests
# ---------------------------------------------------------------------------

def test_api_contract_preserved():
    """Ensure InterpolationResult has all expected fields."""
    arr = _make_array_with_hole()
    r = run_interpolation(arr, method="nearest_neighbor", raster_type="continuous")
    assert hasattr(r, "method")
    assert hasattr(r, "band_index")
    assert hasattr(r, "raster_type")
    assert hasattr(r, "config")
    assert hasattr(r, "observed_pixel_count")
    assert hasattr(r, "interpolated_pixel_count")
    assert hasattr(r, "remaining_nodata_count")
    assert hasattr(r, "excluded_pixel_count")
    assert hasattr(r, "processing_duration_ms")
    assert hasattr(r, "provenance_mask")
    assert hasattr(r, "reconstructed")


# ---------------------------------------------------------------------------
# Mathematical unit tests (analytically known results)
# ---------------------------------------------------------------------------

def test_bilinear_reproduces_plane_exactly():
    """Bilinear at the centre of four diagonal neighbours reproduces a plane.

    z = 3*row + col.  The missing cell (1, 1) is surrounded by the four
    diagonal neighbours (0,0)=0, (0,2)=2, (2,0)=6, (2,2)=8 whose bilinear
    centre weights (0.25 each) give exactly z(1,1) = 4.
    """
    arr = np.array([[0.0, 1.0, 2.0],
                    [3.0, np.nan, 5.0],
                    [6.0, 7.0, 8.0]])
    r = interpolate_bilinear(arr)
    assert r.provenance_mask[1, 1] == "interpolated"
    assert abs(r.reconstructed[1, 1] - 4.0) < 1e-9


def test_bicubic_reproduces_plane_exactly():
    """Bicubic (cubic B-spline) reproduces a plane exactly.

    The cubic B-spline reproduces polynomials up to degree three, so the
    estimate at the missing cell (1, 1) equals z(1, 1) = 4.
    """
    arr = np.array([[0.0, 1.0, 2.0],
                    [3.0, np.nan, 5.0],
                    [6.0, 7.0, 8.0]])
    r = interpolate_bicubic(arr)
    assert r.provenance_mask[1, 1] == "interpolated"
    assert abs(r.reconstructed[1, 1] - 4.0) < 1e-9


def test_clip_to_observed_range_false_is_respected():
    """An explicit clip_to_observed_range=False must not be overridden to True."""
    arr = np.array([[0.0, 1.0, 2.0],
                    [3.0, np.nan, 5.0],
                    [6.0, 7.0, 8.0]])
    r_false = interpolate_bicubic(arr, clip_to_observed_range=False)
    assert r_false.config["clip_to_observed_range"] is False
    r_true = interpolate_bicubic(arr)  # default True
    assert r_true.config["clip_to_observed_range"] is True


def test_edge_handling_policies():
    """clamp / nodata / reject behave differently at a corner missing cell."""
    arr = np.array([[np.nan, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0]])
    # reject: a diagonal neighbour is out of bounds -> excluded
    r = interpolate_bilinear(arr, edge_handling="reject")
    assert r.provenance_mask[0, 0] == "excluded"
    # nodata: out of bounds skipped; only diagonal (1, 1) = 4 is valid
    r = interpolate_bilinear(arr, edge_handling="nodata")
    assert abs(r.reconstructed[0, 0] - 4.0) < 1e-9
    # clamp: (-1,1)->(0,1)=1, (1,-1)->(1,0)=3, (1,1)=4 valid -> mean 8/3
    r = interpolate_bilinear(arr, edge_handling="clamp")
    assert abs(r.reconstructed[0, 0] - 8.0 / 3.0) < 1e-9


def test_haversine_is_geodesic():
    """Haversine distance is in metres and respects the latitude scale."""
    # One degree of latitude is ~111320 m everywhere.
    assert abs(_haversine_meters(0.0, 0.0, 1.0, 0.0) - 111320.0) < 600.0
    # One degree of longitude at the equator is ~111320 m.
    assert abs(_haversine_meters(0.0, 0.0, 0.0, 1.0) - 111320.0) < 600.0
    # One degree of longitude at 60 deg latitude is about half (~cos 60 = 0.5).
    d_equator = _haversine_meters(0.0, 0.0, 0.0, 1.0)
    d_60 = _haversine_meters(60.0, 0.0, 60.0, 1.0)
    assert d_60 < d_equator * 0.6


def test_idw_geographic_uses_metres():
    """Geographic IDW converts degrees to metres via haversine."""
    arr = np.array([[10.0, np.nan, 20.0],
                    [np.nan, np.nan, np.nan],
                    [30.0, np.nan, 40.0]])
    # Nearest neighbour is ~111 km away (1 deg at the equator).
    # A 50 km search radius excludes every neighbour.
    r = interpolate_idw(arr, geographic_crs=True,
                        pixel_size_x=1.0, pixel_size_y=1.0,
                        search_radius_pixels=50000)
    assert r.provenance_mask[1, 1] == "excluded"
    # A 200 km search radius reaches the neighbours -> interpolated.
    r = interpolate_idw(arr, geographic_crs=True,
                        pixel_size_x=1.0, pixel_size_y=1.0,
                        search_radius_pixels=200000)
    assert r.provenance_mask[1, 1] == "interpolated"
    # Without pixel sizes, geographic falls back to pixel-Euclidean (no crash).
    r = interpolate_idw(arr, geographic_crs=True)
    assert r.provenance_mask[1, 1] == "interpolated"


def test_idw_projected_uses_pixel_coordinates():
    """Projected IDW uses pixel-Euclidean distance (no degree/radian mix)."""
    arr = np.array([[np.nan, 1.0, 2.0],
                    [3.0, np.nan, 5.0],
                    [6.0, 7.0, 8.0]])
    r = interpolate_idw(arr, geographic_crs=False)
    assert r.provenance_mask[0, 0] == "interpolated"
    assert np.isfinite(r.reconstructed[0, 0])
    # Anisotropic pixel sizes are honoured (run without error, finite result).
    r = interpolate_idw(arr, geographic_crs=False,
                        pixel_size_x=10.0, pixel_size_y=1.0)
    assert np.isfinite(r.reconstructed[0, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
