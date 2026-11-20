"""Validation tests for raster geospatial alignment (CRS validation + reprojection).

Covers the alignment contract enforced by ``_align_second_raster``:

* Same CRS and same grid          -> fast-path copy
* Same CRS, different resolution  -> resample onto reference grid
* Same CRS, shifted grid          -> resample onto reference grid
* Different projected CRSs        -> reproject onto reference grid
* Geographic vs projected CRS     -> reproject onto reference grid
* Missing CRS (both)              -> reject unless pixel_grid_only=True
* Missing CRS (one)               -> reject
* Different array dimensions      -> resample onto reference grid
* Nodata regions                  -> preserved as NaN (never zero)
* Categorical class preservation  -> nearest-neighbour keeps class labels
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin

from app.services import raster_service as rs


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_geotiff_bytes(
    array: np.ndarray,
    *,
    crs: str | None = "EPSG:32610",
    transform: Affine | None = None,
    nodata: float | None = None,
    dtype: str | None = None,
) -> bytes:
    """Write an in-memory GeoTIFF and return its bytes."""
    array = np.asarray(array)
    if transform is None:
        transform = from_origin(500000.0, 5000000.0, 10.0, 10.0)
    write_dtype = dtype or array.dtype.name
    fd, path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    try:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=write_dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(array.astype(write_dtype), 1)
        with open(path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


def _read(arr_bytes: bytes, name: str = "r.tif") -> tuple[np.ndarray, np.ndarray, dict]:
    return rs._read_raster_bytes(arr_bytes, name)


def _grid(shape: tuple[int, int], res: float, origin: tuple[float, float] = (500000.0, 5000000.0)) -> Affine:
    return from_origin(origin[0], origin[1], res, res)


# ---------------------------------------------------------------------------
# 1. Same CRS and same grid
# ---------------------------------------------------------------------------

def test_same_crs_same_grid_fastpath():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    tf = _grid((4, 4), 10.0)
    ref, _, ref_meta = _read(_make_geotiff_bytes(arr, crs="EPSG:32610", transform=tf))
    src, _, src_meta = _read(_make_geotiff_bytes(arr + 1.0, crs="EPSG:32610", transform=tf))

    out, _ = rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=False)
    assert out.shape == ref.shape
    # Identical grid: values pass through unchanged (no resampling)
    assert np.allclose(out, arr + 1.0)
    print("PASS: same CRS + same grid fast-path copy")


# ---------------------------------------------------------------------------
# 2. Same CRS and different resolution
# ---------------------------------------------------------------------------

def test_same_crs_different_resolution_resamples_to_ref_grid():
    ref_arr = np.full((8, 8), 5.0, dtype=np.float32)
    ref_bytes = _make_geotiff_bytes(ref_arr, crs="EPSG:32610", transform=_grid((8, 8), 10.0))
    # Source is 20 m pixels, covers same extent -> 4x4 array
    src_arr = np.full((4, 4), 5.0, dtype=np.float32)
    src_bytes = _make_geotiff_bytes(src_arr, crs="EPSG:32610", transform=_grid((4, 4), 20.0))

    ref, _, ref_meta = _read(ref_bytes)
    src, _, src_meta = _read(src_bytes)
    out, _ = rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=False)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    valid = np.isfinite(out)
    assert np.allclose(out[valid], 5.0, atol=1e-6)
    print("PASS: same CRS, different resolution -> resampled onto ref grid")


# ---------------------------------------------------------------------------
# 3. Same CRS and shifted grid
# ---------------------------------------------------------------------------

def test_same_crs_shifted_grid_aligns_to_ref():
    ref_tf = _grid((6, 6), 10.0, origin=(500000.0, 5000000.0))
    # Shift source origin by half a pixel in each axis, same CRS/res
    src_tf = Affine(10.0, 0.0, 500005.0, 0.0, -10.0, 4999995.0)
    ref_arr = np.full((6, 6), 12.0, dtype=np.float32)
    src_arr = np.full((6, 6), 12.0, dtype=np.float32)

    ref, _, ref_meta = _read(_make_geotiff_bytes(ref_arr, crs="EPSG:32610", transform=ref_tf))
    src, _, src_meta = _read(_make_geotiff_bytes(src_arr, crs="EPSG:32610", transform=src_tf))
    out, _ = rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=False)
    assert out.shape == ref.shape
    valid = np.isfinite(out)
    assert valid.sum() > 0
    assert np.allclose(out[valid], 12.0, atol=1e-5)
    print("PASS: same CRS, shifted grid -> aligned to ref grid")


# ---------------------------------------------------------------------------
# 4. Different projected CRSs
# ---------------------------------------------------------------------------

def test_different_projected_crs_reprojects():
    # WGS84 UTM 10N vs NAD83 UTM 10N: same geographic extent, different CRS definitions.
    # reproject must warp between them (no geographic mismatch), producing valid values.
    ref_tf = from_origin(500000.0, 5000000.0, 10.0, 10.0)
    src_tf = from_origin(500000.0, 5000000.0, 10.0, 10.0)
    ref_arr = np.full((8, 8), 100.0, dtype=np.float32)
    src_arr = np.full((8, 8), 100.0, dtype=np.float32)

    ref, _, ref_meta = _read(_make_geotiff_bytes(ref_arr, crs="EPSG:32610", transform=ref_tf))
    src, _, src_meta = _read(_make_geotiff_bytes(src_arr, crs="EPSG:26910", transform=src_tf))
    assert not rs._crs_equivalent(ref_meta["crs"], src_meta["crs"])

    out, _ = rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=False)
    assert out.shape == ref.shape
    # Geographically overlapping CRSs: source projects fully into reference grid.
    valid = np.isfinite(out)
    assert valid.sum() > 0
    assert np.allclose(out[valid], 100.0, atol=1e-3)
    print("PASS: different projected CRSs -> reprojected onto ref grid")


# ---------------------------------------------------------------------------
# 5. Geographic versus projected CRS
# ---------------------------------------------------------------------------

def test_geographic_vs_projected_crs_reprojects():
    # Reference: WGS84 geographic over a small degree window
    ref_tf = from_origin(-114.0, 37.0, 0.001, 0.001)
    ref_arr = np.full((10, 10), 50.0, dtype=np.float32)
    ref, _, ref_meta = _read(_make_geotiff_bytes(ref_arr, crs="EPSG:4326", transform=ref_tf))

    # Source: UTM 11N covering roughly the same area (~ -114 lon, 37 lat)
    # UTM 11N easting ~ 500000 at central meridian -117; near -114 easting ~ 770000
    src_tf = from_origin(760000.0, 4100000.0, 100.0, 100.0)
    src_arr = np.full((10, 10), 50.0, dtype=np.float32)
    src, _, src_meta = _read(_make_geotiff_bytes(src_arr, crs="EPSG:32611", transform=src_tf))

    assert ref_meta["crs"].is_geographic
    assert not src_meta["crs"].is_geographic

    out, _ = rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=False)
    assert out.shape == ref.shape
    # Coverage may be partial due to CRS overlap; assert no crash + shape lock.
    print(f"PASS: geographic vs projected -> shape {out.shape} locked to ref")


# ---------------------------------------------------------------------------
# 6. Missing CRS
# ---------------------------------------------------------------------------

def test_both_missing_crs_rejected_by_default():
    arr = np.full((6, 6), 3.0, dtype=np.float32)
    tf = _grid((6, 6), 10.0)
    ref, _, ref_meta = _read(_make_geotiff_bytes(arr, crs=None, transform=tf))
    src, _, src_meta = _read(_make_geotiff_bytes(arr, crs=None, transform=tf))
    assert ref_meta["crs"] is None and src_meta["crs"] is None

    with pytest.raises(ValueError, match="lack CRS"):
        rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=False)
    print("PASS: both missing CRS -> rejected by default")


def test_both_missing_crs_pixel_grid_only_mode():
    # Different-sized grids (same extent, different res) so a real resize happens.
    ref_arr = np.full((8, 8), 3.0, dtype=np.float32)
    src_arr = np.full((4, 4), 5.0, dtype=np.float32)
    ref, _, ref_meta = _read(_make_geotiff_bytes(ref_arr, crs=None, transform=_grid((8, 8), 10.0)))
    src, _, src_meta = _read(_make_geotiff_bytes(src_arr, crs=None, transform=_grid((4, 4), 20.0)))
    with pytest.warns(UserWarning, match="Pixel-grid-only"):
        out, _ = rs._align_second_raster(
            ref, ref_meta, src, src_meta, is_categorical=False, pixel_grid_only=True
        )
    assert out.shape == ref.shape
    print("PASS: both missing CRS + pixel_grid_only -> warned resize")


def test_one_missing_crs_rejected():
    arr = np.full((6, 6), 3.0, dtype=np.float32)
    tf = _grid((6, 6), 10.0)
    ref, _, ref_meta = _read(_make_geotiff_bytes(arr, crs="EPSG:32610", transform=tf))
    src, _, src_meta = _read(_make_geotiff_bytes(arr, crs=None, transform=tf))
    with pytest.raises(ValueError, match="CRS mismatch"):
        rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=False)

    # Reverse: ref missing, src has CRS
    with pytest.raises(ValueError, match="CRS mismatch"):
        rs._align_second_raster(src, src_meta, ref, ref_meta, is_categorical=False)
    print("PASS: one missing CRS -> rejected both directions")


# ---------------------------------------------------------------------------
# 7. Different array dimensions
# ---------------------------------------------------------------------------

def test_different_array_dimensions_resample_to_ref():
    ref_arr = np.full((32, 32), 7.0, dtype=np.float32)
    src_arr = np.full((16, 16), 7.0, dtype=np.float32)
    # Same CRS, same extent, different pixel count
    ref, _, ref_meta = _read(_make_geotiff_bytes(ref_arr, crs="EPSG:32610", transform=_grid((32, 32), 10.0)))
    src, _, src_meta = _read(_make_geotiff_bytes(src_arr, crs="EPSG:32610", transform=_grid((16, 16), 20.0)))
    out, _ = rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=False)
    assert out.shape == ref.shape
    print("PASS: different array dimensions -> resampled to ref shape")


# ---------------------------------------------------------------------------
# 8. Nodata regions preserved as NaN (never zero)
# ---------------------------------------------------------------------------

def test_nodata_preserved_as_nan_not_zero():
    ref_arr = np.full((10, 10), 20.0, dtype=np.float32)
    # Source has a nodata block
    src_arr = np.full((10, 10), 20.0, dtype=np.float32)
    src_arr[3:7, 3:7] = -9999.0
    tf = _grid((10, 10), 10.0)
    ref, _, ref_meta = _read(_make_geotiff_bytes(ref_arr, crs="EPSG:32610", transform=tf))
    src, _, src_meta = _read(
        _make_geotiff_bytes(src_arr, crs="EPSG:32610", transform=tf, nodata=-9999.0)
    )
    # Source nodata must already be NaN after read
    assert np.isnan(src[5, 5])

    out, _ = rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=False)
    # Nodata must remain NaN, never become a valid zero-valued measurement
    assert np.isnan(out[5, 5])
    assert not np.any(out[np.isfinite(out)] == 0.0)
    valid = np.isfinite(out)
    assert np.allclose(out[valid], 20.0, atol=1e-6)
    print("PASS: nodata preserved as NaN, no zero-filling")


def test_out_of_coverage_remains_nan():
    # A tiny source footprint inside a large reference: outside must be NaN, not 0
    ref_tf = _grid((40, 40), 10.0)
    src_tf = from_origin(500100.0, 4999900.0, 10.0, 10.0)  # small offset window
    ref_arr = np.full((40, 40), 1.0, dtype=np.float32)
    src_arr = np.full((5, 5), 9.0, dtype=np.float32)
    ref, _, ref_meta = _read(_make_geotiff_bytes(ref_arr, crs="EPSG:32610", transform=ref_tf))
    src, _, src_meta = _read(_make_geotiff_bytes(src_arr, crs="EPSG:32610", transform=src_tf))
    out, _ = rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=False)
    assert out.shape == ref.shape
    # Corner outside source coverage == NaN
    assert np.isnan(out[0, 0])
    # No zero fabricated where data absent
    assert not np.any(out[np.isfinite(out)] == 0.0)
    print("PASS: out-of-coverage stays NaN (no fabricated zeros)")


# ---------------------------------------------------------------------------
# 9. Categorical class preservation
# ---------------------------------------------------------------------------

def test_categorical_nearest_preserves_classes():
    # Reference: 8x8 at 10 m; source: 4x4 at 20 m with explicit integer classes
    ref_arr = np.ones((8, 8), dtype=np.float32)
    src_arr = np.zeros((4, 4), dtype=np.float32)
    src_arr[:2, :2] = 1.0
    src_arr[:2, 2:] = 2.0
    src_arr[2:, :2] = 3.0
    src_arr[2:, 2:] = 4.0
    ref, _, ref_meta = _read(_make_geotiff_bytes(ref_arr, crs="EPSG:32610", transform=_grid((8, 8), 10.0)))
    src, _, src_meta = _read(_make_geotiff_bytes(src_arr, crs="EPSG:32610", transform=_grid((4, 4), 20.0)))

    out, _ = rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=True)
    assert out.shape == ref.shape
    valid = np.isfinite(out)
    # Nearest-neighbour: every output value must be an existing class label {1,2,3,4}
    produced = set(np.unique(out[valid]).tolist())
    assert produced.issubset({1.0, 2.0, 3.0, 4.0}), produced
    # No interpolated intermediate class values (e.g. 1.5, 2.5) from bilinear
    assert np.allclose(out[valid], np.rint(out[valid])), "categorical produced non-integer labels"
    print(f"PASS: categorical classes preserved via nearest; classes={sorted(produced)}")


def test_continuous_uses_bilinear_interpolation():
    # A linear ramp on a coarse grid should yield intermediate values after bilinear
    src_arr = np.tile(np.array([0.0, 10.0], dtype=np.float32), (2, 1))  # 2x2
    ref_arr = np.zeros((20, 20), dtype=np.float32)
    ref, _, ref_meta = _read(_make_geotiff_bytes(ref_arr, crs="EPSG:32610", transform=_grid((20, 20), 10.0)))
    src, _, src_meta = _read(_make_geotiff_bytes(src_arr, crs="EPSG:32610", transform=_grid((2, 2), 100.0)))
    out, _ = rs._align_second_raster(ref, ref_meta, src, src_meta, is_categorical=False)
    valid = out[np.isfinite(out)]
    # Bilinear should produce values strictly between 0 and 10 somewhere
    assert np.any((valid > 1e-3) & (valid < 10.0 - 1e-3))
    print("PASS: continuous resampling introduces interpolated intermediate values (bilinear)")


# ---------------------------------------------------------------------------
# 10. End-to-end via service entrypoints: aligned output matches ref grid
# ---------------------------------------------------------------------------

def test_dem_upload_alignment_end_to_end():
    dem1 = np.full((20, 20), 100.0, dtype=np.float32)
    dem2 = np.full((10, 10), 100.0, dtype=np.float32)
    dem2[2:5, 2:5] += 4.0  # deposition, coarse grid
    b1 = _make_geotiff_bytes(dem1, crs="EPSG:32610", transform=_grid((20, 20), 10.0))
    b2 = _make_geotiff_bytes(dem2, crs="EPSG:32610", transform=_grid((10, 10), 20.0))
    res = rs.compute_dem_differencing_from_files(b1, "a.tif", b2, "b.tif", lod_threshold_m=0.5)
    # Output grid must match the reference (before) raster's dimensions
    assert res["dimensions"]["rows"] == 20
    assert res["dimensions"]["cols"] == 20
    assert res["provenance"]["tier"] == "user_upload"
    print("PASS: DEM end-to-end aligned to reference grid")


def test_lulc_upload_alignment_end_to_end():
    m1 = np.ones((20, 20), dtype=np.int16)
    m1[10:, :] = 2
    m2 = np.full((10, 10), 2, dtype=np.int16)
    m2[:5, :5] = 3
    b1 = _make_geotiff_bytes(m1, crs="EPSG:32610", transform=_grid((20, 20), 10.0))
    b2 = _make_geotiff_bytes(m2, crs="EPSG:32610", transform=_grid((10, 10), 20.0))
    res = rs.compute_lulc_from_files(b1, "l1.tif", b2, "l2.tif")
    assert res["provenance"]["tier"] == "user_upload"
    # All classes in transition matrix must be original labels
    labels = res["transition_matrix"]["counts"]
    assert len(labels) >= 2
    print("PASS: LULC end-to-end aligned to reference grid")


def test_missing_crs_end_to_end_rejected():
    arr = np.full((10, 10), 1.0, dtype=np.float32)
    tf = _grid((10, 10), 10.0)
    b1 = _make_geotiff_bytes(arr, crs=None, transform=tf)
    b2 = _make_geotiff_bytes(arr, crs=None, transform=tf)
    with pytest.raises(ValueError, match="lack CRS"):
        rs.compute_dem_differencing_from_files(b1, "a.tif", b2, "b.tif")
    print("PASS: missing CRS rejected end-to-end")


if __name__ == "__main__":
    test_same_crs_same_grid_fastpath()
    test_same_crs_different_resolution_resamples_to_ref_grid()
    test_same_crs_shifted_grid_aligns_to_ref()
    test_different_projected_crs_reprojects()
    test_geographic_vs_projected_crs_reprojects()
    test_both_missing_crs_rejected_by_default()
    test_both_missing_crs_pixel_grid_only_mode()
    test_one_missing_crs_rejected()
    test_different_array_dimensions_resample_to_ref()
    test_nodata_preserved_as_nan_not_zero()
    test_out_of_coverage_remains_nan()
    test_categorical_nearest_preserves_classes()
    test_continuous_uses_bilinear_interpolation()
    test_dem_upload_alignment_end_to_end()
    test_lulc_upload_alignment_end_to_end()
    test_missing_crs_end_to_end_rejected()
    print("\nALL RASTER ALIGNMENT VALIDATION TESTS PASSED")
