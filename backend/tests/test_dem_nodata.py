"""Regression tests: DEM differencing must respect joint validity mask.

High-priority hardening: the DEM-of-Difference pipeline previously computed
``np.max`` / ``np.min`` / ``np.mean`` / ``np.median`` reductions over a
difference array without first ensuring any valid pixels existed.  When both
input DEMs were entirely nodata (or had no spatial overlap), NumPy reductions
raised unhandled exceptions.

The fixed pipeline:
  * validates both inputs before differencing
  * builds a joint validity mask (finite in BOTH rasters)
  * excludes invalid pixels from differencing, LoD classification, volumes
  * short-circuits with a safe structured result when zero valid pixels exist
  * reports valid / excluded pixel counts and min/max/mean/std on valid pixels

Synthetic GeoTIFFs and arrays are used to compare expected counts and volumes
against independently calculated values.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services import raster_service as rs


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

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


def _dem(before, after, **kwargs) -> dict:
    return rs.compute_dem_differencing(dem_before=before, dem_after=after, **kwargs)


# ---------------------------------------------------------------------------
# 1. Both DEMs completely valid / known positive & negative differences
# ---------------------------------------------------------------------------

def test_dem_both_valid_known_positive_negative_elevation_differences():
    """All-valid pair with known +/- changes: min/max/mean/std are exact."""
    z1 = np.full((8, 8), 100.0, dtype=float)
    z2 = z1.copy()
    z2[0:4, 0:4] += 5.0   # deposition +5
    z2[4:8, 4:8] -= 3.0   # erosion -3

    res = _dem(z1, z2, lod_threshold_m=0.5)
    s = res["summary"]
    assert s["valid_pixels"] == 64
    assert s["excluded_pixels"] == 0
    # 16 px * +5, 16 px * -3, 32 px * 0  => mean = (80 - 48) / 64 = 0.5
    assert abs(s["mean_change_m"] - 0.5) < 1e-6
    assert abs(s["max_change_m"] - 5.0) < 1e-6
    assert abs(s["min_change_m"] - (-3.0)) < 1e-6
    # LoD threshold 0.5 -> all +/- pixels significant
    assert abs(s["max_deposition_m"] - 5.0) < 1e-6
    assert abs(s["max_erosion_m"] - (-3.0)) < 1e-6
    # std: values are {+5 (16x), 0 (32x), -3 (16x)}; mean=0.5
    expected_var = (16 * (5 - 0.5) ** 2 + 32 * (0 - 0.5) ** 2 + 16 * (-3 - 0.5) ** 2) / 64
    assert abs(s["std_change_m"] - round(float(np.sqrt(expected_var)), 3)) < 0.01
    print(f"PASS: both valid -> mean={s['mean_change_m']}, std={s['std_change_m']}, min={s['min_change_m']}, max={s['max_change_m']}")


# ---------------------------------------------------------------------------
# 2. One DEM entirely nodata
# ---------------------------------------------------------------------------

def test_dem_one_all_nodata_no_crash_structured_result():
    """Before valid, after entirely nodata -> structured zero result, no crash."""
    z1 = np.full((6, 6), 50.0, dtype=float)
    z2 = np.full((6, 6), np.nan, dtype=float)
    res = _dem(z1, z2, lod_threshold_m=0.5)
    s = res["summary"]
    assert s["valid_pixels"] == 0
    assert s["excluded_pixels"] == 36
    assert s["mean_change_m"] == 0.0
    assert s["min_change_m"] == 0.0
    assert s["max_change_m"] == 0.0
    assert s["std_change_m"] == 0.0
    assert s["deposition_volume_m3"] == 0.0
    assert s["erosion_volume_m3"] == 0.0
    assert s["net_volume_change_m3"] == 0.0
    assert s["pct_area_changed"] == 0.0
    assert res["histogram"]["counts"] == []
    assert res["images"]["dod_png"]  # still rendered (transparent)
    print("PASS: one all-nodata DEM -> valid=0, excluded=36, structured zero result")


# ---------------------------------------------------------------------------
# 3. Both DEMs entirely nodata
# ---------------------------------------------------------------------------

def test_dem_both_all_nodata_short_circuit():
    """Both inputs entirely NaN -> all-invalid short circuit, no crash."""
    z1 = np.full((4, 4), np.nan, dtype=float)
    z2 = np.full((4, 4), np.nan, dtype=float)
    res = _dem(z1, z2, lod_threshold_m=0.75)
    s = res["summary"]
    assert s["valid_pixels"] == 0
    assert s["excluded_pixels"] == 16
    assert s["mean_elevation_change_m"] == 0.0
    assert s["median_change_m"] == 0.0
    assert s["pct_area_changed"] == 0.0
    print("PASS: both all-nodata DEMs -> valid=0, excluded=16, no crash")


# ---------------------------------------------------------------------------
# 4. Partial nodata
# ---------------------------------------------------------------------------

def test_dem_partial_nodata_excluded_from_volume():
    """Nodata block inside a changed region must not contribute to volume."""
    z1 = np.full((10, 10), 100.0, dtype=float)
    z2 = z1.copy()
    z2[2:6, 2:6] += 4.0      # 4x4 change
    z2[2:4, 2:4] = np.nan    # 2x2 nodata inside changed block
    res = _dem(z1, z2, lod_threshold_m=0.5, pixel_size_m=10.0)
    s = res["summary"]
    # 16 changed - 4 nodata = 12 valid changed pixels
    assert s["valid_pixels"] == 100 - 4
    assert s["excluded_pixels"] == 4
    # deposition volume = 12 px * 100 m^2 * 4 m = 4800 m^3
    assert abs(s["deposition_volume_m3"] - 4800.0) < 1.0
    print(f"PASS: partial nodata -> valid={s['valid_pixels']}, excluded={s['excluded_pixels']}, dep_vol={s['deposition_volume_m3']} (ref 4800)")


# ---------------------------------------------------------------------------
# 5. Different nodata patterns in before vs after
# ---------------------------------------------------------------------------

def test_dem_different_nodata_patterns_before_after():
    """Different nodata locations in before vs after -> union excluded."""
    z1 = np.full((10, 10), 100.0, dtype=float)
    z1[0:2, 0:2] = np.nan     # top-left in before
    z2 = z1.copy()
    z2[8:10, 8:10] = np.nan   # bottom-right in after
    res = _dem(z1, z2, lod_threshold_m=0.5)
    s = res["summary"]
    # 4 + 4 = 8 invalid pixels excluded (union of nodata regions)
    assert s["excluded_pixels"] == 8
    assert s["valid_pixels"] == 92
    assert s["pct_area_changed"] == 0.0
    print("PASS: different nodata patterns -> excluded=8, valid=92 (union)")


# ---------------------------------------------------------------------------
# 6. NaN and Infinity in either input
# ---------------------------------------------------------------------------

def test_dem_nan_and_inf_excluded_from_statistics():
    """NaN / Inf in either DEM must be excluded from validity and stats."""
    z1 = np.full((6, 6), 100.0, dtype=float)
    z2 = z1.copy()
    z1[1, 1] = np.nan
    z1[1, 2] = np.inf
    z2[3, 3] = np.nan
    z2[5, 5] = np.inf
    res = _dem(z1, z2, lod_threshold_m=0.5)
    s = res["summary"]
    assert s["excluded_pixels"] == 4
    assert s["valid_pixels"] == 32
    assert s["pct_area_changed"] == 0.0
    print("PASS: NaN + Inf excluded -> valid=32, excluded=4")


# ---------------------------------------------------------------------------
# 7. No spatial overlap (same CRS, disjoint extents)
# ---------------------------------------------------------------------------

def test_dem_no_spatial_overlap_structured_zero_result():
    """Same CRS but disjoint extents -> zero valid overlap, no crash."""
    # Before: 10 m grid at (500000, 5000000); After: 10 m grid offset far east
    tf1 = from_origin(500000.0, 5000000.0, 10.0, 10.0)
    tf2 = from_origin(600000.0, 5000000.0, 10.0, 10.0)  # 10 km east -> no overlap
    z1 = np.full((8, 8), 100.0, dtype=np.float32)
    z2 = np.full((8, 8), 105.0, dtype=np.float32)
    b1 = _make_geotiff_bytes(z1, crs="EPSG:32610", transform=tf1)
    b2 = _make_geotiff_bytes(z2, crs="EPSG:32610", transform=tf2)
    res = rs.compute_dem_differencing_from_files(b1, "a.tif", b2, "b.tif", lod_threshold_m=0.5)
    s = res["summary"]
    assert s["valid_pixels"] == 0
    assert s["excluded_pixels"] == 64
    assert s["mean_elevation_change_m"] == 0.0
    assert s["pct_area_changed"] == 0.0
    assert res["histogram"]["counts"] == []
    print("PASS: no spatial overlap -> valid=0, excluded=64, structured zero result")


# ---------------------------------------------------------------------------
# 8. GeoTIFF-based: one entirely nodata, both entirely nodata
# ---------------------------------------------------------------------------

def test_dem_geotiff_one_all_nodata():
    """GeoTIFF before valid, after entirely nodata -> structured zero result."""
    z1 = np.full((6, 6), 100.0, dtype=np.float32)
    z2 = np.full((6, 6), -9999.0, dtype=np.float32)
    b1 = _make_geotiff_bytes(z1, crs="EPSG:32610")
    b2 = _make_geotiff_bytes(z2, crs="EPSG:32610", nodata=-9999.0)
    res = rs.compute_dem_differencing_from_files(b1, "a.tif", b2, "b.tif", lod_threshold_m=0.5)
    s = res["summary"]
    assert s["valid_pixels"] == 0
    assert s["excluded_pixels"] == 36
    assert s["deposition_volume_m3"] == 0.0
    assert s["erosion_volume_m3"] == 0.0
    assert s["net_volume_change_m3"] == 0.0
    print("PASS: GeoTIFF one all-nodata -> valid=0, excluded=36, volumes=0")


def test_dem_geotiff_both_all_nodata():
    """GeoTIFF both entirely nodata -> structured zero result, no crash."""
    b1 = _make_geotiff_bytes(np.full((4, 4), -9999.0, dtype=np.float32), crs="EPSG:32610", nodata=-9999.0)
    b2 = _make_geotiff_bytes(np.full((4, 4), -9999.0, dtype=np.float32), crs="EPSG:32610", nodata=-9999.0)
    res = rs.compute_dem_differencing_from_files(b1, "a.tif", b2, "b.tif", lod_threshold_m=0.75)
    s = res["summary"]
    assert s["valid_pixels"] == 0
    assert s["excluded_pixels"] == 16
    assert s["mean_elevation_change_m"] == 0.0
    assert res["histogram"]["counts"] == []
    print("PASS: GeoTIFF both all-nodata -> valid=0, excluded=16, no crash")


# ---------------------------------------------------------------------------
# 9. Valid pixels surrounding nodata boundaries: no contamination
# ---------------------------------------------------------------------------

def test_dem_valid_pixels_surrounding_nodata_not_contaminated():
    """Valid pixels adjacent to nodata must retain exact elevation differences."""
    z1 = np.full((7, 7), 200.0, dtype=float)
    z2 = z1.copy()
    z2[3, 3] = np.nan           # nodata hole
    z2[0, 0] = 210.0            # known +10 m change far from hole
    z2[6, 6] = 195.0            # known -5 m change far from hole
    res = _dem(z1, z2, lod_threshold_m=0.5)
    s = res["summary"]
    assert s["valid_pixels"] == 48
    assert s["excluded_pixels"] == 1
    assert abs(s["max_change_m"] - 10.0) < 1e-6
    assert abs(s["min_change_m"] - (-5.0)) < 1e-6
    assert abs(s["mean_elevation_change_m"] - round((10.0 - 5.0) / 48.0, 3)) < 1e-6
    print(f"PASS: nodata hole excluded -> valid={s['valid_pixels']}, max={s['max_change_m']}, min={s['min_change_m']}")


# ---------------------------------------------------------------------------
# 10. API contract preservation: existing keys still present
# ---------------------------------------------------------------------------

def test_dem_api_contract_preserved_existing_keys():
    """Existing summary keys from the API contract must remain present."""
    z1 = np.full((5, 5), 100.0, dtype=float)
    z2 = z1.copy()
    z2[1:3, 1:3] += 2.0
    res = _dem(z1, z2, lod_threshold_m=0.5)
    s = res["summary"]
    for key in (
        "mean_elevation_change_m",
        "median_change_m",
        "max_deposition_m",
        "max_erosion_m",
        "pct_area_changed",
        "deposition_volume_m3",
        "erosion_volume_m3",
        "net_volume_change_m3",
    ):
        assert key in s, f"missing contract key {key}"
    # New hardening keys also present
    for key in ("valid_pixels", "excluded_pixels", "min_change_m", "max_change_m", "mean_change_m", "std_change_m"):
        assert key in s, f"missing hardening key {key}"
    print("PASS: API contract preserved (existing + new summary keys)")


# ---------------------------------------------------------------------------
# 11. Synthetic DEM pair sanity check
# ---------------------------------------------------------------------------

def test_dem_synthetic_pair_valid():
    """Synthetic DEM benchmark pair must remain fully valid."""
    res = rs.compute_dem_differencing(scenario="mount_st_helens_lahar", lod_threshold_m=0.75)
    s = res["summary"]
    assert s["valid_pixels"] > 0
    assert s["excluded_pixels"] == 0
    assert s["mean_elevation_change_m"] is not None
    print(f"PASS: synthetic DEM pair -> valid={s['valid_pixels']}, excluded=0")


if __name__ == "__main__":
    test_dem_both_valid_known_positive_negative_elevation_differences()
    test_dem_one_all_nodata_no_crash_structured_result()
    test_dem_both_all_nodata_short_circuit()
    test_dem_partial_nodata_excluded_from_volume()
    test_dem_different_nodata_patterns_before_after()
    test_dem_nan_and_inf_excluded_from_statistics()
    test_dem_no_spatial_overlap_structured_zero_result()
    test_dem_geotiff_one_all_nodata()
    test_dem_geotiff_both_all_nodata()
    test_dem_valid_pixels_surrounding_nodata_not_contaminated()
    test_dem_api_contract_preserved_existing_keys()
    test_dem_synthetic_pair_valid()
    print("\nALL DEM NODATA HARDENING TESTS PASSED")
