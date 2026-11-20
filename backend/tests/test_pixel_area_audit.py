"""Pixel-area and physical-unit audit for raster_service.py.

Covers projected CRS (affine determinant + unit conversion), geographic CRS
(geodesic per-cell area), non-square / rotated grids, feet-based units,
geodesic-dependent geographic area, nodata handling, and end-to-end DEM
volume / SAR area / LULC transition-area correctness against independent
geodesic or projected reference calculations.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import rasterio
from pyproj import Geod
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


# ---------------------------------------------------------------------------
# PART A: Projected CRS pixel-area validation
# ---------------------------------------------------------------------------

def test_10m_square_projected_pixel_area():
    arr = np.ones((6, 8), dtype=np.float32)
    tf = from_origin(500000.0, 5000000.0, 10.0, 10.0)  # 10 m square
    _, _, meta = _read(_make_geotiff_bytes(arr, crs="EPSG:32610", transform=tf))
    areas = rs._compute_pixel_areas(arr, meta["transform"], meta["crs"], area_kind="projected")
    assert areas.shape == arr.shape
    assert np.allclose(areas, 100.0, atol=1e-6)
    meta_out = rs._pixel_area_metadata(meta["crs"], areas)
    assert meta_out["methodology"] == "affine_determinant"
    assert meta_out["area_units"] == "m2"
    assert meta_out["crs"] is not None
    assert abs(meta_out["linear_units"] == "metre")
    assert meta_out["area_constant"] is True
    print("PASS: 10 m square projected pixel area = 100 m^2")


def test_rectangular_10x20m_pixel_area():
    arr = np.ones((6, 8), dtype=np.float32)
    tf = Affine(10.0, 0.0, 500000.0, 0.0, -20.0, 5000000.0)  # 10 m x 20 m
    _, _, meta = _read(_make_geotiff_bytes(arr, crs="EPSG:32610", transform=tf))
    areas = rs._compute_pixel_areas(arr, meta["transform"], meta["crs"], area_kind="projected")
    assert np.allclose(areas, 200.0, atol=1e-6), areas.flat[:3]
    print("PASS: 10 m x 20 m rectangular pixel area = 200 m^2")


def test_different_pixel_sizes():
    for px, expected in [(5.0, 25.0), (30.0, 900.0)]:
        arr = np.ones((4, 4), dtype=np.float32)
        tf = from_origin(500000.0, 5000000.0, px, px)
        _, _, meta = _read(_make_geotiff_bytes(arr, crs="EPSG:32610", transform=tf))
        areas = rs._compute_pixel_areas(arr, meta["transform"], meta["crs"], area_kind="projected")
        assert np.allclose(areas, expected, atol=1e-6), (px, areas.flat[0], expected)
    print("PASS: different pixel sizes handled")


def test_feet_based_projected_crs_unit_conversion():
    # EPSG:2236 = NAD83 / Florida East, US survey foot -> 0.3048006096 m per foot
    factor = rs._get_crs_linear_units_to_meters(rasterio.crs.CRS.from_epsg(2236))
    assert abs(factor - 0.30480060960121924) < 1e-12, factor
    # 10 ft x 10 ft pixels = 100 ft^2 = 100 * 0.30480060960121924^2 m^2
    arr = np.ones((4, 4), dtype=np.float32)
    tf = from_origin(500000.0, 5000000.0, 10.0, 10.0)
    _, _, meta = _read(_make_geotiff_bytes(arr, crs="EPSG:2236", transform=tf))
    areas = rs._compute_pixel_areas(arr, meta["transform"], meta["crs"], area_kind="projected")
    expected = 100.0 * (factor ** 2)
    assert np.allclose(areas, expected, atol=1e-6), (areas.flat[0], expected)
    meta_out = rs._pixel_area_metadata(meta["crs"], areas)
    assert meta_out["linear_units"] != "metre"
    print(f"PASS: feet-based CRS unit conversion factor={factor}")


def test_rotated_affine_transform_pixel_area():
    # Rotated grid: area from |a*e - b*d| handles rotation
    arr = np.ones((6, 8), dtype=np.float32)
    # 20 degree clockwise rotation, 10 m nominal pixel
    ang = np.deg2rad(20.0)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    scale = np.diag([10.0, 10.0])
    M = rot @ scale
    tf = Affine(M[0, 0], M[0, 1], 500000.0, M[1, 0], M[1, 1], 5000000.0)
    # area should be det(M) = 100 m^2 (rotation preserves area)
    _, _, meta = _read(_make_geotiff_bytes(arr, crs="EPSG:32610", transform=tf))
    areas = rs._compute_pixel_areas(arr, meta["transform"], meta["crs"], area_kind="projected")
    assert np.allclose(areas, 100.0, atol=1e-6), areas.flat[:3]
    print(f"PASS: rotated affine pixel area preserved = {areas.flat[0]:.6f} m^2")


# ---------------------------------------------------------------------------
# PART B: Geographic CRS geodesic pixel-area validation
# ---------------------------------------------------------------------------

def test_geographic_pixel_area_latitude_dependence():
    # Geographic: pixel area must shrink with latitude (cos(latitude)).
    lat_h, lat_l = 60.0, 10.0
    deg = 0.001
    for lat in (lat_h, lat_l):
        tf = from_origin(-114.0, lat, deg, deg)
        arr = np.ones((4, 4), dtype=np.float32)
        _, _, meta = _read(_make_geotiff_bytes(arr, crs="EPSG:4326", transform=tf))
        areas = rs._compute_pixel_areas(arr, meta["transform"], meta["crs"], area_kind="geographic")
        meta_out = rs._pixel_area_metadata(meta["crs"], areas)
        assert meta_out["methodology"] == "geodesic_per_cell"
        assert meta_out["area_units"] == "m2"
        # independent geodesic reference for the same cell
        geod = Geod(ellps="WGS84")
        a_ref, _ = geod.polygon_area_perimeter(
            [-114.0, -114.0 + deg, -114.0 + deg, -114.0],
            [lat, lat, lat + deg, lat + deg],
        )
        expected = abs(a_ref)
        assert np.allclose(areas.flat[0], expected, rtol=1e-3), (lat, areas.flat[0], expected)
    # Higher latitude -> smaller pixel area
    tf_h = from_origin(-114.0, lat_h, deg, deg)
    tf_l = from_origin(-114.0, lat_l, deg, deg)
    arr = np.ones((4, 4), dtype=np.float32)
    areas_h = rs._compute_pixel_areas(arr, tf_h, rasterio.crs.CRS.from_epsg("4326"), area_kind="geographic")
    areas_l = rs._compute_pixel_areas(arr, tf_l, rasterio.crs.CRS.from_epsg("4326"), area_kind="geographic")
    assert areas_h.flat[0] < areas_l.flat[0]
    print(f"PASS: geographic area latitude-dependent (60N={areas_h.flat[0]:.0f} < 10N={areas_l.flat[0]:.0f})")


def test_geographic_nonuniform_pixel_areas():
    # Different latitudes within one grid produce non-constant areas
    tf = from_origin(-114.0, 60.0, 1.0, 1.0)  # spans 60 -> 53 degrees latitude
    arr = np.ones((8, 4), dtype=np.float32)
    _, _, meta = _read(_make_geotiff_bytes(arr, crs="EPSG:4326", transform=tf))
    areas = rs._compute_pixel_areas(arr, meta["transform"], meta["crs"], area_kind="geographic")
    assert not np.allclose(areas, areas.flat[0], rtol=1e-9, atol=1e-3), "areas should vary with latitude"
    meta_out = rs._pixel_area_metadata(meta["crs"], areas)
    assert meta_out["area_constant"] is False
    print("PASS: geographic pixel areas spatially variable")


# ---------------------------------------------------------------------------
# PART C: DEM volume in cubic metres
# ---------------------------------------------------------------------------

def test_dem_volume_cubic_metres_independent_reference():
    # 10m square projected grid: pixel area 100 m^2
    dem1 = np.full((12, 12), 100.0, dtype=np.float32)
    dem2 = dem1.copy()
    dem2[4:8, 4:8] += 5.0   # 4x4 = 16 pixels * 100 m^2 * 5 m = 8000 m^3 deposition
    dem2[0:4, 0:4] -= 3.0   # 4x4 = 16 pixels * 100 m^2 * 3 m = 4800 m^3 erosion
    b = _make_geotiff_bytes(dem1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0))
    a = _make_geotiff_bytes(dem2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0))
    res = rs.compute_dem_differencing_from_files(b, "dem1.tif", a, "dem2.tif", lod_threshold_m=0.5)
    assert res["provenance"]["tier"] == "user_upload"
    assert abs(res["summary"]["deposition_volume_m3"] - 8000.0) < 10.0
    assert abs(res["summary"]["erosion_volume_m3"] - 4800.0) < 10.0
    assert abs(res["summary"]["net_volume_change_m3"] - 3200.0) < 10.0
    assert "pixel_area_meta" in res["provenance"]
    assert res["provenance"]["pixel_area_meta"]["area_units"] == "m2"
    print(f"PASS: DEM volume = dep {res['summary']['deposition_volume_m3']} erosion {res['summary']['erosion_volume_m3']} net {res['summary']['net_volume_change_m3']}")


def test_dem_volume_nodata_never_zero_filled():
    # Nodata block must not contribute to volume
    dem1 = np.full((10, 10), 100.0, dtype=np.float32)
    dem2 = np.full((10, 10), 100.0, dtype=np.float32)
    dem2[2:6, 2:6] += 4.0
    dem2[2:4, 2:4] = -9999.0  # nodata inside changed block
    b = _make_geotiff_bytes(dem1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0))
    a = _make_geotiff_bytes(dem2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999.0)
    res = rs.compute_dem_differencing_from_files(b, "d1.tif", a, "d2.tif", lod_threshold_m=0.5)
    # 4x4=16 changed pixels minus 2x2=4 nodata = 12 valid changed * 100 m^2 * 4 m = 4800 m^3
    assert abs(res["summary"]["deposition_volume_m3"] - 4800.0) < 10.0
    print("PASS: nodata excluded from DEM volume")


# ---------------------------------------------------------------------------
# PART D: LULC transition areas
# ---------------------------------------------------------------------------

def test_lulc_transition_areas_independent_reference():
    # 20x20 grid at 500m pixels => each pixel 250000 m^2 = 0.25 km^2
    m1 = np.ones((20, 20), dtype=np.int16)
    m1[10:, :] = 2
    m2 = m1.copy()
    m2[:5, :10] = 3  # 50 pixels 1->3
    # pixel_area = 500*500 = 250000 m^2 = 0.25 km^2
    b = _make_geotiff_bytes(m1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 500.0, 500.0))
    c = _make_geotiff_bytes(m2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 500.0, 500.0))
    res = rs.compute_lulc_from_files(b, "l1.tif", c, "l2.tif")
    assert res["provenance"]["tier"] == "user_upload"
    # area_km2 matrix: class 1->3 count 50 * 0.25 = 12.5 km^2
    area_km2 = res["transition_matrix"]["area_km2"]
    labels = res["transition_matrix"]["labels"]
    ci = labels.index("Evergreen Forest")
    cj = labels.index("Cropland")
    assert abs(area_km2[ci][cj] - 12.5) < 0.01, (area_km2[ci][cj], ci, cj, labels)
    # net area for class 3: +50 px * 0.25 = +12.5 km^2 ; class 1: -12.5 km^2
    by_class = {c["class_id"]: c for c in res["classes"]}
    assert abs(by_class[3]["net_area_km2"] - 12.5) < 0.01
    assert abs(by_class[1]["net_area_km2"] + 12.5) < 0.01
    assert "pixel_area_meta" in res["provenance"]
    print(f"PASS: LULC transition areas match reference (1->3 = {area_km2[ci][cj]} km^2)")


# ---------------------------------------------------------------------------
# PART E: SAR area in square kilometres
# ---------------------------------------------------------------------------

def test_sar_area_square_kilometres_independent_reference():
    # 36x36 grid at 20m pixels => each pixel 400 m^2 = 0.0004 km^2
    sar1 = np.full((36, 36), 0.40, dtype=np.float32)
    sar2 = sar1.copy()
    sar2[10:22, 10:22] = 0.04  # 12x12 = 144 pixels drop = 144 * 400 m^2 = 57600 m^2 = 0.0576 km^2
    b = _make_geotiff_bytes(sar1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 20.0, 20.0))
    a = _make_geotiff_bytes(sar2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 20.0, 20.0))
    res = rs.compute_sar_change_from_files(b, "s1.tif", a, "s2.tif", speckle_filter="none", threshold_db=3.0)
    assert abs(res["summary"]["backscatter_drop_area_km2"] - 0.0576) < 0.001
    assert abs(res["summary"]["total_area_km2"] - (36 * 36 * 400.0 / 1_000_000.0)) < 0.001
    print(f"PASS: SAR drop area = {res['summary']['backscatter_drop_area_km2']} km^2 (ref 0.0576)")


# ---------------------------------------------------------------------------
# PART F: Metadata fields present
# ---------------------------------------------------------------------------

def test_pixel_area_metadata_fields():
    arr = np.ones((6, 8), dtype=np.float32)
    tf = from_origin(500000.0, 5000000.0, 10.0, 10.0)
    b1 = _make_geotiff_bytes(arr, crs="EPSG:32610", transform=tf)
    b2 = _make_geotiff_bytes(arr + 1.0, crs="EPSG:32610", transform=tf)
    res = rs.compute_dem_differencing_from_files(b1, "a.tif", b2, "b.tif", lod_threshold_m=0.5)
    meta = res["provenance"]["pixel_area_meta"]
    for key in ("methodology", "area_units", "crs", "linear_units", "area_constant"):
        assert key in meta, meta
    assert meta["area_units"] == "m2"
    assert meta["methodology"] == "affine_determinant"
    assert meta["area_constant"] is True
    print("PASS: pixel_area_meta contains methodology, area_units, crs, linear_units, area_constant")


def test_metadata_synthetic_constant_area():
    res = rs.compute_dem_differencing(scenario="mount_st_helens_lahar", lod_threshold_m=0.75)
    meta = res["provenance"].get("pixel_area_meta")
    # synthetic path: no CRS; constant area
    assert meta is not None
    assert meta["area_constant"] is True
    assert meta["methodology"] == "synthetic_demo_constant_square_pixel"
    print("PASS: synthetic metadata constant area")


if __name__ == "__main__":
    test_10m_square_projected_pixel_area()
    test_rectangular_10x20m_pixel_area()
    test_different_pixel_sizes()
    test_feet_based_projected_crs_unit_conversion()
    test_rotated_affine_transform_pixel_area()
    test_geographic_pixel_area_latitude_dependence()
    test_geographic_nonuniform_pixel_areas()
    test_dem_volume_cubic_metres_independent_reference()
    test_dem_volume_nodata_never_zero_filled()
    test_lulc_transition_areas_independent_reference()
    test_sar_area_square_kilometres_independent_reference()
    test_pixel_area_metadata_fields()
    test_metadata_synthetic_constant_area()
    print("\nALL PIXEL-AREA AUDIT TESTS PASSED")
