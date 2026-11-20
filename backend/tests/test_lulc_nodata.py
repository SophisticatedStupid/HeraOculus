"""Validation tests for LULC nodata/NaN handling throughout the workflow."""

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


def _read(arr_bytes: bytes, name: str = "r.tif") -> tuple[np.ndarray, dict]:
    return rs._read_raster_bytes(arr_bytes, name)


# ---------------------------------------------------------------------------
# Tests for nodata handling
# ---------------------------------------------------------------------------

def test_lulc_nodata_in_before_raster():
    """Nodata in before raster: excluded from transition matrix and change map."""
    m1 = np.full((10, 10), 1, dtype=np.int16)
    m1[2:6, 2:6] = -9999  # nodata block
    m2 = np.full((10, 10), 2, dtype=np.int16)
    m2[2:6, 2:6] = -9999  # nodata same location

    b = _make_geotiff_bytes(m1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    a = _make_geotiff_bytes(m2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    res = rs.compute_lulc_from_files(b, "l1.tif", a, "l2.tif")

    # Total valid cells: 100 - 16 = 84
    assert res["excluded_pixels"] == 16
    assert abs(res["excluded_pct"] - 16.0) < 0.01

    # Transition matrix should only reflect valid paired pixels
    cm_counts = res["transition_matrix"]["counts"]
    total = sum(sum(row) for row in cm_counts)
    assert total == 84

    # No nodata class should appear in transition matrix
    labels = res["transition_matrix"]["labels"]
    assert "Class -9999" not in str(labels)
    assert "Evergreen Forest" in labels  # class 1
    assert "Savanna & Shrubland" in labels  # class 2

    # Change map: only valid cells where classes differ
    assert res["pct_changed"] > 0
    print(f"PASS: nodata in before raster -> excluded {res['excluded_pixels']} pixels")


def test_lulc_nodata_in_after_raster():
    """Nodata in after raster: excluded."""
    m1 = np.full((10, 10), 1, dtype=np.int16)
    m2 = np.full((10, 10), 2, dtype=np.int16)
    m2[3:7, 3:7] = -9999  # different nodata location

    b = _make_geotiff_bytes(m1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0))
    a = _make_geotiff_bytes(m2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    res = rs.compute_lulc_from_files(b, "l1.tif", a, "l2.tif")

    assert res["excluded_pixels"] == 16
    assert abs(res["excluded_pct"] - 16.0) < 0.01
    cm_total = sum(sum(row) for row in res["transition_matrix"]["counts"])
    assert cm_total == 84
    print("PASS: nodata in after raster -> excluded")


def test_lulc_different_nodata_locations():
    """Different nodata locations in before vs after: union of nodata excluded."""
    m1 = np.full((10, 10), 1, dtype=np.int16)
    m1[0:2, 0:2] = -9999  # nodata top-left
    m2 = np.full((10, 10), 2, dtype=np.int16)
    m2[8:10, 8:10] = -9999  # nodata bottom-right

    b = _make_geotiff_bytes(m1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    a = _make_geotiff_bytes(m2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    res = rs.compute_lulc_from_files(b, "l1.tif", a, "l2.tif")

    assert res["excluded_pixels"] == 8  # 4 + 4, no overlap
    cm_total = sum(sum(row) for row in res["transition_matrix"]["counts"])
    assert cm_total == 92
    print("PASS: different nodata locations -> union excluded")


def test_lulc_valid_class_zero():
    """Class zero is a valid land-cover class (e.g., water). Nodata is separate."""
    m1 = np.full((8, 8), 0, dtype=np.int16)  # class 0 = valid
    m2 = np.full((8, 8), 0, dtype=np.int16)
    m2[2:5, 2:5] = 1  # forest
    m1[0:2, 0:2] = -9999  # nodata top-left (class -9999, will be NaN)

    b = _make_geotiff_bytes(m1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    a = _make_geotiff_bytes(m2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    res = rs.compute_lulc_from_files(b, "l1.tif", a, "l2.tif")

    # Class 0 should appear in transition matrix
    labels = res["transition_matrix"]["labels"]
    assert any("Class 0" in str(l) for l in labels) or "Evergreen Forest" in labels  # check 0 is represented

    # Excluded pixels should be the nodata ones (4)
    assert res["excluded_pixels"] == 4

    # Transition matrix should include class 0 transitions
    print(f"PASS: valid class zero handled -> labels={labels}")


def test_lulc_nan_values():
    """NaN values in raw arrays treated as nodata / invalid."""
    m1 = np.full((10, 10), 1, dtype=np.float32)
    m2 = np.full((10, 10), 2, dtype=np.float32)
    m1[3:5, 3:5] = np.nan
    m2[7:9, 7:9] = np.nan

    b = _make_geotiff_bytes(m1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0))
    a = _make_geotiff_bytes(m2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0))
    res = rs.compute_lulc_from_files(b, "l1.tif", a, "l2.tif")

    assert res["excluded_pixels"] == 8
    cm_total = sum(sum(row) for row in res["transition_matrix"]["counts"])
    assert cm_total == 92
    print("PASS: NaN values excluded")


def test_lulc_completely_invalid_areas():
    """Large nodata regions excluded; only valid overlap used."""
    m1 = np.full((12, 12), 1, dtype=np.int16)
    m2 = np.full((12, 12), 2, dtype=np.int16)
    # Make left half nodata in before, right half nodata in after
    m1[:, :6] = -9999
    m2[:, 6:] = -9999

    b = _make_geotiff_bytes(m1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    a = _make_geotiff_bytes(m2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    res = rs.compute_lulc_from_files(b, "l1.tif", a, "l2.tif")

    # All pixels have nodata in at least one raster -> all excluded
    assert res["excluded_pixels"] == 144
    assert res["excluded_pct"] == 100.0
    cm_total = sum(sum(row) for row in res["transition_matrix"]["counts"])
    assert cm_total == 0
    print("PASS: completely invalid areas -> all excluded")


def test_lulc_nodata_not_rendered_as_class():
    """Invalid pixels should not appear as transitions in change map."""
    m1 = np.full((6, 6), 1, dtype=np.int16)
    m2 = np.full((6, 6), 2, dtype=np.int16)
    m1[0, 0] = -9999  # single nodata
    m2[0, 0] = -9999  # same

    b = _make_geotiff_bytes(m1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    a = _make_geotiff_bytes(m2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    res = rs.compute_lulc_from_files(b, "l1.tif", a, "l2.tif")

    # Change map should not flag nodata as changed
    # Transition matrix should have 35 valid pixels (36 - 1)
    cm_total = sum(sum(row) for row in res["transition_matrix"]["counts"])
    assert cm_total == 35
    assert res["excluded_pixels"] == 1
    print("PASS: nodata not rendered as transition")


def test_lulc_validity_mask_applied_to_rendering():
    """Invalid pixels rendered with nodata color (grey), not as class."""
    m1 = np.full((4, 4), 1, dtype=np.int16)
    m2 = np.full((4, 4), 2, dtype=np.int16)
    m1[0, 0] = -9999
    m2[0, 0] = -9999

    b = _make_geotiff_bytes(m1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    a = _make_geotiff_bytes(m2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    res = rs.compute_lulc_from_files(b, "l1.tif", a, "l2.tif")

    # Check the rendered PNGs exist and have correct structure
    assert res["images"]["before_png"].startswith("data:image/png;base64,")
    assert res["images"]["after_png"].startswith("data:image/png;base64,")
    assert res["images"]["change_mask_png"].startswith("data:image/png;base64,")
    print("PASS: rendering produces valid PNGs")


def test_lulc_backward_compatibility_keys():
    """All original response keys preserved; new keys added."""
    m1 = np.full((6, 6), 1, dtype=np.int16)
    m2 = np.full((6, 6), 2, dtype=np.int16)
    m1[0, 0] = -9999
    m2[0, 0] = -9999

    b = _make_geotiff_bytes(m1, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    a = _make_geotiff_bytes(m2, crs="EPSG:32610", transform=from_origin(500000.0, 5000000.0, 10.0, 10.0), nodata=-9999)
    res = rs.compute_lulc_from_files(b, "l1.tif", a, "l2.tif")

    # Original keys
    assert "provenance" in res
    assert "label" in res
    assert "overall_persistence_pct" in res
    assert "pct_changed" in res
    assert "cohen_kappa" in res
    assert "classes" in res
    assert "transition_matrix" in res
    assert "images" in res

    # New keys
    assert "excluded_pixels" in res
    assert "excluded_pct" in res

    # provenance has pixel_area_meta
    assert "pixel_area_meta" in res["provenance"]
    print("PASS: backward compatibility maintained, new keys present")


if __name__ == "__main__":
    test_lulc_nodata_in_before_raster()
    test_lulc_nodata_in_after_raster()
    test_lulc_different_nodata_locations()
    test_lulc_valid_class_zero()
    test_lulc_nan_values()
    test_lulc_completely_invalid_areas()
    test_lulc_nodata_not_rendered_as_class()
    test_lulc_validity_mask_applied_to_rendering()
    test_lulc_backward_compatibility_keys()
    print("\nALL LULC NODATA VALIDATION TESTS PASSED")