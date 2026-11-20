"""Regression tests: SAR change detection must respect joint validity mask.

High-priority numerical bug fix: the SAR change-detection pipeline previously
replaced NaN values with a median (``np.nan_to_num(..., nan=np.nanmedian(...))``)
before computing the log-ratio.  This turned missing observations into apparently
valid measurements and produced false change classifications.

The fixed pipeline:
  * builds a joint validity mask from both inputs (finite in BOTH),
  * propagates the mask through speckle filtering / log-ratio / thresholding,
  * excludes invalid pixels from Otsu, changed-pixel counts and area stats,
  * uses valid pixels as the denominator for reported percentages,
  * reports valid / excluded / changed pixels and valid-data coverage.

Synthetic GeoTIFFs and arrays are used to compare expected pixel counts and
areas against independently calculated values.
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


def _sar(before, after, **kwargs) -> dict:
    return rs.compute_sar_change_detection(
        img_before=before, img_after=after, **kwargs
    )


# ---------------------------------------------------------------------------
# 1. Completely valid inputs / identical rasters
# ---------------------------------------------------------------------------

def test_sar_completely_valid_identical_rasters_no_change():
    """All-valid, identical rasters -> no change, full valid-data coverage."""
    before = np.full((8, 8), 100.0, dtype=float)
    after = before.copy()
    res = _sar(before, after, speckle_filter="none", threshold_mode="fixed", threshold_db=3.0)
    s = res["summary"]
    assert s["valid_pixels"] == 64
    assert s["excluded_pixels"] == 0
    assert s["changed_pixels"] == 0
    assert s["pct_changed"] == 0.0
    assert s["valid_data_coverage_pct"] == 100.0
    print("PASS: completely valid identical rasters -> no change, coverage=100%")


def test_sar_completely_valid_known_change_exact_counts():
    """Known change region -> changed pixel count and area vs independent calc."""
    before = np.full((10, 10), 100.0, dtype=float)
    after = before.copy()
    after[2:5, 2:5] = 10.0  # 3x3 = 9 pixels at -10 dB drop
    res = _sar(before, after, speckle_filter="none", threshold_mode="fixed", threshold_db=3.0)
    s = res["summary"]
    # 9 changed pixels, all valid
    assert s["valid_pixels"] == 100
    assert s["excluded_pixels"] == 0
    assert s["changed_pixels"] == 9
    # pct changed = 9 / 100 * 100
    assert abs(s["pct_changed"] - 9.0) < 0.01
    # area per pixel = 20m * 20m = 400 m^2 = 0.0004 km^2
    # drop area = 9 * 0.0004 = 0.0036 km^2 -> reported rounded to 3 decimals = 0.004
    assert abs(s["backscatter_drop_area_km2"] - 0.004) < 1e-6
    assert abs(s["backscatter_gain_area_km2"] - 0.0) < 1e-6
    print(f"PASS: known change -> changed={s['changed_pixels']} px, drop_area={s['backscatter_drop_area_km2']} km^2 (ref 0.0036)")


# ---------------------------------------------------------------------------
# 2. Specific regression: three valid pixels and one nodata pixel
# ---------------------------------------------------------------------------

def test_sar_three_valid_one_nodata_never_contributes_to_changed_area():
    """Two rasters with three valid pixels and one nodata pixel.

    The nodata pixel must never contribute to the changed area.  The remaining
    three valid pixels contain known changes; counts/areas are compared against
    independently calculated values.
    """
    before = np.array(
        [
            [100.0, 100.0],
            [100.0, np.nan],
        ],
        dtype=float,
    )
    after = np.array(
        [
            [100.0, 10.0],
            [5.0, 25.0],
        ],
        dtype=float,
    )

    res = _sar(before, after, speckle_filter="none", threshold_mode="fixed", threshold_db=3.0)
    s = res["summary"]

    # Pixel (1,1) is nodata in the before raster -> invalid (never paired).
    # The valid mask must exclude it; total = 4, valid = 3, excluded = 1.
    assert s["valid_pixels"] == 3
    assert s["excluded_pixels"] == 1
    assert s["valid_data_coverage_pct"] == round(100.0 * 3 / 4, 2)

    # Among valid pixels:
    #   (0,0): 100 -> 100  => 0 dB      -> unchanged
    #   (0,1): 100 -> 10   => -10 dB    -> changed (drop)
    #   (1,0): 100 -> 5    => -13.01 dB -> changed (drop)
    # The invalid (1,1) pixel shows a large apparent change (100 -> 25, -6 dB) if
    # median-replaced; it must be excluded instead.
    assert s["changed_pixels"] == 2
    # pct changed uses VALID pixels as denominator: 2 / 3 * 100
    assert abs(s["pct_changed"] - round(100.0 * 2 / 3, 2)) < 0.01
    # drop area: 2 changed * 20m * 20m = 2 * 400 m^2 = 800 m^2 = 0.0008 km^2 -> rounded = 0.001
    assert abs(s["backscatter_drop_area_km2"] - 0.001) < 1e-6
    # total valid area = 3 valid * 400 m^2 = 1200 m^2 = 0.0012 km^2 -> rounded = 0.001
    assert abs(s["total_area_km2"] - 0.001) < 1e-6

    print(
        f"PASS: 3 valid + 1 nodata -> changed={s['changed_pixels']} px, "
        f"excluded={s['excluded_pixels']}, pct_changed={s['pct_changed']} (ref {round(100.0 * 2 / 3, 2)})"
    )


# ---------------------------------------------------------------------------
# 3. NaN and Infinity handling
# ---------------------------------------------------------------------------

def test_sar_nan_and_inf_excluded_from_all_statistics():
    """NaN / Inf in either raster must be excluded from validity, counts and areas."""
    before = np.full((6, 6), 100.0, dtype=float)
    after = before.copy()
    before[1, 1] = np.nan    # nodata-like NaN in before
    before[1, 2] = np.inf    # Inf in before
    after[3, 3] = np.nan     # NaN in after
    after[5, 5] = np.inf     # Inf in after

    res = _sar(before, after, speckle_filter="none", threshold_mode="fixed", threshold_db=3.0)
    s = res["summary"]
    # 4 distinct invalid pixels (before NaN, before Inf, after NaN, after Inf)
    assert s["excluded_pixels"] == 4
    assert s["valid_pixels"] == 32
    assert s["changed_pixels"] == 0
    assert s["pct_changed"] == 0.0
    print("PASS: NaN + Inf excluded -> valid=32, excluded=4, changed=0")


# ---------------------------------------------------------------------------
# 4. All-invalid rasters
# ---------------------------------------------------------------------------

def test_sar_all_invalid_raster_short_circuit():
    """Completely invalid pair -> no crash, all pixels excluded, no change."""
    before = np.full((5, 5), np.nan, dtype=float)
    after = np.full((5, 5), np.inf, dtype=float)
    res = _sar(before, after, speckle_filter="none", threshold_mode="fixed", threshold_db=3.0)
    s = res["summary"]
    assert s["valid_pixels"] == 0
    assert s["excluded_pixels"] == 25
    assert s["changed_pixels"] == 0
    assert s["pct_changed"] == 0.0
    print("PASS: all-invalid raster -> valid=0, excluded=25, changed=0 (no crash)")


# ---------------------------------------------------------------------------
# 5. Valid pixels surrounding nodata boundaries (speckle filter)
# ---------------------------------------------------------------------------

def test_sar_speckle_filter_valid_pixels_near_nodata_not_contaminated():
    """Lee filter must not contaminate valid pixels via nodata neighbours."""
    before = np.full((9, 9), 100.0, dtype=float)
    after = before.copy()
    before[4, 4] = np.nan
    after[4, 4] = 20.0  # strong change at an invalid pixel -> must be excluded

    res = _sar(
        before, after,
        speckle_filter="lee", window_size=5, threshold_mode="fixed", threshold_db=3.0,
    )
    s = res["summary"]
    assert s["valid_pixels"] == 80
    assert s["excluded_pixels"] == 1
    # The only change is at an invalid pixel; no valid pixel should be flagged.
    assert s["changed_pixels"] == 0
    assert s["pct_changed"] == 0.0
    print("PASS: Lee filter near nodata -> valid=80, excluded=1, changed=0 (no contamination)")


# ---------------------------------------------------------------------------
# 6. GeoTIFF-based: different nodata patterns in both inputs
# ---------------------------------------------------------------------------

def test_sar_geotiff_different_nodata_patterns_before_after():
    """Different nodata locations in before vs after -> union excluded."""
    m1 = np.full((10, 10), 100.0, dtype=np.float32)
    m1[0:2, 0:2] = -9999  # nodata block top-left in before
    m2 = np.full((10, 10), 100.0, dtype=np.float32)
    m2[8:10, 8:10] = -9999  # nodata block bottom-right in after

    b1 = _make_geotiff_bytes(m1, nodata=-9999)
    b2 = _make_geotiff_bytes(m2, nodata=-9999)
    res = rs.compute_sar_change_from_files(b1, "b1.tif", b2, "b2.tif", speckle_filter="none", threshold_mode="fixed", threshold_db=3.0)
    s = res["summary"]
    # 4 + 4 = 8 invalid pixels excluded (union of nodata regions)
    assert s["excluded_pixels"] == 8
    assert s["valid_pixels"] == 92
    assert s["changed_pixels"] == 0
    print("PASS: GeoTIFF different nodata patterns -> excluded=8, valid=92")


# ---------------------------------------------------------------------------
# 7. GeoTIFF-based: all-invalid pair
# ---------------------------------------------------------------------------

def test_sar_geotiff_all_invalid_pair():
    """Completely invalid GeoTIFF pair -> all excluded, no change."""
    b1 = _make_geotiff_bytes(np.full((4, 4), -9999.0, dtype=np.float32), nodata=-9999)
    b2 = _make_geotiff_bytes(np.full((4, 4), -9999.0, dtype=np.float32), nodata=-9999)
    res = rs.compute_sar_change_from_files(b1, "b1.tif", b2, "b2.tif", speckle_filter="none", threshold_mode="fixed", threshold_db=3.0)
    s = res["summary"]
    assert s["valid_pixels"] == 0
    assert s["excluded_pixels"] == 16
    assert s["changed_pixels"] == 0
    assert s["pct_changed"] == 0.0
    print("PASS: GeoTIFF all-invalid pair -> valid=0, excluded=16, changed=0")


# ---------------------------------------------------------------------------
# 8. Synthetic benchmark pair (all valid) sanity check
# ---------------------------------------------------------------------------

def test_sar_synthetic_pair_all_valid():
    """Synthetic SAR benchmark pair must be treated as fully valid."""
    res = rs.compute_sar_change_detection(scenario="amazon_seasonal_flooding")
    s = res["summary"]
    assert s["valid_pixels"] > 0
    assert s["excluded_pixels"] == 0
    assert s["valid_data_coverage_pct"] == 100.0
    print("PASS: synthetic SAR pair -> fully valid, coverage=100%")


if __name__ == "__main__":
    test_sar_completely_valid_identical_rasters_no_change()
    test_sar_completely_valid_known_change_exact_counts()
    test_sar_three_valid_one_nodata_never_contributes_to_changed_area()
    test_sar_nan_and_inf_excluded_from_all_statistics()
    test_sar_all_invalid_raster_short_circuit()
    test_sar_speckle_filter_valid_pixels_near_nodata_not_contaminated()
    test_sar_geotiff_different_nodata_patterns_before_after()
    test_sar_geotiff_all_invalid_pair()
    test_sar_synthetic_pair_all_valid()
    print("\nALL SAR NODATA REGRESSION TESTS PASSED")
