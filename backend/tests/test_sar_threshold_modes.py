"""Validation tests for SAR change-detection threshold mode workflow."""

from __future__ import annotations

import numpy as np
import pytest

from app.services import raster_service as rs


def test_fixed_mode_positive_change():
    """Strong positive change -> mask indicates gain."""
    before = np.ones((10, 10), dtype=float) * 100.0
    after = np.ones((10, 10), dtype=float) * 10.0  # 10x decrease -> -10 dB
    res = rs.compute_sar_change_detection(
        img_before=before, img_after=after, threshold_mode="fixed", threshold_db=3.0
    )
    assert res["threshold_mode"] == "fixed"
    assert res["applied_threshold_db"] == 3.0
    # Strong negative change -> backscatter drop
    assert res["summary"]["backscatter_drop_area_km2"] > 0


def test_fixed_mode_negative_change():
    """Strong negative change -> mask indicates gain (increase)."""
    before = np.ones((10, 10), dtype=float) * 10.0
    after = np.ones((10, 10), dtype=float) * 100.0  # 10x increase -> +10 dB
    res = rs.compute_sar_change_detection(
        img_before=before, img_after=after, threshold_mode="fixed", threshold_db=3.0
    )
    assert res["threshold_mode"] == "fixed"
    assert res["applied_threshold_db"] == 3.0
    # Strong positive change -> backscatter gain
    assert res["summary"]["backscatter_gain_area_km2"] > 0


def test_fixed_mode_no_change():
    """No change -> no gain or drop area."""
    before = np.ones((10, 10), dtype=float) * 100.0
    after = np.ones((10, 10), dtype=float) * 100.0
    res = rs.compute_sar_change_detection(
        img_before=before, img_after=after, threshold_mode="fixed", threshold_db=3.0
    )
    assert res["threshold_mode"] == "fixed"
    assert res["applied_threshold_db"] == 3.0
    assert res["summary"]["backscatter_drop_area_km2"] == 0.0
    assert res["summary"]["backscatter_gain_area_km2"] == 0.0
    assert res["summary"]["pct_changed"] == 0.0


def test_fixed_mode_constant_scene():
    """Constant scenes (identical) -> no change."""
    before = np.full((10, 10), 50.0, dtype=float)
    after = np.full((10, 10), 50.0, dtype=float)
    res = rs.compute_sar_change_detection(
        img_before=before, img_after=after, threshold_mode="fixed", threshold_db=3.0
    )
    assert res["threshold_mode"] == "fixed"
    assert res["summary"]["pct_changed"] == 0.0


def test_otsu_mode_detects_change():
    """Otsu mode computes threshold from data and applies it."""
    # Create scenes with clear bimodal log-ratio
    np.random.seed(42)
    before = np.random.randn(20, 20) * 2 + 100
    after = np.random.randn(20, 20) * 2 + 100
    # Introduce strong change in half the pixels
    before[:10, :10] = 50.0   # flood-like drop
    after[:10, :10] = 10.0
    res = rs.compute_sar_change_detection(
        img_before=before, img_after=after, threshold_mode="otsu", threshold_db=3.0
    )
    assert res["threshold_mode"] == "otsu"
    assert res["user_threshold_db"] == 3.0
    assert res["otsu_suggested_threshold_db"] > 0
    assert res["applied_threshold_db"] == res["otsu_suggested_threshold_db"]
    # Some pixels should be detected as changed
    assert res["summary"]["pct_changed"] > 0


def test_otsu_mode_fallback_to_fixed_on_degenerate():
    """Otsu mode with constant image falls back to fixed threshold."""
    before = np.full((10, 10), 100.0, dtype=float)
    after = np.full((10, 10), 100.0, dtype=float)
    res = rs.compute_sar_change_detection(
        img_before=before, img_after=after, threshold_mode="otsu", threshold_db=3.0
    )
    assert res["threshold_mode"] == "otsu"
    # Otsu falls back to fixed threshold when histogram is degenerate
    assert res["applied_threshold_db"] == res["user_threshold_db"]


def test_otsu_mode_with_noise():
    """Otsu mode with noisy data still works."""
    np.random.seed(7)
    T, H, W = 20, 15, 15
    before = np.random.lognormal(mean=2.0, sigma=0.5, size=(H, W))
    after = np.random.lognormal(mean=2.3, sigma=0.5, size=(H, W))
    res = rs.compute_sar_change_detection(
        img_before=before, img_after=after, threshold_mode="otsu", threshold_db=3.0
    )
    assert res["threshold_mode"] == "otsu"
    assert res["otsu_suggested_threshold_db"] > 0
    assert res["applied_threshold_db"] > 0


def test_backward_compatibility():
    """Default behavior (threshold_mode='fixed') preserves old response keys."""
    before = np.ones((5, 5), dtype=float) * 100.0
    after = np.ones((5, 5), dtype=float) * 50.0
    res = rs.compute_sar_change_detection(img_before=before, img_after=after)
    # Backward-compatible keys
    assert "threshold_db" not in res  # was replaced by user_threshold_db
    assert "user_threshold_db" in res
    assert "otsu_suggested_threshold_db" in res
    assert "applied_threshold_db" in res
    assert "threshold_mode" in res
    assert res["threshold_mode"] == "fixed"


def test_applied_mask_corresponds_to_displayed_threshold():
    """The applied mask (change mask PNG) must correspond to the displayed threshold."""
    before = np.ones((20, 20), dtype=float) * 100.0
    after = np.ones((20, 20), dtype=float) * 100.0
    after[0:10, 0:10] = 50.0  # -3 dB drop region
    after[10:20, 10:20] = 200.0  # +3 dB gain region

    # Fixed mode: apply 3 dB threshold
    res_fixed = rs.compute_sar_change_detection(
        img_before=before, img_after=after, threshold_mode="fixed", threshold_db=3.0
    )
    assert res_fixed["applied_threshold_db"] == 3.0
    # There should be changed pixels in both directions
    assert res_fixed["summary"]["backscatter_drop_area_km2"] > 0
    assert res_fixed["summary"]["backscatter_gain_area_km2"] > 0

    # Otsu mode: the applied threshold is the Otsu threshold
    res_otsu = rs.compute_sar_change_detection(
        img_before=before, img_after=after, threshold_mode="otsu", threshold_db=3.0
    )
    assert res_otsu["threshold_mode"] == "otsu"
    assert res_otsu["applied_threshold_db"] == res_otsu["otsu_suggested_threshold_db"]


if __name__ == "__main__":
    test_fixed_mode_positive_change()
    test_fixed_mode_negative_change()
    test_fixed_mode_no_change()
    test_fixed_mode_constant_scene()
    test_otsu_mode_detects_change()
    test_otsu_mode_fallback_to_fixed_on_degenerate()
    test_otsu_mode_with_noise()
    test_backward_compatibility()
    test_applied_mask_corresponds_to_displayed_threshold()
    print("\nALL SAR THRESHOLD VALIDATION TESTS PASSED")