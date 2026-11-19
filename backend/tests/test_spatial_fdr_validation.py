"""Deterministic validation for spatial Mann-Kendall tie correction, temporal slope,
and spatial FDR corrections. Uses independent reference calculations.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from app.services import spatial_fdr_service as svc


def _ref_mk_pixel(values: np.ndarray) -> tuple[float, float]:
    """Naive per-pixel Mann-Kendall with tie correction. Returns (Z, p)."""
    v = np.asarray(values, dtype=float)
    valid = v[np.isfinite(v)]
    n = valid.size
    if n < 5:
        return 0.0, 1.0

    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(valid[j] - valid[i])

    vals, counts = np.unique(valid, return_counts=True)
    tie_sum = float(np.sum(counts.astype(float) * (counts - 1.0) * (2.0 * counts + 5.0)))
    var_s = max((n * (n - 1.0) * (2.0 * n + 5.0) - tie_sum) / 18.0, 1e-9)

    if s > 0:
        z = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / np.sqrt(var_s)
    else:
        z = 0.0
    p = float(np.clip(2.0 * stats.norm.sf(abs(z)), 1e-15, 1.0))
    return float(z), p


def _ref_slope_decade(values: np.ndarray, time: np.ndarray) -> float:
    """Naive per-pixel OLS slope on decimal-year time, expressed per decade."""
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    t = np.asarray(time, dtype=float)[mask]
    y = v[mask]
    if t.size < 5:
        return 0.0
    slope = np.polyfit(t, y, 1)[0]
    return float(slope * 10.0)


def test_no_tie_raster_matches_reference():
    rng = np.random.default_rng(0)
    T, H, W = 12, 3, 4
    cube = 10.0 + 0.1 * np.arange(T)[:, None, None] + rng.normal(0, 0.5, (T, H, W))
    Z, p, slope, valid, n_valid = svc._vectorized_mk_and_slope(cube, np.arange(T, dtype=float))

    for h in range(H):
        for w in range(W):
            rz, rp = _ref_mk_pixel(cube[:, h, w])
            rs = _ref_slope_decade(cube[:, h, w], np.arange(T, dtype=float))
            assert abs(Z[h, w] - rz) < 1e-6, (h, w, Z[h, w], rz)
            assert abs(p[h, w] - rp) < 1e-6, (h, w, p[h, w], rp)
            assert abs(slope[h, w] - rs) < 1e-4, (h, w, slope[h, w], rs)
    assert np.all(valid)
    print("PASS: no-tie raster matches reference")


def test_quantized_raster_tie_correction():
    rng = np.random.default_rng(1)
    T, H, W = 15, 2, 3
    cube = np.round(5.0 + rng.normal(0, 1.0, (T, H, W)), 1)
    Z, p, slope, valid, n_valid = svc._vectorized_mk_and_slope(cube, np.arange(T, dtype=float))

    for h in range(H):
        for w in range(W):
            rz, rp = _ref_mk_pixel(cube[:, h, w])
            assert abs(Z[h, w] - rz) < 1e-6, (h, w, Z[h, w], rz)
            assert abs(p[h, w] - rp) < 1e-6

    col = cube[:, 0, 0]
    vals, counts = np.unique(col, return_counts=True)
    expected_tc = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    got_tc = svc._tie_correction_sum(np.sort(col)[:, None])[0]
    assert abs(got_tc - expected_tc) < 1e-9, (got_tc, expected_tc)
    print("PASS: quantized raster tie correction matches reference")


def test_constant_pixels():
    T, H, W = 12, 2, 2
    cube = np.full((T, H, W), 7.5)
    Z, p, slope, valid, n_valid = svc._vectorized_mk_and_slope(cube, np.arange(T, dtype=float))
    assert np.all(Z == 0.0)
    assert np.all(p == 1.0)
    assert np.all(slope == 0.0)
    assert np.all(valid)
    print("PASS: constant pixels handled (Z=0, p=1, slope=0)")


def test_missing_years():
    rng = np.random.default_rng(2)
    T, H, W = 15, 2, 2
    cube = 10.0 + 0.2 * np.arange(T)[:, None, None] + rng.normal(0, 0.3, (T, H, W))
    cube[2, 0, 0] = np.nan
    cube[5, 0, 0] = np.nan
    cube[9, 1, 1] = np.nan
    Z, p, slope, valid, n_valid = svc._vectorized_mk_and_slope(cube, np.arange(T, dtype=float))

    rz, rp = _ref_mk_pixel(cube[:, 0, 0])
    assert abs(Z[0, 0] - rz) < 1e-6
    assert abs(p[0, 0] - rp) < 1e-6
    rs = _ref_slope_decade(cube[:, 0, 0], np.arange(T, dtype=float))
    assert abs(slope[0, 0] - rs) < 1e-4
    assert n_valid[0, 0] == T - 2
    assert valid[0, 0]
    print("PASS: missing years skip NaNs correctly")


def test_monthly_observations_slope():
    T = 24
    s_month = 0.05
    y = 2.0 + s_month * np.arange(T)
    cube = np.repeat(y[:, None, None], 1, axis=1)
    time = np.arange(T, dtype=float) / 12.0
    _, _, slope, valid, _ = svc._vectorized_mk_and_slope(cube, time)
    expected = s_month * 12.0 * 10.0
    assert abs(slope[0, 0] - expected) < 1e-4, (slope[0, 0], expected)
    print(f"PASS: monthly slope/decade = {slope[0,0]:.4f} (expected {expected:.4f})")


def test_annual_observations_slope():
    T = 20
    s_year = 0.3
    y = 10.0 + s_year * np.arange(T)
    cube = np.repeat(y[:, None, None], 1, axis=1)
    _, _, slope, valid, _ = svc._vectorized_mk_and_slope(cube, np.arange(T, dtype=float))
    expected = s_year * 10.0
    assert abs(slope[0, 0] - expected) < 1e-4, (slope[0, 0], expected)
    print(f"PASS: annual slope/decade = {slope[0,0]:.4f} (expected {expected:.4f})")


def test_irregular_timestamps_slope():
    t = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 12.0])
    s_year = 0.5
    y = 4.0 + s_year * t
    cube = np.repeat(y[:, None, None], 1, axis=1)
    _, _, slope, valid, _ = svc._vectorized_mk_and_slope(cube, t)
    assert abs(slope[0, 0] - s_year * 10.0) < 1e-4

    dates = ["2000", "2001", "2003", "2006", "2010", "2012"]
    meta_t, meta = svc._resolve_time_axis(
        len(dates), timestamps=dates, time_step_years=None, is_synthetic=False
    )
    assert np.allclose(meta_t, [2000.0, 2001.0, 2003.0, 2006.0, 2010.0, 2012.0])
    assert meta["resolution"] == "irregular/custom"
    print("PASS: irregular timestamps slope correct")


def test_all_invalid_pixels():
    rng = np.random.default_rng(3)
    T, H, W = 12, 2, 3
    cube = 10.0 + 0.1 * np.arange(T)[:, None, None] + rng.normal(0, 0.3, (T, H, W))
    cube[:, 0, 1] = np.nan
    Z, p, slope, valid, n_valid = svc._vectorized_mk_and_slope(cube, np.arange(T, dtype=float))
    assert not valid[0, 1]
    assert p[0, 1] == 1.0
    assert slope[0, 1] == 0.0
    assert n_valid[0, 1] == 0
    print("PASS: all-invalid pixels excluded")


def test_small_grids():
    rng = np.random.default_rng(4)
    T, H, W = 6, 2, 2
    cube = 10.0 + 0.1 * np.arange(T)[:, None, None] + rng.normal(0, 0.2, (T, H, W))
    res = svc.run_spatial_fdr_analysis(
        cube_input=cube.tolist(), grid_rows=2, grid_cols=2, n_years=6, seed=4, time_step_years=1.0
    )
    assert res["dimensions"]["time_steps"] == 6
    assert res["temporal"]["n_time_steps"] == 6
    print("PASS: small grids (T=6) run")


def test_fdr_valid_and_conservative():
    """BH, BY, Holm, Bonferroni produce valid q-values in [0,1] with correct conservative ordering."""
    rng = np.random.default_rng(5)
    T, H, W = 14, 3, 3
    cube = 10.0 + 0.15 * np.arange(T)[:, None, None] + rng.normal(0, 0.3, (T, H, W))
    cube[:, 1, 1] = np.nan  # invalid pixel

    results = {}
    for method in ["bh", "by", "holm", "bonferroni"]:
        res = svc.run_spatial_fdr_analysis(
            cube_input=cube.tolist(), alpha=0.05, correction_method=method, seed=5, time_step_years=1.0
        )
        results[method] = res
        # q-values must be in [0,1] and FDR sig <= raw sig per pixel
        H, W_ = res["dimensions"]["rows"], res["dimensions"]["cols"]
        for r in range(H):
            for c in range(W_):
                q = res["grid"]["q_values"][r][c]
                raw_sig = res["grid"]["raw_significant"][r][c]
                fdr_sig = res["grid"]["fdr_significant"][r][c]
                assert 0.0 <= q <= 1.0
                assert fdr_sig <= raw_sig

    # BY should be >= BH on average (conservative under arbitrary dependence)
    bh_q = np.array([v for row in results["bh"]["grid"]["q_values"] for v in row])
    by_q = np.array([v for row in results["by"]["grid"]["q_values"] for v in row])
    valid_comp = (bh_q < 1.0) & (by_q < 1.0)
    if np.any(valid_comp):
        diff = by_q[valid_comp] - bh_q[valid_comp]
        print(f"  BY-BH diff mean: {diff.mean():.6f}")
        assert diff.mean() >= -1e-5, f"BY systematically lower than BH"

    # Holm and Bonferroni should be conservative FWER
    for method in ["holm", "bonferroni"]:
        q_flat = np.array([v for row in results[method]["grid"]["q_values"] for v in row])
        assert np.all((q_flat >= 0) & (q_flat <= 1.0))

    # False positives pruned should be non-negative
    for method in ["bh", "by", "holm", "bonferroni"]:
        res = results[method]
        s = res["summary"]
        assert s["false_positives_pruned"] >= 0
        assert s["fdr_significant_cells"] <= s["raw_significant_cells"]
    print("PASS: FDR/FWER corrections valid with correct ordering")


def test_reproducible_and_metadata():
    rng = np.random.default_rng(6)
    T, H, W = 12, 3, 3
    cube = 10.0 + 0.1 * np.arange(T)[:, None, None] + rng.normal(0, 0.3, (T, H, W))
    r1 = svc.run_spatial_fdr_analysis(cube_input=cube.tolist(), time_step_years=1.0, seed=6)
    r2 = svc.run_spatial_fdr_analysis(cube_input=cube.tolist(), time_step_years=1.0, seed=6)
    assert r1["grid"]["slope_decade"] == r2["grid"]["slope_decade"]
    assert r1["grid"]["q_values"] == r2["grid"]["q_values"]
    assert r1["temporal"]["n_time_steps"] == 12
    assert r1["temporal"]["time_step_years"] == 1.0
    print("PASS: reproducible + temporal metadata present")


def test_ambiguous_spacing_rejected():
    rng = np.random.default_rng(7)
    T, H, W = 12, 2, 2
    cube = 10.0 + rng.normal(0, 0.3, (T, H, W))
    try:
        svc.run_spatial_fdr_analysis(cube_input=cube.tolist(), seed=7)
    except ValueError as exc:
        assert "Ambiguous temporal spacing" in str(exc)
        print("PASS: ambiguous spacing rejected")
        return
    raise AssertionError("Expected ValueError for ambiguous temporal spacing")


def test_monthly_vs_annual_differs_correctly():
    T = 24
    s_month = 0.02
    y = 5.0 + s_month * np.arange(T)
    cube = np.repeat(y[:, None, None], 1, axis=1)
    annual = svc.run_spatial_fdr_analysis(cube_input=cube.tolist(), time_step_years=1.0, seed=1)
    monthly = svc.run_spatial_fdr_analysis(cube_input=cube.tolist(), time_step_years=1.0 / 12.0, seed=1)
    a = annual["grid"]["slope_decade"][0][0]
    m = monthly["grid"]["slope_decade"][0][0]
    assert abs(m / a - 12.0) < 1e-6, (a, m, m / a)
    print(f"PASS: monthly vs annual slope ratio = {m/a:.3f} (expected 12)")


def test_timestamp_parsing():
    t = svc._parse_timestamps_to_decimal_years(["2000", "2000-01", "2000-07", "2001"])
    assert abs(t[0] - 2000.0) < 1e-9
    assert abs(t[1] - 2000.0) < 1e-9
    assert abs(t[2] - (2000 + 6 / 12.0)) < 0.01
    assert abs(t[3] - 2001.0) < 1e-9
    print(f"PASS: timestamp parsing -> {np.round(t,4).tolist()}")


if __name__ == "__main__":
    test_no_tie_raster_matches_reference()
    test_quantized_raster_tie_correction()
    test_constant_pixels()
    test_missing_years()
    test_monthly_observations_slope()
    test_annual_observations_slope()
    test_irregular_timestamps_slope()
    test_all_invalid_pixels()
    test_small_grids()
    test_fdr_valid_and_conservative()
    test_reproducible_and_metadata()
    test_ambiguous_spacing_rejected()
    test_monthly_vs_annual_differs_correctly()
    test_timestamp_parsing()
    print("\nALL SPATIAL FDR VALIDATION TESTS PASSED")
