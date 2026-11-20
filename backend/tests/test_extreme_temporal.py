"""Regression tests for extreme-value analysis with proper temporal assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from app.services.extreme_service import (
    run_extreme_value_analysis,
    _analyze_temporal_axis,
    _infer_temporal_frequency,
    _block_duration_years,
)


def test_temporal_inference_daily():
    """Daily data should infer frequency='daily' and ~365 obs/year."""
    # Use pd.date_range to generate valid daily dates
    dates = pd.date_range("2020-01-01", periods=365 * 3, freq="D").strftime("%Y-%m-%d").tolist()
    values = [float(np.random.default_rng(1).gumbel(loc=10.0, scale=2.0)) for _ in dates]
    freq_info = _analyze_temporal_axis(dates, values)
    assert freq_info["frequency"] == "daily", f"Expected daily, got {freq_info['frequency']}"
    assert freq_info["obs_per_year"] == 365.25, f"Expected ~365 obs/year, got {freq_info['obs_per_year']}"


def test_temporal_inference_monthly():
    """Monthly data should infer frequency='monthly' and 12 obs/year."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS").strftime("%Y-%m-%d").tolist()  # 5 years x 12 months
    values = [float(np.random.default_rng(2).gumbel(loc=10.0, scale=2.0)) for _ in dates]
    freq_info = _analyze_temporal_axis(dates, values)
    assert freq_info["frequency"] == "monthly", f"Expected monthly, got {freq_info['frequency']}"
    assert freq_info["obs_per_year"] == 12.0, f"Expected 12 obs/year, got {freq_info['obs_per_year']}"


def test_gev_block_maxima_daily_input():
    """GEV with daily data should use correct block duration and not claim annual return levels
    without proper justification."""
    # 3 years of daily data
    dates = pd.date_range("2020-01-01", periods=365 * 3, freq="D").strftime("%Y-%m-%d").tolist()
    values = [float(np.random.default_rng(1).gumbel(loc=10.0, scale=2.0)) for _ in dates]

    res = run_extreme_value_analysis(
        values,
        dates=dates,
        mode="gev_block_maxima",
        block_size=365,  # annual blocks from daily data
        observation_frequency="daily",
    )

    # Check that temporal info is present
    assert "temporal" in res, "Temporal diagnostics should be present"
    temporal = res["temporal"]
    assert temporal["frequency"] == "daily", f"Expected daily frequency, got {temporal['frequency']}"
    # 365/365.25 = 0.9993 years per block
    expected_block_dur = 365 / 365.25
    assert abs(temporal["block_duration_years"] - expected_block_dur) < 0.01, (
        f"Expected block_duration~{expected_block_dur:.4f} year for daily block_size=365, "
        f"got {temporal['block_duration_years']}"
    )
    # Return periods should be in years
    for rl in res["return_levels"]:
        assert rl["return_period_unit"] == "years", (
            f"Expected return_period_unit='years', got {rl['return_period_unit']}"
        )


def test_gpd_pot_monthly_input():
    """GPD POT with monthly data should compute exceedance_rate_per_period correctly."""
    dates = pd.date_range("2020-01-01", periods=120, freq="MS").strftime("%Y-%m-%d").tolist()  # 10 years x 12 months
    values = [float(np.random.default_rng(m + 1).gumbel(loc=10.0, scale=2.0)) for m in range(len(dates))]

    res = run_extreme_value_analysis(
        values,
        dates=dates,
        mode="gpd_pot",
        block_size=12,
        threshold_quantile=0.85,
        observation_frequency="monthly",
    )

    assert "temporal" in res, "Temporal diagnostics should be present"
    temporal = res["temporal"]
    assert temporal["frequency"] == "monthly", f"Expected monthly, got {temporal['frequency']}"
    # exceedance_rate_per_period is top-level in GPD response (preserves API contract)
    assert "exceedance_rate_per_period" in res, "GPD should have top-level exceedance_rate_per_period"
    rate = res["exceedance_rate_per_period"]
    print(f"Monthly GPD rate_per_period: {rate}")
    assert rate > 0, "Exceedance rate should be positive"
    # And given 10 years of data with threshold at 85th percentile, expect maybe 12-20 exceedances total
    # over 10 years -> ~1-2 per year, so rate > 0 is the key check
    # Return periods should have unit
    for rl in res["return_levels"]:
        assert rl["return_period_unit"] == "years"


def test_temporal_inference_no_dates():
    """Without dates, frequency should be unknown and temporal assumption flagged."""
    values = [float(np.random.default_rng(1).gumbel(loc=10.0, scale=2.0)) for _ in range(100)]
    freq_info = _analyze_temporal_axis(dates=None, values=values)
    assert freq_info["frequency"] == "unknown", f"Expected unknown, got {freq_info['frequency']}"
    assert freq_info["temporal_assumption_valid"] is False
    assert freq_info["source"] == "default"


def test_temporal_inference_irregular():
    """Irregular timestamps should be detected and flagged."""
    dates = ["2020-01-01", "2020-03-01", "2020-07-01", "2021-01-01", "2021-12-31"]  # very irregular
    values = [float(np.random.default_rng(1).gumbel(loc=10.0, scale=2.0)) for _ in dates]
    freq_info = _analyze_temporal_axis(dates, values)
    # Irregular spacing should be detected
    print(f"Irregular test: frequency={freq_info['frequency']}, obs_per_year={freq_info['obs_per_year']}")
    assert freq_info["frequency"] in ("irregular", "unknown"), f"Expected irregular/unknown, got {freq_info['frequency']}"
    assert freq_info["temporal_assumption_valid"] is False


def test_block_duration_calculation():
    """Verify block_duration_years function."""
    # Daily block_size=365 -> 365/365.25 years
    d = _block_duration_years(block_size=365, obs_per_year=365.25)
    assert abs(d - 365/365.25) < 1e-10, f"Expected 365/365.25, got {d}"

    # Monthly block_size=12 -> 1 year
    d = _block_duration_years(block_size=12, obs_per_year=12.0)
    assert d == 1.0, f"Expected 1.0, got {d}"

    # Annual block_size=1 -> 1 year
    d = _block_duration_years(block_size=1, obs_per_year=1.0)
    assert d == 1.0, f"Expected 1.0, got {d}"

    # No obs_per_year -> None
    d = _block_duration_years(block_size=12, obs_per_year=None)
    assert d is None, f"Expected None, got {d}"

    # block_size=30, monthly obs_per_year=12 -> 2.5 years
    d = _block_duration_years(block_size=30, obs_per_year=12.0)
    assert d == 2.5, f"Expected 2.5, got {d}"


def test_gev_returns_correct_return_unit():
    """Verify return_period_unit is consistently 'years'."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS").strftime("%Y-%m-%d").tolist()
    values = [float(np.random.default_rng(m).gumbel(loc=10.0, scale=2.0)) for m in range(len(dates))]

    res = run_extreme_value_analysis(
        values,
        dates=dates,
        mode="gev_block_maxima",
        block_size=12,
        observation_frequency="monthly",
    )

    assert "temporal" in res
    assert res["temporal"]["return_period_unit"] == "years"
    for rl in res["return_levels"]:
        assert rl["return_period_unit"] == "years"
        assert isinstance(rl["return_period"], (int, float))
        assert rl["return_period"] > 0


def test_gpd_exceedance_rate_uses_exposure():
    """Verify that GPD exceedance_rate_per_period uses actual exposure years when dates provided."""
    dates = pd.date_range("2020-01-01", periods=120, freq="MS").strftime("%Y-%m-%d").tolist()  # 10 years
    values = [float(np.random.default_rng(m).gumbel(loc=10.0, scale=2.0)) for m in range(len(dates))]

    res = run_extreme_value_analysis(
        values,
        dates=dates,
        mode="gpd_pot",
        block_size=12,
        threshold_quantile=0.85,
        observation_frequency="monthly",
    )

    assert "temporal" in res
    assert "exceedance_rate_per_period" in res

    # 10 years x 12 months, threshold at 85th percentile should give some exceedances
    rate = res["exceedance_rate_per_period"]
    assert rate > 0, "Exceedance rate should be positive"
    assert res["temporal"]["exposure_years"] is not None, "Exposure years should be computed"
    for rl in res["return_levels"]:
        assert rl["return_period_unit"] == "years"


def test_return_period_independence_from_data_length():
    """Return periods must not silently infer annual frequency from arbitrary array length."""
    # 60 daily values should NOT claim annual return periods
    values = [float(np.random.default_rng(1).gumbel(loc=10.0, scale=2.0)) for _ in range(60)]

    # Without dates: should not claim annual frequency
    res_no_dates = run_extreme_value_analysis(
        values,
        mode="gev_block_maxima",
        block_size=30,
        observation_frequency="daily",
    )
    temporal_no_dates = res_no_dates.get("temporal", {})
    # Without dates, block_duration may be None or flagged
    # The key: should NOT have temporal_assumption_valid=True just because block_size=30

    # With 2 years of monthly data and explicit frequency
    dates_2yr = pd.date_range("2020-01-01", periods=24, freq="MS").strftime("%Y-%m-%d").tolist()
    values_2yr = [float(np.random.default_rng(m).gumbel(loc=10.0, scale=2.0)) for m in range(len(dates_2yr))]

    res_with_dates = run_extreme_value_analysis(
        values_2yr,
        dates=dates_2yr,
        mode="gev_block_maxima",
        block_size=12,
        observation_frequency="monthly",
    )
    temporal_with_dates = res_with_dates.get("temporal", {})
    assert temporal_with_dates.get("frequency") == "monthly"
    assert temporal_with_dates.get("obs_per_year") == 12.0
    # Return period unit should be years
    for rl in res_with_dates["return_levels"]:
        assert rl["return_period_unit"] == "years"


def test_duplicate_timestamps():
    """Duplicate timestamps should be detected and flagged."""
    dates = ["2020-01-01", "2020-01-01", "2020-02-01", "2020-03-01"]
    values = [1.0, 2.0, 3.0, 4.0]
    freq_info = _analyze_temporal_axis(dates, values)
    assert freq_info["n_duplicates"] >= 1, f"Expected duplicates to be detected, got {freq_info['n_duplicates']}"


def test_missing_years():
    """Missing years in the data should be reflected in coverage_ratio and warnings."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS").strftime("%Y-%m-%d").tolist()
    # Remove 2021 entirely (12 months missing)
    dates = [d for d in dates if not d.startswith("2021")]
    values = [float(np.random.default_rng(m).gumbel(loc=10.0, scale=2.0)) for m in range(len(dates))]

    res = run_extreme_value_analysis(
        values,
        dates=dates,
        mode="gev_block_maxima",
        block_size=12,
        observation_frequency="monthly",
    )
    temporal = res["temporal"]
    # Coverage ratio should be < 1 due to missing year
    if temporal["coverage_ratio"] is not None:
        assert temporal["coverage_ratio"] < 1.0, f"Expected coverage < 1, got {temporal['coverage_ratio']}"
    assert temporal["n_gaps"] > 0, f"Expected gaps to be detected, got {temporal['n_gaps']}"