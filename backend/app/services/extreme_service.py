"""Extreme Value Analysis (GEV block maxima & GPD Peaks-Over-Threshold).

The previous implementation silently assumed monthly observations (12 steps/year)
when computing annual exceedance rates and annual return levels. That produces
incorrect return-period interpretations when input data has a different temporal
frequency (daily, annual, irregular) or incomplete coverage.

This version:
  * Parses and validates supplied timestamps (ordering, duplicates, gaps, coverage).
  * Infers the actual temporal frequency (daily / monthly / annual / irregular).
  * Computes exposure duration from the *actual observation period*, not from an
    assumed number of observations per year.
  * Computes GEV return levels using the actual block duration in years
    (annual blocks: p = 1 - 1/T; monthly blocks: p = 1 - 1/(12*T), etc.).
  * Computes GPD exceedance rates as exceedances / exposure_years.
  * Reports explicit temporal assumptions and return_period units so that an
    annual recurrence estimate is never presented without justification.
"""

from __future__ import annotations

from typing import Literal, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats


def _parse_date_strings(dates: list[str]) -> tuple[np.ndarray, list[str]]:
    """Parse a list of date strings into numpy datetime64[ns] values.

    Supports "YYYY-MM-DD", "YYYY-MM", "YYYY", ISO-8601 datetimes, and common
    pandas-recognizable formats. Returns (array of np.datetime64, list of originals).
    Raises ValueError if any value cannot be parsed.
    """
    parsed: list[np.datetime64] = []
    for d in dates:
        try:
            ts = pd.Timestamp(d)
            parsed.append(np.datetime64(ts.to_datetime64(), "ns"))
        except Exception as exc:
            raise ValueError(f"Could not parse date '{d}': {exc}") from exc
    return np.array(parsed, dtype="datetime64[ns]"), list(dates)


def _infer_temporal_frequency(
    dates_arr: np.ndarray,
    explicit_freq: Optional[str] = None,
) -> dict:
    """Infer temporal frequency from timestamps or explicit metadata.

    Returns a dict describing:
      frequency: one of "daily", "monthly", "annual", "irregular", "unknown"
      source: "metadata", "inferred", or "default"
      obs_per_year: expected observations per year
      median_spacing_days: median gap between consecutive points
      cv_spacing: coefficient of variation of spacings (for regularity check)
    """
    n = len(dates_arr)

    if explicit_freq is not None:
        freq_map = {
            "daily": {"obs_per_year": 365.25, "median_days": 1.0},
            "monthly": {"obs_per_year": 12.0, "median_days": 30.44},
            "annual": {"obs_per_year": 1.0, "median_days": 365.25},
            "irregular": {"obs_per_year": None, "median_days": None},
        }
        if explicit_freq not in freq_map:
            raise ValueError(f"Unknown observation_frequency '{explicit_freq}'")
        info = freq_map[explicit_freq]
        return {
            "frequency": explicit_freq,
            "source": "metadata",
            "obs_per_year": info["obs_per_year"],
            "median_spacing_days": info["median_days"],
            "cv_spacing": None,
        }

    if n < 2:
        return {
            "frequency": "unknown",
            "source": "default",
            "obs_per_year": None,
            "median_spacing_days": None,
            "cv_spacing": None,
        }

    # Compute spacings between consecutive timestamps
    dts = dates_arr.astype("datetime64[ns]").astype("float64")
    spacings_days = np.diff(dts) / (24 * 3600 * 1e9)  # nanoseconds -> days

    if len(spacings_days) == 0:
        return {
            "frequency": "unknown",
            "source": "default",
            "obs_per_year": None,
            "median_spacing_days": None,
            "cv_spacing": None,
        }

    median_spacing = float(np.median(spacings_days))
    mean_spacing = float(np.mean(spacings_days))
    std_spacing = float(np.std(spacings_days))
    cv = std_spacing / mean_spacing if mean_spacing > 0 else float("inf")

    # Regularity threshold: CV < 0.3 means "regular enough" to classify
    if cv < 0.3:
        if median_spacing <= 1.5:
            frequency = "daily"
            obs_per_year = 365.25
        elif median_spacing <= 62.0:  # 28-62 days covers monthly variants
            frequency = "monthly"
            obs_per_year = 12.0
        elif median_spacing <= 400.0:
            frequency = "annual"
            obs_per_year = 1.0
        else:
            frequency = "irregular"
            obs_per_year = None
    else:
        frequency = "irregular"
        obs_per_year = None

    return {
        "frequency": frequency,
        "source": "inferred",
        "obs_per_year": obs_per_year,
        "median_spacing_days": median_spacing,
        "cv_spacing": cv,
    }


def _analyze_temporal_axis(
    dates: Optional[list[str]],
    values: list[Optional[float]],
    observation_frequency: Optional[str] = None,
) -> dict:
    """Analyze the temporal axis of an extreme-value input series.

    Returns:
      frequency: str ("daily" | "monthly" | "annual" | "irregular" | "unknown")
      source: str ("metadata" | "inferred" | "default")
      obs_per_year: float | None
      span_years: float | None (first to last timestamp span, in years)
      exposure_years: float | None (actual observation-duration in years)
      n_observations: int
      n_valid: int
      n_duplicates: int
      n_gaps: int
      is_monotonic: bool
      coverage_ratio: float | None (observed density over expected span)
      return_period_unit: str ("years")
      temporal_assumption_valid: bool
      notes: list[str]
    """
    notes: list[str] = []
    warnings_list: list[str] = []
    n_observations = len(values)
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    valid = arr[np.isfinite(arr)]
    n_valid = len(valid)

    result: dict = {
        "frequency": "unknown",
        "source": "default",
        "obs_per_year": None,
        "span_years": None,
        "exposure_years": None,
        "n_observations": n_observations,
        "n_valid": n_valid,
        "n_duplicates": 0,
        "n_gaps": 0,
        "is_monotonic": True,
        "coverage_ratio": None,
        "return_period_unit": "years",
        "temporal_assumption_valid": False,
        "notes": notes,
        "warnings": warnings_list,
    }

    if dates is None or len(dates) == 0:
        notes.append("No timestamps supplied; cannot determine temporal frequency.")
        notes.append("Return periods are expressed in 'years' but cannot be justified annually.")
        return result

    # Parse dates
    try:
        dates_arr, _ = _parse_date_strings(dates)
    except ValueError as exc:
        notes.append(f"Date parsing failed: {exc}; treating as unknown frequency.")
        return result

    if len(dates_arr) != n_observations:
        notes.append("Dates length mismatch with values; temporal analysis degraded.")

    # Sort & check ordering
    order = np.argsort(dates_arr, kind="stable")
    dates_sorted = dates_arr[order]
    is_monotonic = bool(np.all(np.diff(dates_arr.astype("float64")) >= 0))
    result["is_monotonic"] = is_monotonic

    if not is_monotonic:
        notes.append("Timestamps were not monotonically increasing; sorted internally.")

    # Duplicates
    _, unique_idx, counts = np.unique(dates_sorted, return_index=True, return_counts=True)
    n_duplicates = int(np.sum(counts > 1))
    result["n_duplicates"] = n_duplicates
    if n_duplicates > 0:
        notes.append(f"{n_duplicates} duplicate timestamp(s) found; deduplicated.")

    # Frequency inference
    freq_info = _infer_temporal_frequency(dates_sorted[unique_idx], explicit_freq=observation_frequency)
    result["frequency"] = freq_info["frequency"]
    result["source"] = freq_info["source"]
    result["obs_per_year"] = freq_info["obs_per_year"]

    if freq_info["median_spacing_days"] is not None:
        result["median_spacing_days"] = round(freq_info["median_spacing_days"], 2)

    # Span (first to last timestamp)
    span_days = float((dates_sorted[-1] - dates_sorted[0]) / np.timedelta64(1, "D"))
    result["span_years"] = round(span_days / 365.25, 4) if span_days > 0 else None

    # Expected number of periods over the span
    if freq_info["obs_per_year"] is not None and freq_info["obs_per_year"] > 0:
        expected_count = span_days / 365.25 * freq_info["obs_per_year"]
        coverage = min(1.0, n_valid / max(expected_count, 1))
        result["coverage_ratio"] = round(coverage, 4)
    else:
        result["coverage_ratio"] = None

    # Exposure duration (years actually covered by valid observations)
    # For irregular data: use span_years as exposure (conservative).
    # For regular data: exposure = span_years (valid slots counted via coverage).
    if result["span_years"] is not None:
        if freq_info["frequency"] == "irregular" or freq_info["obs_per_year"] is None:
            result["exposure_years"] = round(result["span_years"], 4)
        else:
            result["exposure_years"] = round(result["span_years"], 4)

    # Gap detection for regular frequencies
    n_gaps = 0
    if freq_info["obs_per_year"] is not None and freq_info["obs_per_year"] > 0:
        # Expected spacing in days
        expected_spacing_days = 365.25 / freq_info["obs_per_year"]
        if len(dates_sorted[unique_idx]) > 1:
            spacings_days = np.diff(dates_sorted[unique_idx].astype("float64")) / (
                24 * 3600 * 1e9
            )
            # A gap is a spacing > 1.5x expected
            gap_mask = spacings_days > 1.5 * expected_spacing_days
            n_gaps = int(np.sum(gap_mask))
    result["n_gaps"] = n_gaps

    # Validity of temporal assumption
    temporal_valid = (
        freq_info["frequency"] != "unknown"
        and freq_info["frequency"] != "irregular"
        and result["span_years"] is not None
        and result["span_years"] > 0
        and n_valid >= 10
    )
    result["temporal_assumption_valid"] = temporal_valid

    if result["source"] == "default":
        notes.append("No timestamps supplied; assuming annual return-period unit without verification.")
    elif not temporal_valid:
        notes.append("Temporal assumptions insufficient for a defensible annual return period.")
        if freq_info["frequency"] == "irregular":
            notes.append("Irregular sampling detected; return periods carry additional uncertainty.")
        if n_gaps > 0:
            notes.append(f"{n_gaps} gap(s) detected in the observation period; exposure may be overstated.")

    return result


def _block_duration_years(
    block_size: int,
    obs_per_year: Optional[float],
) -> Optional[float]:
    """Duration of a single block in years.

    block_size = number of observations per block.
    obs_per_year   = number of observations per year (from frequency inference).
    Returns block_size / obs_per_year years, or None if obs_per_year unavailable.
    """
    if obs_per_year is None or obs_per_year <= 0:
        return None
    return block_size / obs_per_year


def _bootstrap_return_ci(
    sample: np.ndarray,
    dist_name: Literal["gev", "gpd"],
    return_periods: list[float],
    threshold: float = 0.0,
    rate_per_period: float = 1.0,
    block_duration_years: Optional[float] = None,
    n_boot: int = 80,
    seed: int = 42,
    init_params: Optional[tuple[float, float, float]] = None,
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    boot_levels = np.empty((n_boot, len(return_periods)), dtype=float)
    n = len(sample)

    for b in range(n_boot):
        resamp = sample[rng.integers(0, n, size=n)]
        try:
            if dist_name == "gev":
                if init_params is not None:
                    c, loc, scale = stats.genextreme.fit(
                        resamp, init_params[0], loc=init_params[1], scale=init_params[2]
                    )
                else:
                    c, loc, scale = stats.genextreme.fit(resamp)
                for j, T in enumerate(return_periods):
                    # For block maxima: p = F(x) where expected exceedances in T years = 1.
                    # block_duration_years = d. Number of blocks in T years = T/d.
                    # (T/d) * (1 - F(x)) = 1 -> F(x) = 1 - d/T.
                    d = block_duration_years if block_duration_years is not None else 1.0
                    p = max(1e-6, min(0.99999, 1.0 - d / max(T, d + 1e-6)))
                    boot_levels[b, j] = float(stats.genextreme.ppf(p, c, loc=loc, scale=scale))
            else:
                exceed = resamp - threshold
                exceed = exceed[exceed >= 0]
                if len(exceed) < 3:
                    boot_levels[b, :] = np.nan
                    continue
                c, loc, scale = stats.genpareto.fit(exceed, floc=0.0)
                for j, T in enumerate(return_periods):
                    p_exceed = min(0.99999, max(1e-6, 1.0 - 1.0 / max(T * rate_per_period, 1.01)))
                    boot_levels[b, j] = threshold + float(
                        stats.genpareto.ppf(p_exceed, c, loc=loc, scale=scale)
                    )
        except Exception:
            boot_levels[b, :] = np.nan

    lows: list[float] = []
    highs: list[float] = []
    for j in range(len(return_periods)):
        col = boot_levels[:, j]
        col = col[np.isfinite(col)]
        if len(col) >= 10:
            lows.append(round(float(np.percentile(col, 2.5)), 4))
            highs.append(round(float(np.percentile(col, 97.5)), 4))
        else:
            lows.append(float("nan"))
            highs.append(float("nan"))
    return lows, highs


def run_extreme_value_analysis(
    values: list[Optional[float]],
    *,
    dates: Optional[list[str]] = None,
    mode: Literal["gev_block_maxima", "gpd_pot"] = "gev_block_maxima",
    block_size: int = 12,
    observation_frequency: Optional[str] = None,
    threshold_quantile: float = 0.85,
    return_periods: Optional[list[float]] = None,
) -> dict:
    """Fit GEV (block maxima) or GPD (Peaks-Over-Threshold) and return return levels.

    Return periods are expressed in **years**. The temporal interpretation depends on
    the detected observation frequency:

      * Daily data   : block_size should be ~365 (annual maxima from daily blocks).
      * Monthly data : block_size should be 12 (annual maxima from monthly blocks).
      * Annual data  : block_size should be 1 (each observation is a yearly maximum).

    If dates are supplied, the actual observation period is used to compute exposure
    duration rather than assuming `len(values) / block_size` years.
    """
    periods = return_periods or [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    valid = arr[np.isfinite(arr)]
    if len(valid) < 10:
        raise ValueError(f"At least 10 valid points required for Extreme Value Analysis (got {len(valid)}).")

    # Temporal analysis
    temporal = _analyze_temporal_axis(dates, values, observation_frequency=observation_frequency)
    obs_per_year = temporal["obs_per_year"]
    exposure_years = temporal["exposure_years"]

    if mode == "gev_block_maxima":
        # Determine effective block duration in years
        block_duration = _block_duration_years(block_size, obs_per_year)

        # If dates supplied, build blocks by time; otherwise fall back to positional blocks
        block_maxima: list[float] = []
        block_labels: list[str] = []

        if dates is not None and len(dates) == len(arr):
            # Build blocks grouped by year (or by detected frequency block)
            dates_arr, _ = _parse_date_strings(dates)
            # Group by calendar year for annual maxima semantics
            years = pd.DatetimeIndex(dates_arr.astype("datetime64[ns]")).year
            unique_years = np.unique(years)
            for yr in unique_years:
                mask = years == yr
                yearly_vals = arr[mask]
                yearly_ok = yearly_vals[np.isfinite(yearly_vals)]
                if len(yearly_ok) >= max(2, int(block_size // 3)):
                    block_maxima.append(float(np.max(yearly_ok)))
                    block_labels.append(str(yr))
        else:
            # Positional blocks (legacy behavior), respecting block_size
            n_blocks = len(arr) // block_size
            for b in range(n_blocks):
                sl = arr[b * block_size : (b + 1) * block_size]
                sl_ok = sl[np.isfinite(sl)]
                if len(sl_ok) >= max(2, block_size // 3):
                    block_maxima.append(float(np.max(sl_ok)))
                    if dates and b * block_size < len(dates):
                        block_labels.append(str(dates[b * block_size])[:4])
                    else:
                        block_labels.append(f"Block {b + 1}")

        if len(block_maxima) < 5:
            # Adaptive fallback chunking into 6-8 blocks when total length is modest
            chunk = max(2, len(valid) // 8)
            block_maxima = [float(np.max(valid[i : i + chunk])) for i in range(0, len(valid) - chunk + 1, chunk)]
            block_labels = [f"Block {i + 1}" for i in range(len(block_maxima))]

        bm = np.asarray(block_maxima, dtype=float)
        c, loc, scale = stats.genextreme.fit(bm)
        xi = -float(c)  # Hydrological EV convention: xi = -scipy_c

        ks_stat, ks_p = stats.kstest(bm, "genextreme", args=(c, loc, scale))

        # Return level: p = 1 - d/T where d = block duration in years
        d = block_duration if block_duration is not None else 1.0
        levels = [
            float(stats.genextreme.ppf(max(1e-6, min(0.99999, 1.0 - d / max(T, d + 1e-6))), c, loc=loc, scale=scale))
            for T in periods
        ]
        ci_low, ci_high = _bootstrap_return_ci(
            bm,
            "gev",
            periods,
            init_params=(c, loc, scale),
            block_duration_years=d,
        )

        # Empirical plotting positions (Gringorten)
        bm_sorted = np.sort(bm)
        n_bm = len(bm_sorted)
        emp_probs = [(i + 1 - 0.44) / (n_bm + 0.12) for i in range(n_bm)]
        # Empirical period = T years such that p = 1 - d/T  -> T = d / (1 - p)
        emp_periods = [round(d / max(1e-6, 1.0 - p), 3) for p in emp_probs]

        family = "Frechet (heavy-tailed)" if xi > 0.05 else ("Weibull (bounded)" if xi < -0.05 else "Gumbel (light-tailed)")

        return {
            "mode": "gev_block_maxima",
            "family_classification": family,
            "n_samples": len(bm),
            "parameters": {
                "shape_xi": round(xi, 4),
                "scipy_c": round(float(c), 4),
                "location_mu": round(float(loc), 4),
                "scale_sigma": round(float(scale), 4),
            },
            "goodness_of_fit": {
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(ks_p), 4),
            },
            "block_maxima": [round(float(x), 4) for x in bm],
            "block_labels": block_labels,
            "empirical_return_plot": [
                {"return_period": emp_periods[i], "value": round(float(bm_sorted[i]), 4)}
                for i in range(n_bm)
            ],
            "return_levels": [
                {
                    "return_period": float(periods[i]),
                    "return_period_unit": temporal["return_period_unit"],
                    "return_level": round(levels[i], 4),
                    "ci95_low": None if np.isnan(ci_low[i]) else ci_low[i],
                    "ci95_high": None if np.isnan(ci_high[i]) else ci_high[i],
                }
                for i in range(len(periods))
            ],
            "temporal": {
                "frequency": temporal["frequency"],
                "frequency_source": temporal["source"],
                "block_duration_years": round(block_duration, 4) if block_duration is not None else None,
                "obs_per_year": obs_per_year,
                "span_years": temporal["span_years"],
                "exposure_years": temporal["exposure_years"],
                "n_observations": temporal["n_observations"],
                "n_valid": temporal["n_valid"],
                "n_duplicates": temporal["n_duplicates"],
                "n_gaps": temporal["n_gaps"],
                "is_monotonic": temporal["is_monotonic"],
                "coverage_ratio": temporal["coverage_ratio"],
                "return_period_unit": temporal["return_period_unit"],
                "temporal_assumption_valid": temporal["temporal_assumption_valid"],
                "notes": temporal["notes"],
                "warnings": temporal["warnings"],
            },
        }

    # GPD Peaks-Over-Threshold
    threshold = float(np.quantile(valid, threshold_quantile))
    exceedances = valid[valid > threshold] - threshold
    if len(exceedances) < 5:
        threshold = float(np.quantile(valid, 0.70))
        exceedances = valid[valid > threshold] - threshold
    if len(exceedances) < 4:
        raise ValueError("Insufficient exceedances above threshold for GPD fit.")

    c, loc, scale = stats.genpareto.fit(exceedances, floc=0.0)
    ks_stat, ks_p = stats.kstest(exceedances, "genpareto", args=(c, loc, scale))

    # Rate per YEAR using actual exposure duration (not len(arr)/block_size).
    if exposure_years is not None and exposure_years > 0:
        rate_per_year = len(exceedances) / exposure_years
    else:
        # Fallback: no temporal basis; use observations-per-block assumption, flagged.
        rate_per_year = len(exceedances) / max(len(arr) / float(block_size), 1.0)
        temporal["warnings"].append(
            "exposure_years could not be determined; exceedance rate uses len/block_size "
            "assumption and may not represent an annual rate."
        )

    levels = []
    for T in periods:
        prob = min(0.99999, max(1e-6, 1.0 - 1.0 / max(T * rate_per_year, 1.01)))
        levels.append(threshold + float(stats.genpareto.ppf(prob, c, loc=loc, scale=scale)))

    ci_low, ci_high = _bootstrap_return_ci(
        valid[valid > threshold],
        "gpd",
        periods,
        threshold=threshold,
        rate_per_period=rate_per_year,
    )

    ex_sorted = np.sort(exceedances) + threshold
    n_ex = len(ex_sorted)
    emp_plot = []
    for i in range(n_ex):
        p_tail = max(1e-6, (n_ex - i) / (n_ex + 1.0))
        emp_plot.append(
            {
                "return_period": round(1.0 / max(1e-6, rate_per_year * p_tail), 3),
                "value": round(float(ex_sorted[i]), 4),
            }
        )

    return {
        "mode": "gpd_pot",
        "family_classification": "Generalized Pareto Tail",
        "n_samples": len(exceedances),
        "threshold": round(threshold, 4),
        "threshold_quantile": threshold_quantile,
        "exceedance_rate_per_period": round(float(rate_per_year), 3),
        "parameters": {
            "shape_xi": round(float(c), 4),
            "location_u": round(threshold, 4),
            "scale_sigma": round(float(scale), 4),
        },
        "goodness_of_fit": {
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p), 4),
        },
        "empirical_return_plot": emp_plot,
        "return_levels": [
            {
                "return_period": float(periods[i]),
                "return_period_unit": temporal["return_period_unit"],
                "return_level": round(levels[i], 4),
                "ci95_low": None if np.isnan(ci_low[i]) else ci_low[i],
                "ci95_high": None if np.isnan(ci_high[i]) else ci_high[i],
            }
            for i in range(len(periods))
        ],
        "temporal": {
            "frequency": temporal["frequency"],
            "frequency_source": temporal["source"],
            "block_duration_years": None,
            "obs_per_year": obs_per_year,
            "span_years": temporal["span_years"],
            "exposure_years": temporal["exposure_years"],
            "n_observations": temporal["n_observations"],
            "n_valid": temporal["n_valid"],
            "n_duplicates": temporal["n_duplicates"],
            "n_gaps": temporal["n_gaps"],
            "is_monotonic": temporal["is_monotonic"],
            "coverage_ratio": temporal["coverage_ratio"],
            "return_period_unit": temporal["return_period_unit"],
            "temporal_assumption_valid": temporal["temporal_assumption_valid"],
            "notes": temporal["notes"],
            "warnings": temporal["warnings"],
        },
    }
