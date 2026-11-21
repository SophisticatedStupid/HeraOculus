"""Vectorized gridded Mann-Kendall + Spatial False Discovery Rate (BH/BY/Holm/Bonferroni).

Implements Wilks (2016, BAMS) spatial field significance control over gridded
climate data cubes (T, H, W), rendering crisp RGBA heatmap PNG previews alongside
per-cell trend and corrected q-value matrices.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Literal, Optional

import numpy as np
import warnings
from PIL import Image
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _build_synthetic_climate_cube(
    *,
    center_lat: float,
    center_lon: float,
    rows: int,
    cols: int,
    n_years: int,
    seed: int,
) -> tuple[np.ndarray, list[float], list[float], str]:
    """Create a realistic spatio-temporal anomaly grid (n_years, rows, cols)."""
    rng = np.random.default_rng(seed)
    lat_span = 14.0
    lon_span = 22.0
    lats = np.linspace(max(-88.0, center_lat - lat_span / 2), min(88.0, center_lat + lat_span / 2), rows)
    lons = np.linspace(center_lon - lon_span / 2, center_lon + lon_span / 2, cols)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    t = np.arange(n_years, dtype=float)
    t_norm = t - np.mean(t)

    # Polar/regional warming dipole + realistic spatial AR noise
    amp_boost = 1.6 if abs(center_lat) > 55 else 1.0
    core1 = np.exp(-(((lat_grid - (center_lat + 1.5)) / 4.2) ** 2 + ((lon_grid - (center_lon - 2.5)) / 5.5) ** 2))
    core2 = 0.72 * np.exp(-(((lat_grid - (center_lat - 2.2)) / 3.8) ** 2 + ((lon_grid - (center_lon + 4.0)) / 5.2) ** 2))
    cooling_pocket = -0.32 * np.exp(-(((lat_grid - (center_lat - 3.5)) / 2.6) ** 2 + ((lon_grid - (center_lon - 4.5)) / 3.2) ** 2))

    slope_per_year = amp_boost * (0.065 * core1 + 0.052 * core2 + 0.03 * cooling_pocket)

    # Correlated spatial noise at each time step (3x3 box smoothing in space)
    noise = rng.normal(0.0, 0.34, size=(n_years, rows, cols))
    for k in range(n_years):
        padded = np.pad(noise[k], 1, mode="edge")
        noise[k] = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + 2.0 * padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 10.0 * 1.45

    # Baseline climatology gradient + trend + noise
    baseline = 12.0 - 0.45 * np.abs(lat_grid)
    cube = baseline[None, :, :] + t_norm[:, None, None] * slope_per_year[None, :, :] + noise
    return (
        cube,
        [round(float(x), 3) for x in lats],
        [round(float(x), 3) for x in lons],
        "synthetic_demo",
    )


def _tie_correction_sum(sorted_vals: np.ndarray) -> np.ndarray:
    """Compute sum over tied-value groups of t*(t-1)*(2t+5) for each spatial cell.

    Runs a stable sort along the time axis and detects equal-value run lengths.
    Memory stays O(T * N); no T x T x H x W arrays are allocated.
    """
    T, N = sorted_vals.shape
    run_len = np.zeros(N, dtype=float)
    prev = np.full(N, np.nan, dtype=float)
    tie_sum = np.zeros(N, dtype=float)

    for k in range(T):
        cur = sorted_vals[k]
        cont = np.isfinite(cur) & np.isfinite(prev) & (cur == prev)
        ended = np.isfinite(prev) & (~cont)
        if np.any(ended):
            idx = np.nonzero(ended)[0]
            lens = run_len[idx]
            tie_sum[idx] = tie_sum[idx] + lens * (lens - 1.0) * (2.0 * lens + 5.0)
            run_len[idx] = 0.0
        run_len = np.where(cont, run_len + 1.0, np.where(np.isfinite(cur), 1.0, 0.0))
        prev = cur

    trailing = run_len > 0
    if np.any(trailing):
        idx = np.nonzero(trailing)[0]
        lens = run_len[idx]
        tie_sum[idx] = tie_sum[idx] + lens * (lens - 1.0) * (2.0 * lens + 5.0)

    return tie_sum


def _vectorized_mk_and_slope(
    cube: np.ndarray,
    time_years: Optional[np.ndarray] = None,
    autocorr_correction: Literal["none", "hamed_rao"] = "none",
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict
]:
    """Compute vectorized per-cell Mann-Kendall Z, two-sided p-value, OLS slope/decade, n_valid,
    and temporal autocorrelation diagnostics.

    Uses tie-corrected variance; only valid observations count for ties.
    Slope is calculated on the supplied time_years (decimal years).

    Autocorrelation handling:
      - "none": no temporal autocorrelation correction (MK independence assumption).
      - "hamed_rao": Hamed & Rao (1998) variance modification using detrended
        rank autocorrelation at consecutive-year lags.

    Returns tuple (Z, p_vals, slope_decade, valid_mask, n_valid, autocorr_diagnostics).
    autocorr_diagnostics contains per-cell "lag1_rho" (float array) and
    "variance_correction" (float array; =1.0 when "none").
    """
    T, H, W = cube.shape
    flat = cube.reshape(T, H * W)  # (T, N)
    N = flat.shape[1]

    n_valid = np.sum(np.isfinite(flat), axis=0).astype(float)

    # Pairwise sign differences; NaN pairs are excluded (memory-safe)
    processed_S = np.zeros(N, dtype=float)
    for i in range(T - 1):
        diff = flat[i + 1 :, :] - flat[i : i + 1, :]
        valid_pairs = np.isfinite(diff)
        processed_S += np.sum(np.where(valid_pairs, np.sign(diff), 0.0), axis=0)

    tie_sum = _tie_correction_sum(np.sort(flat, axis=0))
    n = n_valid
    base_var = n * (n - 1.0) * (2.0 * n + 5.0)
    var_s = np.maximum((base_var - tie_sum) / 18.0, 1e-9)

    # Estimate lag-1 autocorrelation per cell (consecutive time steps only)
    lag1_rho = _estimate_lag1_rho(flat)

    # Hamed & Rao (1998) variance correction factor (per cell)
    variance_correction = np.ones(N, dtype=float)
    if autocorr_correction == "hamed_rao":
        variance_correction = _hamed_rao_correction_factor(flat, time_years, n_valid)
        var_s = var_s * variance_correction

    Z = np.zeros(N, dtype=float)
    pos = processed_S > 0
    neg = processed_S < 0
    Z[pos] = (processed_S[pos] - 1.0) / np.sqrt(var_s[pos])
    Z[neg] = (processed_S[neg] + 1.0) / np.sqrt(var_s[neg])
    Z[n_valid < 5] = 0.0

    p_vals = 2.0 * stats.norm.sf(np.abs(Z))
    p_vals = np.where(n_valid >= 5, np.clip(p_vals, 1e-15, 1.0), 1.0)

    # Temporal OLS slope on the supplied time coordinate; slope is per year
    if time_years is None:
        time_years = np.arange(T, dtype=float)
    t = np.asarray(time_years, dtype=float).reshape(-1, 1)
    mask = np.isfinite(flat)
    t_masked = np.where(mask, t, np.nan)
    valid_count = n_valid > 0
    # Suppress benign "mean of empty slice" for fully-invalid columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        t_mean = np.where(valid_count, np.nanmean(t_masked, axis=0, keepdims=True), 0.0)
        y_mean = np.where(valid_count, np.nanmean(flat, axis=0, keepdims=True), 0.0)
    num = np.nansum((t_masked - t_mean) * (flat - y_mean), axis=0)
    den = np.nansum((t_masked - t_mean) ** 2, axis=0)
    slope_per_year = np.where((den > 1e-12) & (n_valid >= 5), num / np.maximum(den, 1e-12), 0.0)
    slope_decade = slope_per_year * 10.0

    autocorr_diagnostics = {
        "lag1_rho": lag1_rho.reshape(H, W),
        "variance_correction": variance_correction.reshape(H, W),
        "autocorr_method": autocorr_correction,
    }

    return (
        Z.reshape(H, W),
        p_vals.reshape(H, W),
        slope_decade.reshape(H, W),
        (n_valid >= 5).reshape(H, W),
        n_valid.reshape(H, W),
        autocorr_diagnostics,
    )


def _estimate_lag1_rho(flat: np.ndarray) -> np.ndarray:
    """Estimate per-cell lag-1 autocorrelation using consecutive time steps only.

    NaN/missing observations are excluded; a lag-1 pair contributes only when both
    endpoints are valid.  Cells with insufficient valid pairs or zero variance return 0.
    Missing temporal sampling is handled explicitly: non-consecutive valid pairs are
    never treated as lag-1.
    """
    T, N = flat.shape
    rho = np.zeros(N, dtype=float)

    if T < 3:
        return rho

    # Pairwise products of consecutive deviations (population mean centering per cell)
    y = flat
    valid_mask = np.isfinite(y)

    # Compute cell means over valid observations only
    n_valid = np.sum(valid_mask, axis=0)
    y_mean = np.zeros(N, dtype=float)
    np.divide(
        np.nansum(np.where(valid_mask, y, 0.0), axis=0),
        np.maximum(n_valid, 1),
        out=y_mean,
        where=n_valid > 0,
    )

    numerator = np.zeros(N, dtype=float)
    denominator = np.zeros(N, dtype=float)
    pair_count = np.zeros(N, dtype=float)

    for i in range(T - 1):
        a = y[i]
        b = y[i + 1]
        pair_valid = valid_mask[i] & valid_mask[i + 1]
        dev_a = np.where(pair_valid, a - y_mean, 0.0)
        dev_b = np.where(pair_valid, b - y_mean, 0.0)
        numerator += np.where(pair_valid, dev_a * dev_b, 0.0)
        denominator += np.where(pair_valid, dev_a * dev_a, 0.0)
        pair_count += pair_valid.astype(float)

    sufficient = (pair_count >= 2) & (denominator > 1e-12)
    rho[sufficient] = numerator[sufficient] / denominator[sufficient]
    # Clamp to [-1, 1]
    rho = np.clip(rho, -1.0, 1.0)
    return rho


def _hamed_rao_correction_factor(
    flat: np.ndarray,
    time_years: Optional[np.ndarray],
    n_valid: np.ndarray,
) -> np.ndarray:
    """Hamed & Rao (1998) variance modification factor for Mann-Kendall.

    Computes centered detrended rank autocorrelation at consecutive-year lags and
    returns n/n* variance inflation factor (bounded [0.5, 6.0]).
    Cells with n_valid < 6 or zero trend/variance return 1.0 (no correction).

    Missing data and irregular time spacing are handled explicitly: only valid
    observations with matching year lags contribute; gaps are never treated as
    consecutive.
    """
    T, N = flat.shape
    correction = np.ones(N, dtype=float)

    if T < 6:
        return correction

    if time_years is None:
        time_years = np.arange(T, dtype=float)
    time_years = np.asarray(time_years, dtype=float)

    for cell_idx in range(N):
        col = flat[:, cell_idx]
        valid = np.isfinite(col)
        n_cell = int(valid.sum())
        if n_cell < 6:
            continue

        # Detrend using Sen slope on decimal-year time coordinate
        t_vals = time_years[valid]
        y_vals = col[valid]
        slope = _sen_slope_1d(y_vals, t_vals)
        detrended = y_vals - slope * t_vals

        # Rank the detrended values (average ranks for ties)
        ranks = _rank_with_ties(detrended)
        mean_rank = ranks.mean()
        centered = ranks - mean_rank
        denom = float(np.sum(centered * centered))
        if denom <= 1e-12:
            continue

        # Map each valid observation's original year index to position
        # year positions are 0..T-1 based on time index
        year_positions = np.where(valid)[0]

        max_lag = min(12, max(1, n_cell // 4))
        crit = 1.96 / np.sqrt(n_cell)
        correction_sum = 0.0

        for lag in range(1, max_lag + 1):
            num = 0.0
            pair_count = 0
            for k in range(len(year_positions)):
                pos_k = year_positions[k]
                target_year = pos_k + lag
                j = np.searchsorted(year_positions, target_year)
                if j < len(year_positions) and year_positions[j] == target_year:
                    num += centered[k] * centered[j]
                    pair_count += 1
            if pair_count == 0:
                continue
            rho = num / denom
            if abs(rho) > crit:
                m = n_cell
                correction_sum += (
                    (m - lag) * (m - lag - 1) * (m - lag - 2) * rho
                )

        if abs(correction_sum) > 0:
            n_over_nstar = (
                1 + (2.0 / (n_cell * (n_cell - 1) * (n_cell - 2))) * correction_sum
            )
            correction[cell_idx] = float(np.clip(n_over_nstar, 0.5, 6.0))

    return correction


def _sen_slope_1d(values: np.ndarray, times: np.ndarray) -> float:
    """Sen's slope (median of pairwise slopes) for a 1D series."""
    n = values.size
    if n < 2:
        return 0.0
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dt = times[j] - times[i]
            if abs(dt) > 1e-12:
                slopes.append((values[j] - values[i]) / dt)
    if not slopes:
        return 0.0
    slopes.sort()
    m = len(slopes)
    mid = m // 2
    return slopes[mid] if m % 2 == 1 else (slopes[mid - 1] + slopes[mid]) / 2.0


def _rank_with_ties(values: np.ndarray) -> np.ndarray:
    """Rank values with average ranks for ties."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _render_spatial_png(
    slope_grid: np.ndarray,
    sig_mask: np.ndarray,
    mode: Literal["diverging", "fdr_mask"],
) -> str:
    """Render clean north-up HeraOculus spatial PNG encoded as base64 data URI."""
    H, W = slope_grid.shape
    vmax = max(float(np.percentile(np.abs(slope_grid), 98)), 0.1)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # Flip vertically so row 0 corresponds to northernmost latitude at top of image
    s_flip = np.flipud(slope_grid)
    m_flip = np.flipud(sig_mask)

    norm = np.clip(s_flip / vmax, -1.0, 1.0)
    if mode == "diverging":
        # Cold cyan/blue (-1) -> slate (0) -> fiery orange/red (+1)
        r = np.where(norm >= 0, 30 + 225 * norm, 30 * (1 + norm)).astype(np.uint8)
        g = np.where(norm >= 0, 110 + 70 * (1 - norm), 130 + 95 * (-norm)).astype(np.uint8)
        b = np.where(norm >= 0, 145 * (1 - norm), 170 + 85 * (-norm)).astype(np.uint8)
        rgba[:, :, 0] = r
        rgba[:, :, 1] = g
        rgba[:, :, 2] = b
        rgba[:, :, 3] = 240
    else:
        # Highlight only FDR-significant pixels strongly, mute non-significant cells
        r = np.where(norm >= 0, 244, 56).astype(np.uint8)
        g = np.where(norm >= 0, 92, 189).astype(np.uint8)
        b = np.where(norm >= 0, 66, 248).astype(np.uint8)
        rgba[:, :, 0] = np.where(m_flip, r, 18)
        rgba[:, :, 1] = np.where(m_flip, g, 32)
        rgba[:, :, 2] = np.where(m_flip, b, 56)
        rgba[:, :, 3] = np.where(m_flip, 250, 175)

    img = Image.fromarray(rgba, mode="RGBA").resize((max(360, W * 16), max(280, H * 16)), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_timestamps_to_decimal_years(timestamps: list[str]) -> np.ndarray:
    """Convert ISO-like timestamps ('YYYY', 'YYYY-MM', 'YYYY-MM-DD', optional time) to decimal years."""
    out = []
    for label in timestamps:
        text = str(label).strip().replace("Z", "").split("T")[0].split(" ")[0]
        parts = text.split("-")
        if not parts or not parts[0]:
            raise ValueError(f"Unparseable timestamp: {label!r}")
        try:
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 and parts[1] else 1
            day = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Unparseable timestamp: {label!r}") from exc
        if not (1 <= month <= 12) or not (1 <= day <= 31):
            raise ValueError(f"Out-of-range timestamp: {label!r}")
        year_start = datetime(year, 1, 1)
        year_end = datetime(year + 1, 1, 1)
        frac = (datetime(year, month, day) - year_start).total_seconds() / (
            year_end - year_start
        ).total_seconds()
        out.append(year + frac)
    return np.asarray(out, dtype=float)


def _classify_resolution(step_years: float) -> tuple[str, str]:
    """Return (resolution_label, units_description) for a (median) time step in years."""
    if not np.isfinite(step_years) or step_years <= 0:
        return "unknown", "years"
    if abs(step_years - 1.0) <= 0.02:
        return "annual", "decimal years (annual steps)"
    if abs(step_years - 1.0 / 12.0) <= 0.01:
        return "monthly", "decimal years (monthly steps)"
    if abs(step_years - 0.25) <= 0.01:
        return "quarterly", "decimal years (quarterly steps)"
    if abs(step_years - 0.5) <= 0.01:
        return "semi-annual", "decimal years (semi-annual steps)"
    return "irregular/custom", f"decimal years (median step {step_years:.5f} yrs)"


def _resolve_time_axis(
    n_steps: int,
    *,
    timestamps: Optional[list[str]],
    time_step_years: Optional[float],
    is_synthetic: bool,
) -> tuple[np.ndarray, dict]:
    """Resolve an explicit decimal-year time coordinate and its resolution metadata.

    Priority: explicit timestamps > explicit uniform step > synthetic-annual default.
    Uploaded cubes (non-synthetic) without either temporal spec are rejected rather than
    silently assumed annual.
    """
    if timestamps is not None:
        t = _parse_timestamps_to_decimal_years(timestamps)
        if t.size != n_steps:
            raise ValueError(
                f"timestamps length ({t.size}) must equal the number of time steps ({n_steps})."
            )
        if not np.all(np.isfinite(t)):
            raise ValueError("timestamps contain non-finite values.")
        steps = np.diff(t)
        if np.any(steps <= 0):
            raise ValueError("timestamps must be strictly increasing (ambiguous/duplicate spacing).")
        median_step = float(np.median(steps))
        label, units = _classify_resolution(median_step)
        return t, {
            "source": "timestamps",
            "resolution": label,
            "units": units,
            "median_step_years": median_step,
            "is_uniform": bool(np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9)),
        }

    if time_step_years is not None:
        step = float(time_step_years)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("time_step_years must be a positive finite number of years.")
        t = np.arange(n_steps, dtype=float) * step
        label, units = _classify_resolution(step)
        return t, {
            "source": "time_step_years",
            "resolution": label,
            "units": units,
            "median_step_years": step,
            "is_uniform": True,
        }

    if is_synthetic:
        t = np.arange(n_steps, dtype=float)
        return t, {
            "source": "synthetic_annual",
            "resolution": "annual",
            "units": "decimal years (annual steps)",
            "median_step_years": 1.0,
            "is_uniform": True,
        }

    raise ValueError(
        "Ambiguous temporal spacing for uploaded cube: supply `timestamps` or `time_step_years` "
        "so the decadal slope uses a well-defined time coordinate instead of assuming annual steps."
    )


def run_spatial_fdr_analysis(
    *,
    region_name: str = "Arctic & Boreal Sub-Grid",
    center_lat: float = 68.0,
    center_lon: float = 20.0,
    grid_rows: int = 20,
    grid_cols: int = 24,
    n_years: int = 20,
    alpha: float = 0.05,
    correction_method: Literal["bh", "by", "holm", "bonferroni"] = "by",
    temporal_autocorr_correction: Literal["none", "hamed_rao"] = "none",
    cube_input: Optional[list[list[list[Optional[float]]]]] = None,
    seed: int = 42,
    timestamps: Optional[list[str]] = None,
    time_step_years: Optional[float] = None,
) -> dict:
    """Execute vectorized gridded Mann-Kendall + Spatial FDR correction.

    Users may provide explicit timestamps (e.g. ['2001', '2001-01', ...]) that will be
    parsed to decimal years, or an explicit uniform `time_step_years` (e.g. 1.0 for annual,
    1/12 ≈ 0.0833 for monthly).  Either is required for uploaded cubes; supplying either
    ensures the decadal slope uses a well-defined time coordinate instead of silently
    assuming annual spacing.

    Existing synthetic-annual benchmarks require no extra temporal metadata.

    Temporal autocorrelation correction (applied per pixel, independent of spatial FDR):
      - "none": no correction; standard MK assumes temporal independence.
      - "hamed_rao": Hamed & Rao (1998) variance modification using detrended
        rank autocorrelation at consecutive-year lags. Missing/invalid data and
        irregular sampling are handled explicitly (gaps never treated as consecutive).

    Spatial FDR methods (BH, BY, Holm, Bonferroni) correct for *multiple testing*
    dependence across pixels. They do NOT correct temporal autocorrelation within
    each pixel's time series. Use `temporal_autocorr_correction` for that.
    """
    if cube_input is not None:
        cube = np.array(
            [[[np.nan if v is None else float(v) for v in row] for row in t_slice] for t_slice in cube_input],
            dtype=float,
        )
        if cube.ndim != 3 or cube.shape[0] < 6:
            raise ValueError("Uploaded cube must be 3D (T, H, W) with T >= 6 time steps.")
        H, W = cube.shape[1], cube.shape[2]
        lats = [round(center_lat - 5.0 + 10.0 * i / max(1, H - 1), 3) for i in range(H)]
        lons = [round(center_lon - 8.0 + 16.0 * j / max(1, W - 1), 3) for j in range(W)]
        provenance_tier = "user_upload"
    else:
        cube, lats, lons, provenance_tier = _build_synthetic_climate_cube(
            center_lat=center_lat,
            center_lon=center_lon,
            rows=grid_rows,
            cols=grid_cols,
            n_years=n_years,
            seed=seed,
        )

    time_metadata = _resolve_time_axis(
        n_steps=cube.shape[0],
        timestamps=timestamps,
        time_step_years=time_step_years,
        is_synthetic=provenance_tier != "user_upload",
    )
    Z, p_vals, slope_decade, valid_mask, n_valid, autocorr_diag = _vectorized_mk_and_slope(
        cube, time_years=time_metadata[0], autocorr_correction=temporal_autocorr_correction
    )
    valid_p = p_vals[valid_mask]
    raw_sig = (p_vals <= alpha) & valid_mask

    method_map = {
        "bh": "fdr_bh",
        "by": "fdr_by",
        "holm": "holm",
        "bonferroni": "bonferroni",
    }
    if len(valid_p) == 0:
        raise ValueError("No valid cells found in spatio-temporal cube.")

    rejected_flat, q_flat, _, _ = multipletests(valid_p, alpha=alpha, method=method_map[correction_method])

    q_vals = np.ones_like(p_vals)
    q_vals[valid_mask] = q_flat
    fdr_sig = np.zeros_like(valid_mask, dtype=bool)
    fdr_sig[valid_mask] = rejected_flat

    n_tests = int(np.sum(valid_mask))
    n_raw_sig = int(np.sum(raw_sig))
    n_fdr_sig = int(np.sum(fdr_sig))
    false_positives_pruned = max(0, n_raw_sig - n_fdr_sig)
    area_fraction_sig = round(100.0 * n_fdr_sig / max(1, n_tests), 2)

    # Autocorrelation diagnostics summary
    lag1_rho = autocorr_diag["lag1_rho"]
    var_corr = autocorr_diag["variance_correction"]
    n_mean_rho = float(np.mean(np.abs(lag1_rho[valid_mask]))) if n_tests > 0 else 0.0
    n_mean_varcorr = float(np.mean(var_corr[valid_mask])) if n_tests > 0 else 0.0
    n_excluded = int(cube.shape[1] * cube.shape[2] - n_tests)

    # Wilks (2016) global field significance threshold (p_(1) vs alpha_global)
    p_sorted = np.sort(valid_p)
    alpha_wilks = alpha / (1.0 + sum(1.0 / k for k in range(2, n_tests + 1))) if correction_method == "by" else alpha
    field_significant = n_fdr_sig > 0

    return {
        "region_name": region_name,
        "provenance": {
            "tier": provenance_tier,
            "source_name": "Uploaded Spatial Cube" if provenance_tier == "user_upload" else "HeraOculus Gridded Climate Benchmark Field",
            "dataset_id": "SPATIAL_GRID_MK_FDR",
            "is_live": False,
            "status_message": (
                "Vectorized Mann-Kendall & Spatial FDR evaluated over uploaded 3D cube."
                if provenance_tier == "user_upload"
                else "Synthetic demonstration spatio-temporal climate grid (20x24 cells) illustrating Wilks (2016) Spatial FDR."
            ),
        },
        "dimensions": {"time_steps": int(cube.shape[0]), "rows": int(cube.shape[1]), "cols": int(cube.shape[2])},
        "method": correction_method,
        "alpha": alpha,
        "summary": {
            "total_pixels": int(cube.shape[1] * cube.shape[2]),
            "total_valid_pixels": n_tests,
            "excluded_pixels": n_excluded,
            "tested_hypotheses": n_tests,
            "raw_significant_cells": n_raw_sig,
            "fdr_significant_cells": n_fdr_sig,
            "false_positives_pruned": false_positives_pruned,
            "pct_area_raw_sig": round(100.0 * n_raw_sig / max(1, n_tests), 2),
            "pct_area_fdr_sig": area_fraction_sig,
            "field_significant": field_significant,
            "min_raw_p": float(f"{p_sorted[0]:.3e}"),
            "min_adjusted_q": float(f"{np.min(q_vals[valid_mask]):.3e}"),
            "max_positive_slope_decade": round(float(np.max(slope_decade)), 4),
            "min_negative_slope_decade": round(float(np.min(slope_decade)), 4),
            "wilks_effective_alpha": float(f"{alpha_wilks:.3e}"),
        },
        "temporal": {
            "n_time_steps": int(cube.shape[0]),
            "time_step_years": float(np.median(np.diff(time_metadata[0]))) if len(time_metadata[0]) > 1 else float(time_metadata[0][0]),
            "source": time_metadata[1]["source"],
            "resolution": time_metadata[1]["resolution"],
        },
        "autocorrelation": {
            "temporal_correction": temporal_autocorr_correction,
            "method": "hamed_rao_1998" if temporal_autocorr_correction == "hamed_rao" else "none",
            "mean_abs_lag1_rho": round(n_mean_rho, 6),
            "mean_variance_correction": round(n_mean_varcorr, 6),
            "spatial_fdr_method": correction_method,
            "note": (
                "Spatial FDR (BH/BY/Holm/Bonferroni) corrects multiple testing across pixels; "
                "temporal autocorrelation correction is applied per-pixel before FDR."
                if temporal_autocorr_correction != "none"
                else "No temporal autocorrelation correction applied; pixel p-values assume temporal independence."
            ),
        },
        "images": {
            "trend_png": _render_spatial_png(slope_decade, raw_sig, "diverging"),
            "fdr_png": _render_spatial_png(slope_decade, fdr_sig, "fdr_mask"),
        },
        "grid": {
            "latitudes": lats,
            "longitudes": lons,
            "slope_decade": [[round(float(v), 4) for v in row] for row in slope_decade],
            "p_values": [[round(float(v), 6) for v in row] for row in p_vals],
            "q_values": [[round(float(v), 6) for v in row] for row in q_vals],
            "raw_significant": [[bool(v) for v in row] for row in raw_sig],
            "fdr_significant": [[bool(v) for v in row] for row in fdr_sig],
        },
    }
