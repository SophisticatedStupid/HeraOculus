"""Ordinary Kriging interpolation.

Estimates missing continuous values using spatial autocorrelation modeled by
an empirical variogram.  Produces both predictions and kriging variance.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from .base import InterpolationResult, _ensure_float_array, _valid_mask


def _empirical_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15,
    max_lag: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical variogram (method of moments).

    gamma(h) = 0.5 * mean((z(x_i) - z(x_j))^2) for pairs at distance h.

    Returns (lag_distances, semivariance_values).
    """
    n = coords.shape[0]
    if n < 3:
        return np.array([0.0]), np.array([0.0])

    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]  # (n, n, 2)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)

    # Pairwise squared differences
    diff_z = values[:, None] - values[None, :]  # (n, n)
    sq_diff = 0.5 * diff_z * diff_z  # (n, n)

    # Upper triangle only (exclude diagonal)
    iu = np.triu_indices(n, k=1)
    pair_dist = dist[iu]
    pair_sq = sq_diff[iu]

    if max_lag is None:
        max_lag = float(np.percentile(pair_dist, 90))

    # Bin distances into lags
    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2.0
    gamma = np.zeros(n_lags)
    counts = np.zeros(n_lags, dtype=int)

    for k in range(n_lags):
        mask = (pair_dist >= lag_edges[k]) & (pair_dist < lag_edges[k + 1])
        if np.any(mask):
            gamma[k] = float(np.mean(pair_sq[mask]))
            counts[k] = int(np.sum(mask))
        else:
            gamma[k] = np.nan

    # Remove empty lags
    valid_lags = counts > 0
    return lag_centers[valid_lags], gamma[valid_lags]


def _variogram_model_spherical(h: float, nugget: float, sill: float, range_: float) -> float:
    """Spherical variogram model."""
    if h == 0:
        return 0.0
    if h >= range_:
        return nugget + sill
    return nugget + sill * (1.5 * (h / range_) - 0.5 * (h / range_) ** 3)


def _variogram_model_exponential(h: float, nugget: float, sill: float, range_: float) -> float:
    """Exponential variogram model."""
    if h == 0:
        return 0.0
    return nugget + sill * (1.0 - math.exp(-h / range_))


def _variogram_model_gaussian(h: float, nugget: float, sill: float, range_: float) -> float:
    """Gaussian variogram model."""
    if h == 0:
        return 0.0
    return nugget + sill * (1.0 - math.exp(-(h / range_) ** 2))


def _variogram_model_linear(h: float, nugget: float, sill: float, range_: float) -> float:
    """Linear variogram model (bounded by sill at range)."""
    if h == 0:
        return 0.0
    return nugget + sill * min(h / range_, 1.0)


_VARIOGRAM_MODELS = {
    "spherical": _variogram_model_spherical,
    "exponential": _variogram_model_exponential,
    "gaussian": _variogram_model_gaussian,
    "linear": _variogram_model_linear,
}


def _fit_variogram(
    lag_distances: np.ndarray,
    semivariances: np.ndarray,
    model_name: str = "spherical",
    nugget: Optional[float] = None,
    sill: Optional[float] = None,
    range_: Optional[float] = None,
) -> dict:
    """Fit a variogram model to empirical semivariances.

    Uses simple least-squares with optional user-provided parameters.
    Returns dict with model parameters and fitted values.
    """
    if model_name not in _VARIOGRAM_MODELS:
        raise ValueError(f"Unknown variogram model: {model_name}")

    model_fn = _VARIOGRAM_MODELS[model_name]

    # Initial parameter guesses
    if len(semivariances) == 0:
        raise ValueError("No empirical variogram points available for fitting.")

    max_gamma = float(np.nanmax(semivariances))
    min_gamma = float(np.nanmin(semivariances))
    max_lag = float(np.nanmax(lag_distances))

    # Default nugget = small fraction of max, sill = max - nugget, range = max_lag / 2
    init_nugget = nugget if nugget is not None else max(0.0, max_gamma * 0.05)
    init_sill = sill if sill is not None else max(max_gamma - init_nugget, 1e-9)
    init_range = range_ if range_ is not None else max(max_lag / 2.0, 1e-6)

    def objective(params):
        n, s, r = params
        if r <= 1e-9 or s < 0 or n < 0:
            return 1e12
        fitted = np.array([model_fn(h, n, s, r) for h in lag_distances])
        valid = ~np.isnan(semivariances)
        if not np.any(valid):
            return 1e12
        return float(np.sum((fitted[valid] - semivariances[valid]) ** 2))

    # Simple grid-based optimization (avoid scipy dependency complexity)
    best_params = (init_nugget, init_sill, init_range)
    best_obj = objective(best_params)

    # Coordinate descent refinement
    for _ in range(30):
        improved = False
        for param_idx in range(3):
            for direction in [-1, 1]:
                test_params = list(best_params)
                step = [best_params[0] * 0.1, best_params[1] * 0.1, best_params[2] * 0.1]
                if step[param_idx] <= 0:
                    step[param_idx] = 1e-6
                test_params[param_idx] += direction * step[param_idx]
                if test_params[2] <= 1e-9:  # range must be positive
                    continue
                obj = objective(tuple(test_params))
                if obj < best_obj:
                    best_obj = obj
                    best_params = tuple(test_params)
                    improved = True
        if not improved:
            break

    nugget_fit, sill_fit, range_fit = best_params
    fitted_gamma = np.array([model_fn(h, nugget_fit, sill_fit, range_fit) for h in lag_distances])

    # Compute R-squared of fit
    valid = ~np.isnan(semivariances)
    if np.any(valid) and np.std(semivariances[valid]) > 1e-12:
        ss_res = float(np.sum((fitted_gamma[valid] - semivariances[valid]) ** 2))
        ss_tot = float(np.sum((semivariances[valid] - np.mean(semivariances[valid])) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        r_squared = 0.0

    return {
        "model": model_name,
        "nugget": round(float(nugget_fit), 6),
        "sill": round(float(sill_fit), 6),
        "range": round(float(range_fit), 6),
        "r_squared": round(float(r_squared), 6),
        "lag_distances": lag_distances.tolist(),
        "empirical_semivariance": semivariances.tolist(),
        "fitted_semivariance": fitted_gamma.tolist(),
    }


def _ordinary_kriging_system(
    coords: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    variogram_params: dict,
    model_name: str = "spherical",
    max_neighbors: int = 12,
) -> tuple[float, float]:
    """Solve ordinary kriging system for a single target point.

    Returns (prediction, kriging_variance).
    """
    n = coords.shape[0]
    if n == 0:
        return float("nan"), float("nan")

    model_fn = _VARIOGRAM_MODELS[model_name]
    nugget = variogram_params["nugget"]
    sill = variogram_params["sill"]
    range_ = variogram_params["range"]

    # Select nearest neighbors
    diff = coords - target[None, :]  # (n, 2)
    dist = np.sqrt(np.sum(diff * diff, axis=1))  # (n,)
    order = np.argsort(dist)
    n_use = min(max_neighbors, n)
    sel_idx = order[:n_use]
    sel_coords = coords[sel_idx]
    sel_vals = values[sel_idx]
    sel_dist = dist[sel_idx]

    # Build variogram matrix between neighbors (n_use x n_use)
    # Gamma(i,j) = variogram(distance between obs i and obs j)
    n_use = sel_coords.shape[0]
    diff_mat = sel_coords[:, None, :] - sel_coords[None, :, :]  # (n_use, n_use, 2)
    dist_mat = np.sqrt(np.sum(diff_mat * diff_mat, axis=2))  # (n_use, n_use)
    gamma_mat = np.zeros((n_use, n_use))
    for i in range(n_use):
        for j in range(n_use):
            gamma_mat[i, j] = model_fn(float(dist_mat[i, j]), nugget, sill, range_)

    # Ordinary kriging system: [gamma 1; 1^T 0] * [w; mu] = [gamma_target; 1]
    A = np.zeros((n_use + 1, n_use + 1))
    A[:n_use, :n_use] = gamma_mat
    A[:n_use, n_use] = 1.0
    A[n_use, :n_use] = 1.0
    A[n_use, n_use] = 0.0

    # Variogram between target and each neighbor
    gamma_target = np.array([model_fn(float(d), nugget, sill, range_) for d in sel_dist])
    b = np.zeros(n_use + 1)
    b[:n_use] = gamma_target
    b[n_use] = 1.0

    # Solve system
    try:
        solution = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Singular system — add small nugget regularization
        A_reg = A + np.eye(n_use + 1) * 1e-10
        try:
            solution = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")

    weights = solution[:n_use]
    mu = solution[n_use]

    # Prediction
    prediction = float(np.sum(weights * sel_vals))

    # Kriging variance: sigma^2 = sum(w_i * gamma_i_target) + mu
    kriging_var = float(np.sum(weights * gamma_target) + mu)
    kriging_var = max(kriging_var, 0.0)  # variance cannot be negative

    return prediction, kriging_var


def interpolate_ordinary_kriging(
    array: np.ndarray,
    *,
    variogram_model: str = "spherical",
    variogram_nugget: Optional[float] = None,
    variogram_sill: Optional[float] = None,
    variogram_range: Optional[float] = None,
    search_radius_pixels: Optional[float] = None,
    max_neighbors: int = 12,
    min_neighbors: int = 3,
    max_sample_for_fit: int = 500,
    band_index: int = 1,
) -> InterpolationResult:
    """Ordinary kriging interpolation for continuous rasters.

    Estimates missing values using spatial autocorrelation modeled by an
    empirical variogram. Produces both predictions and kriging variance.

    Args:
        array: 2D float64 array, NaN = missing.
        variogram_model: 'spherical', 'exponential', 'gaussian', 'linear'.
        variogram_nugget/sill/range: Optional user-defined variogram parameters.
        search_radius_pixels: Max search radius for neighbor selection.
        max_neighbors: Maximum neighbors per prediction.
        min_neighbors: Minimum neighbors required (else NoData).
        max_sample_for_fit: Max observations used for variogram fitting.
        band_index: 1-based band number.

    Returns:
        InterpolationResult with kriging variance in quality_mask.
    """
    start = time.perf_counter()
    arr = _ensure_float_array(array)
    valid = _valid_mask(arr)
    missing = ~valid

    H, W = arr.shape
    reconstructed = arr.copy()
    provenance = np.full((H, W), "nodata", dtype=object)
    provenance[valid] = "observed"
    kriging_variance = np.full((H, W), np.nan, dtype=float)

    valid_coords = np.argwhere(valid)  # (N, 2)
    missing_coords = np.argwhere(missing)  # (M, 2)
    valid_vals = arr[valid]

    interpolated_count = 0
    excluded_count = 0

    # Fit variogram from valid observations (sample if too many)
    fit_coords = valid_coords
    fit_vals = valid_vals
    if valid_coords.shape[0] > max_sample_for_fit:
        # Stratified random sample for variogram fitting
        rng = np.random.default_rng(42)
        sel = rng.choice(valid_coords.shape[0], size=max_sample_for_fit, replace=False)
        fit_coords = valid_coords[sel]
        fit_vals = valid_vals[sel]

    if fit_coords.shape[0] < min_neighbors:
        # Not enough observations to fit variogram or interpolate
        if missing_coords.size > 0:
            excluded_count = int(missing_coords.shape[0])
            provenance[missing] = "excluded"
        observed_count = int(np.sum(valid))
        remaining_nodata = int(np.sum(missing))
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return InterpolationResult(
            reconstructed=reconstructed,
            provenance_mask=provenance,
            quality_mask=kriging_variance,
            method="ordinary_kriging",
            band_index=band_index,
            raster_type="continuous",
            config={
                "variogram_model": variogram_model,
                "error": "insufficient_observations_for_variogram",
            },
            observed_pixel_count=observed_count,
            interpolated_pixel_count=0,
            remaining_nodata_count=remaining_nodata,
            excluded_pixel_count=excluded_count,
            processing_duration_ms=duration_ms,
        )

    # Compute empirical variogram
    max_lag = None
    if search_radius_pixels is not None:
        max_lag = search_radius_pixels * 1.5  # variogram range extends beyond search

    lag_dist, semivar = _empirical_variogram(fit_coords, fit_vals, n_lags=15, max_lag=max_lag)

    # Fit variogram model
    try:
        vario_fit = _fit_variogram(
            lag_dist, semivar,
            model_name=variogram_model,
            nugget=variogram_nugget,
            sill=variogram_sill,
            range_=variogram_range,
        )
    except ValueError:
        vario_fit = {
            "model": variogram_model,
            "nugget": 0.0,
            "sill": float(np.nanmax(semivar)) if len(semivar) > 0 else 1.0,
            "range": float(np.nanmax(lag_dist)) if len(lag_dist) > 0 else 1.0,
            "r_squared": 0.0,
        }

    vario_params = {
        "nugget": vario_fit["nugget"],
        "sill": vario_fit["sill"],
        "range": vario_fit["range"],
    }

    # Krige each missing pixel
    if missing_coords.size > 0:
        for r, c in missing_coords:
            target = np.array([float(r), float(c)])

            # Select candidate neighbors within search radius
            if search_radius_pixels is not None:
                diff = valid_coords - target[None, :]
                dist = np.sqrt(np.sum(diff * diff, axis=1))
                within = dist <= search_radius_pixels
                if np.sum(within) < min_neighbors:
                    provenance[r, c] = "excluded"
                    excluded_count += 1
                    continue
                cand_coords = valid_coords[within]
                cand_vals = valid_vals[within]
            else:
                cand_coords = valid_coords
                cand_vals = valid_vals

            if cand_coords.shape[0] < min_neighbors:
                provenance[r, c] = "excluded"
                excluded_count += 1
                continue

            pred, var = _ordinary_kriging_system(
                cand_coords, cand_vals, target, vario_params,
                model_name=variogram_model,
                max_neighbors=max_neighbors,
            )

            if np.isnan(pred):
                provenance[r, c] = "excluded"
                excluded_count += 1
            else:
                reconstructed[r, c] = pred
                kriging_variance[r, c] = var
                provenance[r, c] = "interpolated"
                interpolated_count += 1

    observed_count = int(np.sum(valid))
    remaining_nodata = int(np.sum(missing & (provenance != "interpolated")))

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return InterpolationResult(
        reconstructed=reconstructed,
        provenance_mask=provenance,
        quality_mask=kriging_variance,
        method="ordinary_kriging",
        band_index=band_index,
        raster_type="continuous",
        config={
            "variogram_model": variogram_model,
            "variogram_fit": vario_fit,
            "search_radius_pixels": search_radius_pixels,
            "max_neighbors": max_neighbors,
            "min_neighbors": min_neighbors,
        },
        observed_pixel_count=observed_count,
        interpolated_pixel_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_pixel_count=excluded_count,
        processing_duration_ms=duration_ms,
    )
