"""Interpolation validation and quality assessment service.

Performs independent cross-validation of interpolation results:
- Random holdout validation
- Spatial block validation (preferred for spatially autocorrelated data)

Calculates MAE, RMSE, bias, R² (when appropriate), median absolute error,
MAPE (when appropriate), prediction coverage, and sample counts.

Ensures no data leakage: validation pixels are not used as interpolation
inputs during their respective validation run.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from scipy import ndimage

from .raster_interpolation_service import (
    InterpolationResult,
    INTERPOLATION_METHODS,
    run_interpolation,
)
from .raster_inspection_service import RasterInspectionResult


@dataclass
class ValidationResult:
    """Cross-validation quality report for an interpolation method."""
    method: str
    config: dict
    validation_strategy: Literal["random_holdout", "spatial_block"]
    validation_sample_count: int
    n_excluded_from_validation: int

    # Error metrics
    mae: Optional[float]
    rmse: Optional[float]
    bias: Optional[float]  # mean error (predicted - observed)
    median_absolute_error: Optional[float]
    mape: Optional[float]  # only when appropriate
    r_squared: Optional[float]  # only when appropriate

    # Coverage
    reconstruction_percentage: float
    remaining_nodata_percentage: float
    excluded_percentage: float

    # Processing
    processing_duration_ms: float

    # Warnings and limitations
    warnings: list[str]
    limitations: list[str]

    # Per-pixel validation details (for scatter/residual plots)
    validation_predicted: list[float]
    validation_observed: list[float]
    validation_residuals: list[float]
    validation_coords: list[list[int]]


def _spatial_block_indices(
    shape: tuple[int, int],
    n_blocks: int = 4,
) -> list[np.ndarray]:
    """Divide raster into spatial blocks for block validation.

    Returns list of boolean masks, one per block.
    """
    H, W = shape
    blocks = []
    # Split into a grid of n_blocks x n_blocks
    grid_n = int(math.ceil(math.sqrt(n_blocks)))
    block_rows = np.array_split(np.arange(H), grid_n)
    block_cols = np.array_split(np.arange(W), grid_n)

    block_idx = 0
    for rows in block_rows:
        for cols in block_cols:
            mask = np.zeros((H, W), dtype=bool)
            mask[np.ix_(rows, cols)] = True
            blocks.append(mask)
            block_idx += 1
            if block_idx >= n_blocks:
                break
        if block_idx >= n_blocks:
            break
    return blocks


def _select_validation_samples(
    valid_mask: np.ndarray,
    strategy: Literal["random_holdout", "spatial_block"],
    sample_fraction: float = 0.1,
    n_blocks: int = 4,
    block_index: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Select validation sample pixels from valid observations.

    Returns boolean mask of validation pixels (True = held out for validation).
    """
    H, W = valid_mask.shape
    valid_coords = np.argwhere(valid_mask)
    n_valid = len(valid_coords)

    if n_valid == 0:
        return np.zeros((H, W), dtype=bool)

    rng = np.random.default_rng(seed)
    val_mask = np.zeros((H, W), dtype=bool)

    if strategy == "random_holdout":
        n_val = max(1, int(round(n_valid * sample_fraction)))
        n_val = min(n_val, n_valid)
        sel = rng.choice(n_valid, size=n_val, replace=False)
        for idx in sel:
            r, c = valid_coords[idx]
            val_mask[r, c] = True

    elif strategy == "spatial_block":
        blocks = _spatial_block_indices((H, W), n_blocks=n_blocks)
        if block_index is None:
            block_index = rng.integers(0, len(blocks))
        block_index = min(block_index, len(blocks) - 1)
        block_mask = blocks[block_index]
        # Select valid pixels within the chosen block
        val_mask = valid_mask & block_mask

    return val_mask


def _compute_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Compute error metrics between observed and predicted values.

    Returns dict with mae, rmse, bias, median_ae, mape (if appropriate),
    r_squared (if appropriate).
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)

    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]

    n = len(obs)
    if n == 0:
        return {
            "mae": None, "rmse": None, "bias": None,
            "median_absolute_error": None, "mape": None, "r_squared": None,
            "n": 0,
        }

    residuals = pred - obs
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    bias = float(np.mean(residuals))
    median_ae = float(np.median(np.abs(residuals)))

    # MAPE: only when no observed values are zero (division by zero)
    mape = None
    nonzero_mask = np.abs(obs) > 1e-12
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs(residuals[nonzero_mask] / obs[nonzero_mask])) * 100.0)

    # R²: only when there is variance in observed values
    r_squared = None
    obs_var = float(np.var(obs))
    if obs_var > 1e-12 and n > 1:
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
        if ss_tot > 1e-12:
            r_squared = float(1.0 - ss_res / ss_tot)

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "bias": round(bias, 6),
        "median_absolute_error": round(median_ae, 6),
        "mape": round(mape, 6) if mape is not None else None,
        "r_squared": round(r_squared, 6) if r_squared is not None else None,
        "n": n,
    }


def validate_interpolation(
    *,
    array: np.ndarray,
    method: str,
    raster_type: str = "continuous",
    validation_strategy: Literal["random_holdout", "spatial_block"] = "spatial_block",
    sample_fraction: float = 0.1,
    n_blocks: int = 4,
    block_index: Optional[int] = None,
    seed: int = 42,
    max_distance_pixels: Optional[float] = None,
    **kwargs,
) -> ValidationResult:
    """Cross-validate an interpolation method using withheld observations.

    Workflow:
    1. Select validation sample pixels from valid observations.
    2. Temporarily mask those pixels (treat as NoData).
    3. Run interpolation on the masked array.
    4. Compare predicted values at validation pixels against originals.
    5. Compute error metrics.

    No data leakage: validation pixels are excluded from interpolation inputs.

    Args:
        array: 2D float64 array, NaN = missing.
        method: Interpolation method name.
        raster_type: 'continuous' or 'categorical'.
        validation_strategy: 'random_holdout' or 'spatial_block'.
        sample_fraction: Fraction of valid pixels to hold out (random only).
        n_blocks: Number of spatial blocks (spatial_block only).
        block_index: Which block to use as validation (spatial_block only).
        seed: Random seed.
        max_distance_pixels: Max search radius passed to interpolator.
        **kwargs: Additional interpolator parameters.

    Returns:
        ValidationResult with metrics and per-pixel details.
    """
    start = time.perf_counter()

    arr = np.asarray(array, dtype=np.float64)
    valid_mask = np.isfinite(arr)
    missing_mask = ~valid_mask

    H, W = arr.shape
    n_valid = int(np.sum(valid_mask))
    n_missing = int(np.sum(missing_mask))

    warnings: list[str] = []
    limitations: list[str] = []

    if n_valid < 5:
        warnings.append(
            f"Only {n_valid} valid pixels available — insufficient for reliable validation."
        )
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return ValidationResult(
            method=method,
            config=kwargs,
            validation_strategy=validation_strategy,
            validation_sample_count=0,
            n_excluded_from_validation=n_valid,
            mae=None, rmse=None, bias=None, median_absolute_error=None,
            mape=None, r_squared=None,
            reconstruction_percentage=0.0,
            remaining_nodata_percentage=100.0 if n_missing > 0 else 0.0,
            excluded_percentage=0.0,
            processing_duration_ms=duration_ms,
            warnings=warnings,
            limitations=limitations,
            validation_predicted=[],
            validation_observed=[],
            validation_residuals=[],
            validation_coords=[],
        )

    # Select validation samples
    val_mask = _select_validation_samples(
        valid_mask, strategy=validation_strategy,
        sample_fraction=sample_fraction, n_blocks=n_blocks,
        block_index=block_index, seed=seed,
    )
    n_val = int(np.sum(val_mask))

    if n_val == 0:
        warnings.append("No validation pixels selected.")
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return ValidationResult(
            method=method,
            config=kwargs,
            validation_strategy=validation_strategy,
            validation_sample_count=0,
            n_excluded_from_validation=n_valid,
            mae=None, rmse=None, bias=None, median_absolute_error=None,
            mape=None, r_squared=None,
            reconstruction_percentage=0.0,
            remaining_nodata_percentage=round(100.0 * n_missing / (H * W), 4),
            excluded_percentage=0.0,
            processing_duration_ms=duration_ms,
            warnings=warnings,
            limitations=limitations,
            validation_predicted=[],
            validation_observed=[],
            validation_residuals=[],
            validation_coords=[],
        )

    # Create masked array: validation pixels become NaN (excluded from interpolation)
    masked_arr = arr.copy()
    masked_arr[val_mask] = np.nan

    # Run interpolation on masked array
    interp_result: InterpolationResult = run_interpolation(
        masked_arr,
        method=method,
        raster_type=raster_type,
        max_distance_pixels=max_distance_pixels,
        band_index=1,
        **kwargs,
    )

    # Extract predictions at validation pixels
    val_coords = np.argwhere(val_mask)
    predicted_vals = interp_result.reconstructed[val_mask]
    observed_vals = arr[val_mask]

    # Build per-pixel details
    predicted_list = []
    observed_list = []
    residuals_list = []
    coords_list = []

    for i, (r, c) in enumerate(val_coords):
        p = float(predicted_vals[i])
        o = float(observed_vals[i])
        if np.isfinite(p) and np.isfinite(o):
            predicted_list.append(round(p, 6))
            observed_list.append(round(o, 6))
            residuals_list.append(round(p - o, 6))
            coords_list.append([int(r), int(c)])

    # Compute metrics
    metrics = _compute_metrics(observed_vals, predicted_vals)

    # Reconstruction stats from interpolation result
    total_pixels = H * W
    n_interp = interp_result.interpolated_pixel_count
    n_remaining = interp_result.remaining_nodata_count
    n_excluded = interp_result.excluded_pixel_count

    reconstruction_pct = round(100.0 * n_interp / total_pixels, 4)
    remaining_nodata_pct = round(100.0 * n_remaining / total_pixels, 4)
    excluded_pct = round(100.0 * n_excluded / total_pixels, 4)

    # Method-specific limitations
    if method == "ordinary_kriging":
        limitations.append(
            "Kriging variance is not a calibrated confidence interval; "
            "validate independently before using for uncertainty quantification."
        )
    if method in {"bicubic", "idw"}:
        limitations.append(
            "Method may produce values outside the observed local range "
            "(overshooting/undershooting)."
        )
    if validation_strategy == "random_holdout":
        limitations.append(
            "Random holdout validation may underestimate error under spatial "
            "autocorrelation. Spatial block validation is preferred."
        )
    if method == "nearest_neighbor" and raster_type == "continuous":
        limitations.append(
            "Nearest neighbor on continuous rasters produces blocky artifacts; "
            "consider bilinear or IDW for smooth surfaces."
        )

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return ValidationResult(
        method=method,
        config=kwargs,
        validation_strategy=validation_strategy,
        validation_sample_count=len(predicted_list),
        n_excluded_from_validation=n_val - len(predicted_list),
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        bias=metrics["bias"],
        median_absolute_error=metrics["median_absolute_error"],
        mape=metrics["mape"],
        r_squared=metrics["r_squared"],
        reconstruction_percentage=reconstruction_pct,
        remaining_nodata_percentage=remaining_nodata_pct,
        excluded_percentage=excluded_pct,
        processing_duration_ms=duration_ms,
        warnings=warnings,
        limitations=limitations,
        validation_predicted=predicted_list,
        validation_observed=observed_list,
        validation_residuals=residuals_list,
        validation_coords=coords_list,
    )


def get_compatible_methods(
    raster_type: str,
) -> list[dict]:
    """Return list of compatible interpolation methods for a raster type.

    Args:
        raster_type: 'continuous', 'categorical', 'temporal', or 'unknown'.

    Returns:
        List of method info dicts.
    """
    result = []
    for method_id, info in INTERPOLATION_METHODS.items():
        compatible = raster_type in info["supported_raster_types"]
        result.append({
            "id": method_id,
            "label": info["label"],
            "description": info["description"],
            "compatible": compatible,
            "supported_raster_types": info["supported_raster_types"],
            "supports_max_distance": info["supports_max_distance"],
            "supports_categorical": info["supports_categorical"],
            "default_config": info["default_config"],
        })
    return result
