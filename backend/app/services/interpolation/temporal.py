"""Temporal interpolation for raster time-series cubes.

Fills missing values at each pixel independently using observations from
other dates.  Does NOT interpolate across space — purely temporal.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import TemporalInterpolationResult, _ensure_float_array, _valid_mask


def _temporal_interpolate_1d(
    series: np.ndarray,
    times: np.ndarray,
    method: str,
    max_gap_steps: Optional[int] = None,
    allow_extrapolation: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Interpolate missing values in a 1D time series.

    Args:
        series: 1D array, NaN = missing.
        times: 1D array of decimal-year time coordinates (same length).
        method: 'linear', 'nearest_date', 'forward_fill', 'backward_fill', 'local_temporal_mean'.
        max_gap_steps: Max consecutive missing steps to fill (None = unlimited).
        allow_extrapolation: If True, fill leading/trailing missing values.

    Returns:
        (filled_series, fill_mask, extrapolation_info)
        fill_mask: True where value was interpolated/extrapolated.
    """
    T = len(series)
    filled = series.copy()
    fill_mask = np.zeros(T, dtype=bool)
    extrapolation_info = []

    valid = np.isfinite(series)
    valid_idx = np.where(valid)[0]
    missing_idx = np.where(~valid)[0]

    if len(valid_idx) == 0:
        # No valid observations — nothing to interpolate
        return filled, fill_mask, extrapolation_info

    if len(missing_idx) == 0:
        return filled, fill_mask, extrapolation_info

    # Identify contiguous missing segments
    segments = []
    seg_start = None
    for i in range(T):
        if not valid[i]:
            if seg_start is None:
                seg_start = i
        else:
            if seg_start is not None:
                segments.append((seg_start, i - 1))
                seg_start = None
    if seg_start is not None:
        segments.append((seg_start, T - 1))

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start + 1

        # Check max gap constraint
        if max_gap_steps is not None and seg_len > max_gap_steps:
            continue  # leave as missing

        # Determine if this segment is at the boundary (extrapolation)
        is_leading = seg_start == 0
        is_trailing = seg_end == T - 1
        is_boundary = is_leading or is_trailing

        if is_boundary and not allow_extrapolation:
            # Mark as excluded (not interpolated)
            for i in range(seg_start, seg_end + 1):
                extrapolation_info.append({
                    "pixel_index": i,
                    "type": "boundary_no_extrapolation",
                    "time": float(times[i]) if i < len(times) else None,
                })
            continue

        if method == "linear":
            # Linear interpolation between bracketing valid observations
            left_valid = None
            right_valid = None
            for i in range(seg_start - 1, -1, -1):
                if valid[i]:
                    left_valid = i
                    break
            for i in range(seg_end + 1, T):
                if valid[i]:
                    right_valid = i
                    break

            if left_valid is not None and right_valid is not None:
                # True interpolation between two valid observations
                t_left = times[left_valid]
                t_right = times[right_valid]
                v_left = series[left_valid]
                v_right = series[right_valid]
                for i in range(seg_start, seg_end + 1):
                    t_frac = (times[i] - t_left) / (t_right - t_left) if t_right != t_left else 0.0
                    filled[i] = v_left + t_frac * (v_right - v_left)
                    fill_mask[i] = True
            elif left_valid is not None and is_boundary and allow_extrapolation:
                # Forward extrapolation (extend last valid value)
                for i in range(seg_start, seg_end + 1):
                    filled[i] = series[left_valid]
                    fill_mask[i] = True
                    extrapolation_info.append({
                        "pixel_index": i,
                        "type": "forward_extrapolation",
                        "time": float(times[i]) if i < len(times) else None,
                    })
            elif right_valid is not None and is_boundary and allow_extrapolation:
                # Backward extrapolation
                for i in range(seg_start, seg_end + 1):
                    filled[i] = series[right_valid]
                    fill_mask[i] = True
                    extrapolation_info.append({
                        "pixel_index": i,
                        "type": "backward_extrapolation",
                        "time": float(times[i]) if i < len(times) else None,
                    })
            else:
                # Cannot interpolate — leave as missing
                for i in range(seg_start, seg_end + 1):
                    extrapolation_info.append({
                        "pixel_index": i,
                        "type": "insufficient_brackets",
                        "time": float(times[i]) if i < len(times) else None,
                    })

        elif method == "nearest_date":
            # Use the nearest valid observation in time
            for i in range(seg_start, seg_end + 1):
                # Find nearest valid time index
                best_idx = None
                best_dist = float("inf")
                for vi in valid_idx:
                    d = abs(times[i] - times[vi])
                    if d < best_dist:
                        best_dist = d
                        best_idx = vi
                if best_idx is not None:
                    filled[i] = series[best_idx]
                    fill_mask[i] = True
                    if is_boundary:
                        extrapolation_info.append({
                            "pixel_index": i,
                            "type": "nearest_date_boundary",
                            "time": float(times[i]) if i < len(times) else None,
                        })

        elif method == "forward_fill":
            # Use the last valid observation before this segment
            left_valid = None
            for i in range(seg_start - 1, -1, -1):
                if valid[i]:
                    left_valid = i
                    break
            if left_valid is not None:
                for i in range(seg_start, seg_end + 1):
                    filled[i] = series[left_valid]
                    fill_mask[i] = True
                    if is_boundary:
                        extrapolation_info.append({
                            "pixel_index": i,
                            "type": "forward_fill_extrapolation",
                            "time": float(times[i]) if i < len(times) else None,
                        })
            else:
                # No preceding valid observation
                for i in range(seg_start, seg_end + 1):
                    extrapolation_info.append({
                        "pixel_index": i,
                        "type": "no_preceding_valid",
                        "time": float(times[i]) if i < len(times) else None,
                    })

        elif method == "backward_fill":
            # Use the first valid observation after this segment
            right_valid = None
            for i in range(seg_end + 1, T):
                if valid[i]:
                    right_valid = i
                    break
            if right_valid is not None:
                for i in range(seg_start, seg_end + 1):
                    filled[i] = series[right_valid]
                    fill_mask[i] = True
                    if is_boundary:
                        extrapolation_info.append({
                            "pixel_index": i,
                            "type": "backward_fill_extrapolation",
                            "time": float(times[i]) if i < len(times) else None,
                        })
            else:
                for i in range(seg_start, seg_end + 1):
                    extrapolation_info.append({
                        "pixel_index": i,
                        "type": "no_following_valid",
                        "time": float(times[i]) if i < len(times) else None,
                    })

        elif method == "local_temporal_mean":
            # Mean of all valid observations in the time series
            valid_vals = series[valid]
            if len(valid_vals) > 0:
                mean_val = float(np.mean(valid_vals))
                for i in range(seg_start, seg_end + 1):
                    filled[i] = mean_val
                    fill_mask[i] = True

    return filled, fill_mask, extrapolation_info


def interpolate_temporal(
    cube: np.ndarray,
    *,
    times: np.ndarray,
    method: str = "linear",
    max_gap_steps: Optional[int] = None,
    allow_extrapolation: bool = False,
    band_index: int = 1,
) -> TemporalInterpolationResult:
    """Temporal interpolation for a raster time series cube.

    Fills missing values at each pixel independently using observations
    from other dates. Does NOT interpolate across space — purely temporal.

    Args:
        cube: 3D array (T, H, W), NaN = missing.
        times: 1D array of decimal-year time coordinates (length T).
        method: 'linear', 'nearest_date', 'forward_fill', 'backward_fill', 'local_temporal_mean'.
        max_gap_steps: Max consecutive missing time steps to fill.
        allow_extrapolation: If True, fill leading/trailing gaps (marked as extrapolation).
        band_index: 1-based band number.

    Returns:
        TemporalInterpolationResult.
    """
    start = time.perf_counter()

    T, H, W = cube.shape
    if times is None:
        raise ValueError("`times` is required for temporal interpolation methods.")
    times = np.asarray(times, dtype=float)
    if len(times) != T:
        raise ValueError(
            f"times length ({len(times)}) must equal cube time dimension ({T})."
        )

    reconstructed = cube.copy()
    provenance = np.full((T, H, W), "nodata", dtype=object)
    valid_mask = np.isfinite(cube)
    provenance[valid_mask] = "observed"

    valid_count = int(np.sum(valid_mask))
    interpolated_count = 0
    remaining_nodata = int(np.sum(~valid_mask))
    all_extrapolation_info = []

    # Process each pixel independently
    for r in range(H):
        for c in range(W):
            series = cube[:, r, c]
            if not np.any(~np.isfinite(series)):
                continue  # no missing values at this pixel

            filled, fill_mask, extrap_info = _temporal_interpolate_1d(
                series, times,
                method=method,
                max_gap_steps=max_gap_steps,
                allow_extrapolation=allow_extrapolation,
            )

            reconstructed[:, r, c] = filled
            n_interp = int(np.sum(fill_mask))
            interpolated_count += n_interp
            remaining_nodata -= n_interp

            for i in range(T):
                if fill_mask[i]:
                    provenance[i, r, c] = "interpolated"
                elif not np.isfinite(cube[i, r, c]):
                    provenance[i, r, c] = "excluded"

            for info in extrap_info:
                info["row"] = r
                info["col"] = c
                all_extrapolation_info.append(info)

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return TemporalInterpolationResult(
        reconstructed_cube=reconstructed,
        provenance_cube=provenance,
        method=method,
        config={
            "max_gap_steps": max_gap_steps,
            "allow_extrapolation": allow_extrapolation,
        },
        valid_observation_count=valid_count,
        interpolated_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_count=remaining_nodata,
        processing_duration_ms=duration_ms,
        extrapolated_pixels=all_extrapolation_info,
    )
