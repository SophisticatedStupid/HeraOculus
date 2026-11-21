"""SAR speckle filtering, log-ratio change detection & NASA ASF proxy.

Computes SAR backscatter change with Lee / mean speckle filtering, log-ratio
thresholding (fixed or Otsu), and optional NASA ASF granule search.
"""

from __future__ import annotations

import tempfile
from typing import Literal, Optional

import numpy as np
import requests
import os as _os
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu

from .base import (
    _align_second_raster,
    _compute_pixel_areas,
    _determine_area_kind,
    _pixel_area_metadata,
    _read_raster_file,
    _render_scalar_png,
)


def _lee_speckle_filter(
    img: np.ndarray, window_size: int = 5, valid: Optional[np.ndarray] = None
) -> np.ndarray:
    """Classic Lee adaptive multiplicative speckle filter preserving edges.

    When a ``valid`` mask is provided, local mean and variance are computed only
    from valid pixels in the window (via normalized convolution), and the filtered
    result at invalid positions is set to NaN.  Invalid pixels do not contaminate
    the local statistics of valid pixels.
    """
    eps = 1e-6

    # Determine validity mask
    if valid is None:
        valid = np.isfinite(img)

    # Normalised convolution: weight = validity mask (1 for valid, 0 for invalid)
    w = valid.astype(float)
    sum_w = uniform_filter(w, size=window_size)          # count of valid pixels in window
    sum_val = uniform_filter(img * w, size=window_size)  # sum of valid values in window
    sum_sq = uniform_filter((img**2) * w, size=window_size)  # sum of squared valid values

    # Local mean (only where at least one valid pixel exists in window)
    mean_local = np.where(sum_w > 0, sum_val / sum_w, 0.0)

    # Local variance: E[X^2] - E[X]^2
    var_local = np.maximum(0.0, sum_sq / np.maximum(sum_w, eps) - mean_local**2)

    # Estimate global speckle noise CV from valid pixels only
    cv_mask = valid & np.isfinite(mean_local) & (np.maximum(mean_local, eps) > 0) & (var_local > 0)
    cv_count = int(np.sum(cv_mask))
    if cv_count > 0:
        cv_global = float(np.median(np.sqrt(var_local[cv_mask]) / np.maximum(mean_local[cv_mask], eps)))
    else:
        cv_global = 1.0  # default when no valid pixels have statistics

    var_noise = (mean_local * cv_global) ** 2

    # Weight: clip to [0,1]; at invalid positions the weight is computed but will be overridden
    weight = np.clip(np.maximum(0.0, var_local - var_noise) / np.maximum(var_local, eps), 0.0, 1.0)

    # Filtered output
    result = mean_local + weight * (img - mean_local)

    # At invalid positions, set result to NaN (will be masked out downstream)
    result = np.where(valid, result, np.nan)

    return result


def _mean_speckle_filter(
    img: np.ndarray, window_size: int = 5, valid: Optional[np.ndarray] = None
) -> np.ndarray:
    """Boxcar mean filter with optional validity mask.

    Local mean is computed only from valid pixels in the window.  Invalid
    pixels are set to NaN in the output so they never contaminate valid results.
    """
    if valid is None:
        valid = np.isfinite(img)

    w = valid.astype(float)
    sum_w = uniform_filter(w, size=window_size)
    sum_val = uniform_filter(img * w, size=window_size)

    result = np.where(sum_w > 0, sum_val / sum_w, np.nan)

    # Ensure invalid positions are NaN
    result = np.where(valid, result, np.nan)
    return result


def _sar_result_all_invalid(
    *,
    label: str,
    tier: str,
    msg: str,
    scenario: str,
    speckle_filter: str,
    window_size: int,
    threshold_db: float,
    threshold_mode: str,
    valid_pixels: int,
    excluded_pixels: int,
    total_pixels: int,
    coverage_pct: float,
) -> dict:
    """Short-circuit result when every pixel is invalid (no valid data)."""
    provenance = {
        "tier": tier,
        "source_name": label,
        "dataset_id": scenario.upper(),
        "is_live": False,
        "status_message": msg,
    }
    return {
        "provenance": provenance,
        "label": label,
        "speckle_filter": speckle_filter,
        "window_size": window_size,
        "threshold_mode": threshold_mode,
        "user_threshold_db": threshold_db,
        "otsu_suggested_threshold_db": 0.0,
        "applied_threshold_db": threshold_db,
        "summary": {
            "total_area_km2": 0.0,
            "backscatter_drop_area_km2": 0.0,
            "backscatter_gain_area_km2": 0.0,
            "pct_changed": 0.0,
            "valid_pixels": valid_pixels,
            "excluded_pixels": excluded_pixels,
            "changed_pixels": 0,
            "valid_data_coverage_pct": coverage_pct,
            "mean_log_ratio_db": 0.0,
            "std_log_ratio_db": 0.0,
        },
        "histogram": {"centers_db": [], "counts": []},
        "images": {
            "before_png": _render_scalar_png(np.full((1, 1), np.nan), "sar"),
            "after_png": _render_scalar_png(np.full((1, 1), np.nan), "sar"),
            "log_ratio_png": _render_scalar_png(np.full((1, 1), np.nan), "diverging"),
            "change_mask_png": _render_scalar_png(np.full((1, 1), np.nan), "mask"),
        },
    }


def _synthetic_sar_pair(scenario: str) -> tuple[np.ndarray, np.ndarray, str]:
    H, W = 96, 120
    y = np.linspace(-1.0, 1.0, H)
    x = np.linspace(-1.2, 1.2, W)
    xg, yg = np.meshgrid(x, y)
    rng = np.random.default_rng(777 if scenario == "amazon_seasonal_flooding" else (888 if scenario == "rondonia_deforestation_clearcut" else 999))

    # Underlying linear backscatter intensity + Gamma(L=4) SAR speckle
    base_rcs = 0.25 + 0.12 * np.sin(3 * xg) * np.cos(3 * yg)
    if scenario == "amazon_seasonal_flooding":
        # Open floodwater causes specular bounce away -> sharp backscatter DROP (-7 to -11 dB)
        flood_zone = np.exp(-((yg - 0.15 * np.sin(3.5 * xg)) / 0.22) ** 2) > 0.45
        rcs2 = np.where(flood_zone, base_rcs * 0.12, base_rcs)
        label = "Amazon Floodplain Inundation (Specular Water Backscatter Drop)"
    elif scenario == "rondonia_deforestation_clearcut":
        # Fishbone clearcut polygons reduce C/L-band volume scattering (-4.5 to -7 dB)
        fishbone = (np.abs(xg) < 0.75) & ((np.abs(yg - 0.35) < 0.16) | (np.abs(yg + 0.30) < 0.14))
        rcs2 = np.where(fishbone, base_rcs * 0.26, base_rcs)
        label = "Rondonia Forest Clear-Cut Herringbone Loss"
    else:
        # Fresh rough a'a lava flow & urban damage doubles/triples rough backscatter (+5 to +8 dB)
        lava_lobe = ((xg + 0.1) ** 2 / 0.35 + (yg - 0.05) ** 2 / 0.12) < 1.0
        rcs2 = np.where(lava_lobe, base_rcs * 4.2, base_rcs)
        label = "Kilauea Rough Lava Flow Emplacement (+Backscatter Gain)"

    # Apply realistic 4-look Gamma SAR speckle noise
    i1 = base_rcs * rng.gamma(shape=4.0, scale=0.25, size=(H, W))
    i2 = rcs2 * rng.gamma(shape=4.0, scale=0.25, size=(H, W))
    return i1, i2, label


def compute_sar_change_detection(
    *,
    img_before: Optional[np.ndarray] = None,
    img_after: Optional[np.ndarray] = None,
    scenario: str = "amazon_seasonal_flooding",
    speckle_filter: Literal["none", "lee", "mean"] = "lee",
    window_size: int = 5,
    threshold_db: float = 3.0,
    threshold_mode: Literal["fixed", "otsu"] = "fixed",
    pixel_size_m: float = 20.0,
    pixel_areas_m2: Optional[np.ndarray] = None,
    pixel_area_meta: Optional[dict] = None,
    is_upload: bool = False,
) -> dict:
    """Compute SAR log-ratio change detection with Lee speckle filtering & Otsu threshold.

    Parameters
    ----------
    threshold_mode : {"fixed", "otsu"}
        "fixed"  — apply the user-supplied ``threshold_db``.
        "otsu"   — compute Otsu threshold from |log10 ratio| and apply it.
        Defaults to ``"fixed"`` for backward compatibility.

    Nodata / NaN / Inf handling
    ---------------------------
    A joint validity mask is built from both input rasters at the start of
    processing (pixels must be finite in **both** inputs to be valid).  The
    mask is preserved throughout the pipeline:

    * speckle filters compute local statistics **only** from valid pixels
      (invalid pixels in the window are excluded, not filled with a constant)
    * the log-ratio is computed only for valid pixels; invalid pixels are NaN
    * the Otsu threshold is estimated from ``|log-ratio|`` of valid pixels only
    * changed-pixel counts and area statistics use valid pixels as the denominator
    * the change-mask PNG renders invalid pixels as transparent
    """
    if threshold_mode not in ("fixed", "otsu"):
        raise ValueError(f"Invalid threshold_mode '{threshold_mode}'; expected 'fixed' or 'otsu'.")

    if threshold_mode == "otsu" and threshold_db <= 0:
        raise ValueError("threshold_db must be positive when using otsu mode.")

    # ------------------------------------------------------------------
    # Input: either user-uploaded arrays or synthetic benchmark pair
    # ------------------------------------------------------------------
    if img_before is not None and img_after is not None:
        i1_raw, i2_raw = np.asarray(img_before, dtype=float), np.asarray(img_after, dtype=float)
        if i1_raw.shape != i2_raw.shape:
            raise ValueError(
                f"SAR raster shape mismatch: before {i1_raw.shape} != after {i2_raw.shape}"
            )
        label = "Uploaded Dual SAR GeoTIFF Backscatter Pair"
        tier = "user_upload"
        msg = f"Processed uploaded {i1_raw.shape[0]}x{i1_raw.shape[1]} SAR intensity scenes with {speckle_filter} speckle filtering."
    else:
        i1_raw, i2_raw, label = _synthetic_sar_pair(scenario)
        tier = "synthetic_demo"
        msg = f"Sentinel-1 / NISAR backscatter benchmark ({label}); upload two GeoTIFF SAR bands to analyze real scenes."

    # ------------------------------------------------------------------
    # Joint validity mask — finite in BOTH rasters
    # ------------------------------------------------------------------
    valid = np.isfinite(i1_raw) & np.isfinite(i2_raw)
    total_pixels = int(valid.size)
    valid_pixels = int(np.sum(valid))
    excluded_pixels = total_pixels - valid_pixels
    coverage_pct = round(100.0 * valid_pixels / max(1, total_pixels), 2)

    # ------------------------------------------------------------------
    # Guard: all pixels invalid → short-circuit with safe defaults
    # ------------------------------------------------------------------
    if valid_pixels == 0:
        total_area_km2 = 0.0
        return _sar_result_all_invalid(
            label=label,
            tier=tier,
            msg=msg,
            scenario=scenario,
            speckle_filter=speckle_filter,
            window_size=window_size,
            threshold_db=threshold_db,
            threshold_mode=threshold_mode,
            valid_pixels=0,
            excluded_pixels=excluded_pixels,
            total_pixels=total_pixels,
            coverage_pct=0.0,
        )

    # ------------------------------------------------------------------
    # Prepare working arrays: replace invalid with 0, then clamp to eps.
    # This preserves NaN propagation through subsequent steps.
    # ------------------------------------------------------------------
    eps = 1e-6
    i1_filled = np.where(valid, i1_raw, 0.0)
    i2_filled = np.where(valid, i2_raw, 0.0)

    # ------------------------------------------------------------------
    # Speckle filtering — must not contaminate valid pixels
    # ------------------------------------------------------------------
    if speckle_filter == "lee":
        f1 = _lee_speckle_filter(i1_filled, window_size, valid=valid)
        f2 = _lee_speckle_filter(i2_filled, window_size, valid=valid)
    elif speckle_filter == "mean":
        f1 = _mean_speckle_filter(i1_filled, window_size, valid=valid)
        f2 = _mean_speckle_filter(i2_filled, window_size, valid=valid)
    else:
        f1, f2 = i1_filled.copy(), i2_filled.copy()

    # Clamp to eps — NaN from invalid positions propagates naturally
    i1_pos = np.maximum(f1, eps)
    i2_pos = np.maximum(f2, eps)
    # Ensure invalid positions are NaN (not clamped to eps) for rendering / downstream use
    i1_pos = np.where(valid, i1_pos, np.nan)
    i2_pos = np.where(valid, i2_pos, np.nan)

    # ------------------------------------------------------------------
    # Log-ratio — only valid pixels; invalid pixels → NaN
    # ------------------------------------------------------------------
    log_ratio_db = np.full(i1_raw.shape, np.nan, dtype=float)
    log_ratio_db[valid] = 10.0 * np.log10(i2_pos[valid] / i1_pos[valid])
    abs_change = np.abs(log_ratio_db)  # NaN at invalid positions

    # ------------------------------------------------------------------
    # Threshold estimation — valid pixels only
    # ------------------------------------------------------------------
    try:
        otsu_db = float(threshold_otsu(abs_change[valid]))
        # A meaningful Otsu cut must be finite and strictly positive; flat / degenerate
        # histograms (e.g. constant scenes) yield 0.0 which would flag every pixel, so
        # fall back to the user-supplied fixed threshold in that case.
        if not np.isfinite(otsu_db) or otsu_db <= 0.0:
            otsu_db = float(threshold_db)
    except Exception:
        otsu_db = float(threshold_db)

    # Select the applied threshold based on the requested mode
    applied_threshold = otsu_db if threshold_mode == "otsu" else threshold_db

    # ------------------------------------------------------------------
    # Build gain / loss masks — valid pixels only; invalid pixels are not changed
    # ------------------------------------------------------------------
    changed_drop = (log_ratio_db <= -applied_threshold) & valid
    changed_gain = (log_ratio_db >= applied_threshold) & valid
    unchanged_valid = valid & ~(changed_drop | changed_gain)

    # Change mask for rendering: +1 gain, -1 drop, 0 unchanged, NaN for invalid
    mask_arr = np.where(changed_gain, 1.0, np.where(changed_drop, -1.0, 0.0))
    mask_arr = np.where(~valid, np.nan, mask_arr)

    # ------------------------------------------------------------------
    # Area calculations — valid pixels only
    # ------------------------------------------------------------------
    if pixel_areas_m2 is not None:
        areas_m2 = pixel_areas_m2
        assert areas_m2.shape == i1_raw.shape, f"pixel_areas_m2 shape {areas_m2.shape} != SAR shape {i1_raw.shape}"
    else:
        areas_m2 = np.full(i1_raw.shape, pixel_size_m ** 2, dtype=float)

    total_area_km2 = float(np.sum(areas_m2[valid])) / 1_000_000.0
    drop_area_km2 = float(np.sum(areas_m2[changed_drop])) / 1_000_000.0
    gain_area_km2 = float(np.sum(areas_m2[changed_gain])) / 1_000_000.0

    # Changed pixel counts
    changed_pixels = int(np.sum(changed_drop) + np.sum(changed_gain))

    # ------------------------------------------------------------------
    # Histogram — valid pixels only
    # ------------------------------------------------------------------
    valid_log_ratio = log_ratio_db[valid]
    hist_counts, hist_edges = np.histogram(valid_log_ratio, bins=24)

    # ------------------------------------------------------------------
    # Provenance metadata
    # ------------------------------------------------------------------
    provenance = {
        "tier": tier,
        "source_name": label,
        "dataset_id": scenario.upper() if not is_upload else "USER_SAR_PAIR",
        "is_live": False,
        "status_message": msg,
    }
    if pixel_area_meta is not None:
        provenance["pixel_area_meta"] = pixel_area_meta

    # ------------------------------------------------------------------
    # Summary statistics — valid pixels only
    # ------------------------------------------------------------------
    pct_changed = round(100.0 * changed_pixels / max(1, valid_pixels), 2)

    return {
        "provenance": provenance,
        "label": label,
        "speckle_filter": speckle_filter,
        "window_size": window_size,
        "threshold_mode": threshold_mode,
        "user_threshold_db": threshold_db,
        "otsu_suggested_threshold_db": round(float(otsu_db), 3),
        "applied_threshold_db": round(float(applied_threshold), 3),
        "summary": {
            "total_area_km2": round(total_area_km2, 3),
            "backscatter_drop_area_km2": round(drop_area_km2, 3),
            "backscatter_gain_area_km2": round(gain_area_km2, 3),
            "pct_changed": pct_changed,
            "valid_pixels": valid_pixels,
            "excluded_pixels": excluded_pixels,
            "changed_pixels": changed_pixels,
            "valid_data_coverage_pct": coverage_pct,
            "mean_log_ratio_db": round(float(np.mean(valid_log_ratio)), 3),
            "std_log_ratio_db": round(float(np.std(valid_log_ratio)), 3),
        },
        "histogram": {
            "centers_db": [round(float(0.5 * (hist_edges[i] + hist_edges[i + 1])), 3) for i in range(len(hist_counts))],
            "counts": [int(c) for c in hist_counts],
        },
        "images": {
            "before_png": _render_scalar_png(10.0 * np.log10(i1_pos), "sar"),
            "after_png": _render_scalar_png(10.0 * np.log10(i2_pos), "sar"),
            "log_ratio_png": _render_scalar_png(log_ratio_db, "diverging"),
            "change_mask_png": _render_scalar_png(mask_arr, "mask"),
        },
    }


def compute_sar_change_from_paths(
    before_path: str,
    before_name: str,
    after_path: str,
    after_name: str,
    *,
    speckle_filter: Literal["none", "lee", "mean"] = "lee",
    window_size: int = 5,
    threshold_db: float = 3.0,
    threshold_mode: Literal["fixed", "otsu"] = "fixed",
    pixel_grid_only: bool = False,
) -> dict:
    i1, v1, m1 = _read_raster_file(before_path)
    i2_raw, v2, m2 = _read_raster_file(after_path)
    i2, v2_aligned = _align_second_raster(
        i1, m1, i2_raw, m2, is_categorical=False,
        pixel_grid_only=pixel_grid_only, src_valid=v2,
    )
    area_kind = _determine_area_kind(m1.get("crs"))
    areas_m2 = _compute_pixel_areas(i1, m1.get("transform"), m1.get("crs"), area_kind=area_kind)
    pixel_area_meta = _pixel_area_metadata(m1.get("crs"), areas_m2)
    return compute_sar_change_detection(
        img_before=i1,
        img_after=i2,
        speckle_filter=speckle_filter,
        window_size=window_size,
        threshold_db=threshold_db,
        threshold_mode=threshold_mode,
        pixel_areas_m2=areas_m2,
        pixel_area_meta=pixel_area_meta,
        is_upload=True,
    )


def compute_sar_change_from_files(
    before_bytes: bytes,
    before_name: str,
    after_bytes: bytes,
    after_name: str,
    *,
    speckle_filter: Literal["none", "lee", "mean"] = "lee",
    window_size: int = 5,
    threshold_db: float = 3.0,
    threshold_mode: Literal["fixed", "otsu"] = "fixed",
    pixel_grid_only: bool = False,
) -> dict:
    """Backward-compatible wrapper: writes bytes to temp files then delegates."""
    b1_fd, b1_path = tempfile.mkstemp(suffix=".tif")
    b2_fd, b2_path = tempfile.mkstemp(suffix=".tif")
    try:
        with _os.fdopen(b1_fd, "wb") as fh:
            fh.write(before_bytes)
        with _os.fdopen(b2_fd, "wb") as fh:
            fh.write(after_bytes)
        return compute_sar_change_from_paths(
            b1_path, before_name, b2_path, after_name,
            speckle_filter=speckle_filter,
            window_size=window_size,
            threshold_db=threshold_db,
            threshold_mode=threshold_mode,
            pixel_grid_only=pixel_grid_only,
        )
    finally:
        for p in (b1_path, b2_path):
            try:
                _os.remove(p)
            except OSError:
                pass


def search_asf_sar_granules(
    *,
    lat: float,
    lon: float,
    radius_deg: float = 0.4,
    platform: str = "SENTINEL-1",
    start_date: str = "2024-01-01T00:00:00Z",
    end_date: str = "2025-12-31T23:59:59Z",
    max_results: int = 12,
) -> dict:
    """Real keyless query to NASA Alaska Satellite Facility (ASF) SAR Search API."""
    url = "https://api.asf.alaska.edu/services/search/param"
    params = {
        "bbox": f"{lon - radius_deg:.3f},{lat - radius_deg:.3f},{lon + radius_deg:.3f},{lat + radius_deg:.3f}",
        "platform": platform,
        "start": start_date,
        "end": end_date,
        "maxResults": max_results,
        "output": "geojson",
    }
    try:
        resp = requests.get(url, params=params, timeout=18)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        granules = []
        for feat in features:
            props = feat.get("properties", {})
            granules.append(
                {
                    "scene_name": props.get("sceneName"),
                    "platform": props.get("platform"),
                    "beam_mode": props.get("beamModeType"),
                    "polarization": props.get("polarization"),
                    "flight_direction": props.get("flightDirection"),
                    "path_number": props.get("pathNumber"),
                    "frame_number": props.get("frameNumber"),
                    "start_time": props.get("startTime"),
                    "bytes_mb": round(float(props.get("bytes", 0)) / (1024 * 1024), 1) if props.get("bytes") else None,
                    "browse_url": (props.get("browse") or [None])[0] if isinstance(props.get("browse"), list) else props.get("browse"),
                    "download_url": props.get("url"),
                }
            )
        return {
            "provenance": {
                "tier": "satellite_observation",
                "source_name": "NASA Alaska Satellite Facility (ASF) SAR Archive",
                "dataset_id": f"ASF_{platform}",
                "is_live": True,
                "status_message": f"Retrieved {len(granules)} live {platform} SAR scenes near ({lat:.2f}, {lon:.2f}) from NASA ASF.",
                "citation": "Alaska Satellite Facility Distributed Active Archive Center (ASF DAAC)",
            },
            "granule_count": len(granules),
            "granules": granules,
        }
    except Exception as exc:
        return {
            "provenance": {
                "tier": "synthetic_demo",
                "source_name": "NASA ASF SAR Search Fallback",
                "dataset_id": f"ASF_{platform}",
                "is_live": False,
                "status_message": f"ASF SAR catalog request failed ({exc}).",
            },
            "granule_count": 0,
            "granules": [],
        }
