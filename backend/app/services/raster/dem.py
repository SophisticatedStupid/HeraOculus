"""DEM of Difference (DoD) change detection.

Computes cell-by-cell elevation difference between two DEMs with
Level-of-Detection (LoD) filtering and sediment / ice volume budgeting.
"""

from __future__ import annotations

import tempfile
from typing import Optional

import numpy as np
import os as _os

from .base import (
    _align_second_raster,
    _compute_pixel_areas,
    _determine_area_kind,
    _pixel_area_metadata,
    _read_raster_file,
    _render_scalar_png,
)


def _synthetic_dem_pair(scenario: str) -> tuple[np.ndarray, np.ndarray, str]:
    H, W = 96, 120
    y = np.linspace(-1.0, 1.0, H)
    x = np.linspace(-1.2, 1.2, W)
    xg, yg = np.meshgrid(x, y)
    rng = np.random.default_rng(101 if scenario == "mount_st_helens_lahar" else (202 if scenario == "colorado_river_incision" else 303))

    if scenario == "mount_st_helens_lahar":
        dem1 = 1850.0 - 420.0 * np.sqrt(xg ** 2 + yg ** 2) + 25.0 * np.sin(4 * xg) * np.cos(4 * yg)
        # Crater breach / incision upper flank + downstream lahar deposition lobe
        erosion = -28.0 * np.exp(-(((xg + 0.35) / 0.25) ** 2 + ((yg - 0.1) / 0.28) ** 2))
        deposition = 16.5 * np.exp(-(((xg - 0.45) / 0.34) ** 2 + ((yg + 0.08) / 0.24) ** 2))
        delta = erosion + deposition + rng.normal(0.0, 0.28, size=(H, W))
        label = "Mount St. Helens Volcanic Sector Collapse & Lahar Deposition"
    elif scenario == "colorado_river_incision":
        channel = np.exp(-((yg - 0.2 * np.sin(3 * xg)) / 0.14) ** 2)
        dem1 = 920.0 + 180.0 * np.abs(yg) - 95.0 * channel
        erosion = -4.2 * channel * (0.6 + 0.4 * np.cos(2 * xg))
        bar_dep = 2.4 * np.exp(-(((xg - 0.25) / 0.22) ** 2 + ((yg + 0.16) / 0.10) ** 2))
        delta = erosion + bar_dep + rng.normal(0.0, 0.22, size=(H, W))
        label = "Fluvial Canyon Incision & Point-Bar Aggradation"
    else:
        dem1 = 210.0 + 140.0 * (1.0 - xg) + 18.0 * np.sin(5 * yg)
        # Grounding-zone ice shelf dynamic thinning (-18 to -35 m)
        thinning = -26.0 * np.exp(-((xg - 0.3) / 0.55) ** 2) * (0.65 + 0.35 * np.cos(2 * yg))
        delta = thinning + rng.normal(0.0, 0.30, size=(H, W))
        label = "Thwaites Glacier Grounding-Zone Dynamic Ice Thinning"

    return dem1, dem1 + delta, label


def compute_dem_differencing(
    *,
    dem_before: Optional[np.ndarray] = None,
    dem_after: Optional[np.ndarray] = None,
    scenario: str = "mount_st_helens_lahar",
    lod_threshold_m: float = 0.75,
    pixel_size_m: float = 10.0,
    pixel_areas_m2: Optional[np.ndarray] = None,
    pixel_area_meta: Optional[dict] = None,
    is_upload: bool = False,
) -> dict:
    """Compute DEM of Difference (DoD) with Level-of-Detection & sediment/ice volume budget.

    For user-uploaded GeoTIFFs, per-pixel areas in m² may be supplied via ``pixel_areas_m2``;
    otherwise a constant square-pixel assumption of ``pixel_size_m`` (metres) is used.

    If ``pixel_areas_m2`` is supplied, volume calculations use per-pixel areas (so non-square
    / rotated grids are handled). Otherwise each pixel is assumed to have area ``pixel_size_m**2``.
    """
    if dem_before is not None and dem_after is not None:
        z1, z2 = dem_before, dem_after
        label = "Uploaded GeoTIFF DEM Pair"
        tier = "user_upload"
        msg = f"Computed DEM of Difference across uploaded {z1.shape[0]}x{z1.shape[1]} elevation grids."
    else:
        z1, z2, label = _synthetic_dem_pair(scenario)
        tier = "synthetic_demo"
        msg = f"Curated landform change benchmark ({label}); upload two GeoTIFF DEMs (.tif) to evaluate real terrain."

    # ------------------------------------------------------------------
    # Validate inputs: build joint validity mask and check for valid pixels
    # ------------------------------------------------------------------
    z1_valid = np.isfinite(z1)
    z2_valid = np.isfinite(z2)
    valid = z1_valid & z2_valid  # both DEMs must be finite for a valid difference
    excluded = int(np.sum(~valid))
    valid_pixels = int(np.sum(valid))

    dod = z2 - z1
    # Re-apply joint validity: where either input is nodata/invalid, difference is invalid
    dod[~valid] = np.nan
    dod_valid = dod[valid]

    if pixel_areas_m2 is not None:
        areas_m2 = pixel_areas_m2
        assert areas_m2.shape == z1.shape, f"pixel_areas_m2 shape {areas_m2.shape} != DEM shape {z1.shape}"
    else:
        areas_m2 = np.full(z1.shape, pixel_size_m ** 2, dtype=float)

    if pixel_area_meta is None:
        # No explicit metadata: describe the assumed constant square-pixel geometry.
        pixel_area_meta = {
            "methodology": (
                "synthetic_demo_constant_square_pixel" if tier == "synthetic_demo"
                else "constant_square_pixel_assumed"
            ),
            "area_units": "m2",
            "crs": None,
            "linear_units": None,
            "area_constant": True,
        }

    # Handle case with no valid overlapping pixels
    if valid_pixels == 0:
        zeros_mask = np.zeros_like(dod)
        return {
            "provenance": {
                "tier": tier,
                "source_name": label,
                "dataset_id": scenario.upper() if not is_upload else "USER_DEM_PAIR",
                "is_live": False,
                "status_message": msg,
            },
            "label": label,
            "dimensions": {
                "rows": int(z1.shape[0]),
                "cols": int(z1.shape[1]),
                "pixel_size_m": float(pixel_size_m),
            },
            "lod_threshold_m": lod_threshold_m,
            "summary": {
                "valid_pixels": 0,
                "excluded_pixels": excluded if excluded > 0 else int(z1.size),
                "min_change_m": 0.0,
                "max_change_m": 0.0,
                "mean_change_m": 0.0,
                "std_change_m": 0.0,
                "median_change_m": 0.0,
                "max_deposition_m": 0.0,
                "max_erosion_m": 0.0,
                "mean_elevation_change_m": 0.0,
                "pct_area_changed": 0.0,
                "deposition_volume_m3": 0.0,
                "erosion_volume_m3": 0.0,
                "net_volume_change_m3": 0.0,
            },
            "histogram": {
                "centers_m": [],
                "counts": [],
            },
            "images": {
                "before_png": _render_scalar_png(z1, "terrain"),
                "after_png": _render_scalar_png(z2, "terrain"),
                "dod_png": _render_scalar_png(dod, "diverging"),
                "lod_mask_png": _render_scalar_png(zeros_mask, "mask"),
            },
        }

    sig = valid & (np.abs(dod) >= lod_threshold_m)
    deposition_mask = sig & (dod > 0)
    erosion_mask = sig & (dod < 0)

    dep_vol_m3 = float(np.sum(dod[deposition_mask] * areas_m2[deposition_mask])) if np.any(deposition_mask) else 0.0
    ero_vol_m3 = float(np.sum(np.abs(dod[erosion_mask]) * areas_m2[erosion_mask])) if np.any(erosion_mask) else 0.0
    net_vol_m3 = dep_vol_m3 - ero_vol_m3

    sig_class = np.where(deposition_mask, 1.0, np.where(erosion_mask, -1.0, 0.0))
    dod_valid = dod[valid]
    hist_counts, hist_edges = np.histogram(dod_valid, bins=24)

    # Standard deviation of valid differences
    std_change = round(float(np.std(dod_valid)), 3) if valid_pixels > 1 else 0.0

    # Assemble provenance metadata
    provenance = {
        "tier": tier,
        "source_name": label,
        "dataset_id": scenario.upper() if not is_upload else "USER_DEM_PAIR",
        "is_live": False,
        "status_message": msg,
    }
    if pixel_area_meta is not None:
        provenance["pixel_area_meta"] = pixel_area_meta

    return {
        "provenance": provenance,
        "label": label,
        "dimensions": {
            "rows": int(dod.shape[0]),
            "cols": int(dod.shape[1]),
            "pixel_size_m": float(np.sqrt(areas_m2[0, 0])) if areas_m2.size else pixel_size_m,
        },
        "lod_threshold_m": lod_threshold_m,
        "summary": {
            "valid_pixels": valid_pixels,
            "excluded_pixels": excluded,
            "min_change_m": round(float(np.min(dod_valid)), 3),
            "max_change_m": round(float(np.max(dod_valid)), 3),
            "mean_change_m": round(float(np.mean(dod_valid)), 3),
            "std_change_m": std_change,
            "median_change_m": round(float(np.median(dod_valid)), 3),
            "max_deposition_m": round(float(np.max(dod_valid)), 3),
            "max_erosion_m": round(float(np.min(dod_valid)), 3),
            "mean_elevation_change_m": round(float(np.mean(dod_valid)), 3),
            "pct_area_changed": round(100.0 * float(np.sum(sig)) / max(1, valid_pixels), 2),
            "deposition_volume_m3": round(dep_vol_m3, 1),
            "erosion_volume_m3": round(ero_vol_m3, 1),
            "net_volume_change_m3": round(net_vol_m3, 1),
        },
        "histogram": {
            "centers_m": [round(float(0.5 * (hist_edges[i] + hist_edges[i + 1])), 3) for i in range(len(hist_counts))],
            "counts": [int(c) for c in hist_counts],
        },
        "images": {
            "before_png": _render_scalar_png(z1, "terrain"),
            "after_png": _render_scalar_png(z2, "terrain"),
            "dod_png": _render_scalar_png(dod, "diverging"),
            "lod_mask_png": _render_scalar_png(sig_class, "mask"),
        },
    }


def compute_dem_differencing_from_paths(
    before_path: str,
    before_name: str,
    after_path: str,
    after_name: str,
    *,
    lod_threshold_m: float = 0.75,
    pixel_grid_only: bool = False,
) -> dict:
    z1, v1, m1 = _read_raster_file(before_path)
    z2_raw, v2, m2 = _read_raster_file(after_path)
    z2, v2_aligned = _align_second_raster(
        z1, m1, z2_raw, m2, is_categorical=False,
        pixel_grid_only=pixel_grid_only, src_valid=v2,
    )
    # The reference raster's validity mask is preserved as-is (no resampling needed).
    area_kind = _determine_area_kind(m1.get("crs"))
    areas_m2 = _compute_pixel_areas(z1, m1.get("transform"), m1.get("crs"), area_kind=area_kind)
    pixel_area_meta = _pixel_area_metadata(m1.get("crs"), areas_m2)
    return compute_dem_differencing(
        dem_before=z1,
        dem_after=z2,
        lod_threshold_m=lod_threshold_m,
        pixel_areas_m2=areas_m2,
        pixel_area_meta=pixel_area_meta,
        is_upload=True,
    )


def compute_dem_differencing_from_files(
    before_bytes: bytes,
    before_name: str,
    after_bytes: bytes,
    after_name: str,
    *,
    lod_threshold_m: float = 0.75,
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
        return compute_dem_differencing_from_paths(
            b1_path, before_name, b2_path, after_name,
            lod_threshold_m=lod_threshold_m,
            pixel_grid_only=pixel_grid_only,
        )
    finally:
        for p in (b1_path, b2_path):
            try:
                _os.remove(p)
            except OSError:
                pass
