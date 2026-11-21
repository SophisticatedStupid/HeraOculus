"""Land Use / Land Cover (LULC) transition matrix & Cohen's Kappa.

Computes cell-by-cell LULC transitions between two classified rasters with
per-pixel area weighting, Cohen's Kappa, and net class gains/losses.
"""

from __future__ import annotations

import tempfile
from typing import Optional

import numpy as np
import os as _os
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .base import (
    _align_second_raster,
    _compute_pixel_areas,
    _determine_area_kind,
    _encode_png,
    _pixel_area_metadata,
    _read_raster_file,
    _render_scalar_png,
)


LULC_CLASS_LABELS = {
    1: "Evergreen Forest",
    2: "Savanna & Shrubland",
    3: "Cropland",
    4: "Urban & Built-up",
    5: "Surface Water",
    6: "Barren / Salt Flat",
}

LULC_PALETTE = {
    1: (22, 163, 74),    # Forest green
    2: (163, 230, 53),   # Savanna lime
    3: (245, 158, 11),   # Cropland amber
    4: (239, 68, 68),    # Urban crimson
    5: (14, 165, 233),   # Water cyan
    6: (194, 178, 128),  # Barren sand
}


def _synthetic_lulc_pair(scenario: str) -> tuple[np.ndarray, np.ndarray, str]:
    H, W = 90, 110
    rng = np.random.default_rng(505 if scenario == "amazon_arc_deforestation" else (606 if scenario == "pearl_river_urbanization" else 707))
    y, x = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")

    if scenario == "amazon_arc_deforestation":
        t1 = np.ones((H, W), dtype=int)
        t1[(y > 0.35) & (x < 0.2)] = 2
        t1[np.abs(y + 0.55) < 0.08] = 5
        t2 = t1.copy()
        clearcut = (x > -0.3) & (x < 0.75) & (y > -0.45) & (y < 0.6) & ((rng.random((H, W)) + 0.25 * x) > 0.45)
        t2[clearcut] = 3  # Pasture / Cropland conversion
        t2[(x > 0.3) & (y > 0.1) & (rng.random((H, W)) > 0.85)] = 4
        label = "Amazon Arc of Deforestation (2001 -> 2023 MCD12Q1 Scenario)"
    elif scenario == "pearl_river_urbanization":
        t1 = np.full((H, W), 3, dtype=int)
        t1[y < -0.3] = 1
        t1[np.abs(x + 0.1 * y) < 0.14] = 5
        t1[(x ** 2 + y ** 2) < 0.12] = 4
        t2 = t1.copy()
        sprawl = (x ** 2 + (y * 1.1) ** 2) < 0.52
        t2[sprawl & (t1 != 5)] = 4
        label = "Pearl River Delta Rapid Urban Sprawl (2001 -> 2023)"
    else:
        t1 = np.full((H, W), 6, dtype=int)
        basin = (x ** 2 / 0.55 + y ** 2 / 0.65) < 1.0
        t1[basin] = 5
        t1[(y > 0.65) & (x < -0.2)] = 3
        t2 = t1.copy()
        desiccated = (x ** 2 / 0.55 + y ** 2 / 0.65) >= 0.22
        t2[basin & desiccated] = 6  # Salt crust desiccation
        label = "Aral Sea Surface Water Desiccation (2000 -> 2023)"

    return t1, t2, label


NODATA_COLOR = (148, 163, 184)  # neutral grey for invalid / excluded LULC pixels


def _render_categorical_png(
    arr: np.ndarray,
    class_ids: list[int],
    *,
    valid: np.ndarray | None = None,
    nodata_color: tuple[int, int, int] = NODATA_COLOR,
) -> str:
    """Render categorical raster.

    ``valid`` (bool, same shape as ``arr``) controls which pixels are rendered. Invalid
    pixels are rendered with ``nodata_color`` (neutral grey) and are never treated as a
    real land-cover class. When ``valid is None``, all pixels are rendered by class value.
    """
    H, W = arr.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    if valid is None:
        rgba[:, :, 3] = 245
    else:
        rgba[~valid, :] = (nodata_color[0], nodata_color[1], nodata_color[2], 245)
    fallback_colors = [(56, 189, 248), (168, 85, 247), (249, 115, 22), (34, 197, 94), (244, 63, 94), (148, 163, 184)]
    for idx, cid in enumerate(class_ids):
        color = LULC_PALETTE.get(int(cid), fallback_colors[idx % len(fallback_colors)])
        mask = arr == cid
        if valid is None:
            rgba[mask, 0] = color[0]
            rgba[mask, 1] = color[1]
            rgba[mask, 2] = color[2]
        else:
            rgba[(mask & valid), 0] = color[0]
            rgba[(mask & valid), 1] = color[1]
            rgba[(mask & valid), 2] = color[2]
    return _encode_png(rgba)


def compute_lulc_transitions(
    *,
    lulc_before: Optional[np.ndarray] = None,
    lulc_after: Optional[np.ndarray] = None,
    scenario: str = "amazon_arc_deforestation",
    pixel_area_km2: float = 0.25,
    pixel_areas_m2: Optional[np.ndarray] = None,
    pixel_area_meta: Optional[dict] = None,
    is_upload: bool = False,
) -> dict:
    """Compute Land Use / Land Cover KxK transition matrix, Cohen's Kappa & net class gains/losses.

    Transition areas are computed from actual per-pixel areas in m^2 (via ``pixel_areas_m2``) when
    supplied for uploaded rasters; otherwise a constant square-pixel assumption of
    ``pixel_area_km2`` (km^2 per pixel) is used.

    Nodata / NaN / Inf pixels are excluded from all calculations. The validity mask is defined
    by the intersection of valid pixels in both the before and after rasters.
    """
    if lulc_before is not None and lulc_after is not None:
        t1_raw, t2_raw = lulc_before, lulc_after
        # Validity: both before and after must be finite (not NaN / Inf / nodata)
        valid = np.isfinite(t1_raw) & np.isfinite(t2_raw)
        t1 = np.rint(t1_raw[valid]).astype(int)
        t2 = np.rint(t2_raw[valid]).astype(int)
        # For rendering: keep original integer values for valid pixels; invalid pixels keep whatever
        # (they will be masked out by `valid` in rendering)
        map1 = np.rint(np.nan_to_num(t1_raw, nan=0)).astype(int)
        map2 = np.rint(np.nan_to_num(t2_raw, nan=0)).astype(int)
        label = "Uploaded Classified LULC GeoTIFF Pair"
        tier = "user_upload"
        msg = f"Evaluated pixel-by-pixel LULC transitions across {len(t1)} valid cells."
    else:
        map1, map2, label = _synthetic_lulc_pair(scenario)
        valid = np.ones_like(map1, dtype=bool)
        t1 = map1.ravel()
        t2 = map2.ravel()
        tier = "synthetic_demo"
        msg = f"MCD12Q1 benchmark scenario ({label}); upload two classified GeoTIFF rasters to compute custom transitions."

    class_ids = sorted(int(c) for c in np.unique(np.concatenate([t1, t2])))[:12]

    provenance = {
        "tier": tier,
        "source_name": label,
        "dataset_id": scenario.upper() if not is_upload else "USER_LULC_PAIR",
        "is_live": False,
        "status_message": msg,
    }
    if pixel_area_meta is not None:
        provenance["pixel_area_meta"] = pixel_area_meta

    total_pixels = int(valid.size)
    valid_pairs = int(t1.size)
    excluded_pixels = int(total_pixels - valid_pairs)
    excluded_pct = round(100.0 * excluded_pixels / max(1, total_pixels), 2)

    # Graceful all-invalid case: report exclusion, no transitions, never treat nodata as a class.
    if valid_pairs == 0:
        change_map_empty = np.zeros(map1.shape, dtype=float)
        return {
            "provenance": provenance,
            "label": label,
            "overall_persistence_pct": 0.0,
            "pct_changed": 0.0,
            "cohen_kappa": 0.0,
            "classes": [],
            "transition_matrix": {"labels": [], "counts": [], "area_km2": []},
            "excluded_pixels": excluded_pixels,
            "excluded_pct": excluded_pct,
            "images": {
                "before_png": _render_categorical_png(map1, [], valid=valid),
                "after_png": _render_categorical_png(map2, [], valid=valid),
                "change_mask_png": _render_scalar_png(change_map_empty, "mask"),
            },
        }

    if len(class_ids) < 2:
        raise ValueError("LULC transition analysis requires at least 2 distinct land cover classes.")

    cm = confusion_matrix(t1, t2, labels=class_ids)
    total = int(np.sum(cm))
    overall_agreement = float(np.trace(cm) / max(1, total))
    kappa = float(cohen_kappa_score(t1, t2, labels=class_ids))

    if pixel_areas_m2 is not None:
        areas_m2 = pixel_areas_m2
        assert areas_m2.shape == map1.shape, f"pixel_areas_m2 shape {areas_m2.shape} != LULC shape {map1.shape}"
        cell_area_m2 = areas_m2[valid]  # aligned with flat t1/t2
    else:
        cell_area_m2 = np.full(len(t1), float(pixel_area_km2) * 1_000_000.0, dtype=float)

    # Per-class before/after area (m^2) and net change (m^2 -> km^2)
    class_before_m2 = {}
    class_after_m2 = {}
    for i, cid in enumerate(class_ids):
        before_mask = (t1 == cid)
        after_mask = (t2 == cid)
        before_m2 = float(np.sum(cell_area_m2[before_mask]))
        after_m2 = float(np.sum(cell_area_m2[after_mask]))
        class_before_m2[cid] = before_m2
        class_after_m2[cid] = after_m2

    # Transition areas (m^2) per (before, after) pair
    transition_m2 = np.zeros((len(class_ids), len(class_ids)), dtype=float)
    for i, ci in enumerate(class_ids):
        for j, cj in enumerate(class_ids):
            m = (t1 == ci) & (t2 == cj)
            transition_m2[i, j] = float(np.sum(cell_area_m2[m])) if np.any(m) else 0.0

    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)

    class_summary = []
    for i, cid in enumerate(class_ids):
        before_px = int(row_sums[i])
        after_px = int(col_sums[i])
        persisted_px = int(cm[i, i])
        net_px = after_px - before_px
        before_m2 = class_before_m2[cid]
        after_m2 = class_after_m2[cid]
        net_m2 = after_m2 - before_m2
        class_summary.append(
            {
                "class_id": cid,
                "label": LULC_CLASS_LABELS.get(cid, f"Class {cid}"),
                "before_pct": round(100.0 * before_px / max(1, total), 2),
                "after_pct": round(100.0 * after_px / max(1, total), 2),
                "net_change_pct_pts": round(100.0 * net_px / max(1, total), 2),
                "net_area_km2": round(net_m2 / 1_000_000.0, 2),
                "persistence_pct": round(100.0 * persisted_px / max(1, before_px), 2),
            }
        )

    # Change map: only on valid paired pixels
    change_map = np.where(valid & (map1 != map2), -1.0, 0.0)

    provenance = {
        "tier": tier,
        "source_name": label,
        "dataset_id": scenario.upper() if not is_upload else "USER_LULC_PAIR",
        "is_live": False,
        "status_message": msg,
    }
    if pixel_area_meta is not None:
        provenance["pixel_area_meta"] = pixel_area_meta

    total_pixels = int(valid.size)
    excluded_pixels = int(total_pixels - total)
    excluded_pct = round(100.0 * excluded_pixels / max(1, total_pixels), 2)

    return {
        "provenance": provenance,
        "label": label,
        "overall_persistence_pct": round(overall_agreement * 100.0, 2),
        "pct_changed": round((1.0 - overall_agreement) * 100.0, 2),
        "cohen_kappa": round(kappa, 4),
        "classes": class_summary,
        "transition_matrix": {
            "labels": [LULC_CLASS_LABELS.get(c, f"Class {c}") for c in class_ids],
            "counts": cm.tolist(),
            "area_km2": [
                [round(float(transition_m2[i, j]) / 1_000_000.0, 2) for j in range(len(class_ids))] for i in range(len(class_ids))
            ],
        },
        "excluded_pixels": excluded_pixels,
        "excluded_pct": excluded_pct,
        "images": {
            "before_png": _render_categorical_png(map1, class_ids, valid=valid),
            "after_png": _render_categorical_png(map2, class_ids, valid=valid),
            "change_mask_png": _render_scalar_png(change_map, "mask"),
        },
    }


def compute_lulc_from_paths(
    before_path: str,
    before_name: str,
    after_path: str,
    after_name: str,
    *,
    pixel_grid_only: bool = False,
) -> dict:
    t1, v1, m1 = _read_raster_file(before_path)
    t2_raw, v2, m2 = _read_raster_file(after_path)
    t2, v2_aligned = _align_second_raster(
        t1, m1, t2_raw, m2, is_categorical=True,
        pixel_grid_only=pixel_grid_only, src_valid=v2,
    )
    area_kind = _determine_area_kind(m1.get("crs"))
    areas_m2 = _compute_pixel_areas(t1, m1.get("transform"), m1.get("crs"), area_kind=area_kind)
    pixel_area_meta = _pixel_area_metadata(m1.get("crs"), areas_m2)
    return compute_lulc_transitions(
        lulc_before=t1,
        lulc_after=t2,
        pixel_areas_m2=areas_m2,
        pixel_area_meta=pixel_area_meta,
        is_upload=True,
    )


def compute_lulc_from_files(
    before_bytes: bytes,
    before_name: str,
    after_bytes: bytes,
    after_name: str,
    *,
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
        return compute_lulc_from_paths(
            b1_path, before_name, b2_path, after_name,
            pixel_grid_only=pixel_grid_only,
        )
    finally:
        for p in (b1_path, b2_path):
            try:
                _os.remove(p)
            except OSError:
                pass
