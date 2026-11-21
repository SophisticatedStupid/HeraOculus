"""Interpolation provenance and mask generation service.

Maintains provenance classification for every processing operation:
- observed: original valid observations (unchanged)
- interpolated: reconstructed values
- nodata: remaining missing values
- excluded: pixels intentionally excluded from reconstruction

Generates provenance masks, uncertainty rasters, and export-ready
GeoTIFFs with provenance bands. Never overwrites source files.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.profiles import Profile


@dataclass
class ProvenanceMask:
    """Per-pixel provenance classification."""
    # (H, W) array of strings: 'observed' | 'interpolated' | 'nodata' | 'excluded'
    mask: np.ndarray
    shape: tuple[int, int]
    counts: dict[str, int]  # count per class
    percentages: dict[str, float]  # percentage per class


@dataclass
class InterpolationProvenanceRecord:
    """Complete provenance record for one interpolation operation."""
    operation_id: str
    method: str
    raster_type: str
    config: dict
    input_raster_metadata: dict
    output_raster_metadata: dict
    provenance_mask: ProvenanceMask
    quality_report: Optional[dict]
    processing_duration_ms: float
    started_at: str
    completed_at: str
    source_file: Optional[str]
    output_file: Optional[str]


def build_provenance_mask(
    original_array: np.ndarray,
    reconstructed_array: np.ndarray,
    interpolation_mask: np.ndarray,  # True where interpolated
    excluded_mask: Optional[np.ndarray] = None,  # True where excluded
) -> ProvenanceMask:
    """Build a provenance mask from original, reconstructed, and interpolation masks.

    Classification:
    - observed: original was finite and unchanged
    - interpolated: originally missing, now filled
    - nodata: still missing (NaN in both)
    - excluded: intentionally excluded from interpolation

    Args:
        original_array: Original array (NaN = missing).
        reconstructed_array: Interpolated output array.
        interpolation_mask: True where interpolation was applied.
        excluded_mask: True where pixels were excluded.

    Returns:
        ProvenanceMask with string-classified mask and counts.
    """
    orig = np.asarray(original_array, dtype=np.float64)
    recon = np.asarray(reconstructed_array, dtype=np.float64)
    H, W = orig.shape

    interp_mask = np.asarray(interpolation_mask, dtype=bool)
    if excluded_mask is None:
        excl_mask = np.zeros((H, W), dtype=bool)
    else:
        excl_mask = np.asarray(excluded_mask, dtype=bool)

    mask = np.full((H, W), "nodata", dtype=object)

    # Observed: originally valid
    orig_valid = np.isfinite(orig)
    mask[orig_valid] = "observed"

    # Interpolated: originally missing, now has value, marked as interpolated
    newly_valid = (~orig_valid) & np.isfinite(recon) & interp_mask
    mask[newly_valid] = "interpolated"

    # Excluded
    mask[excl_mask] = "excluded"

    # Remaining NoData: still missing
    still_missing = (~orig_valid) & (~np.isfinite(recon)) & (~interp_mask) & (~excl_mask)
    mask[still_missing] = "nodata"

    # Counts
    classes = ["observed", "interpolated", "nodata", "excluded"]
    counts = {c: int(np.sum(mask == c)) for c in classes}
    total = H * W
    percentages = {
        c: round(100.0 * counts[c] / total, 4) if total > 0 else 0.0
        for c in classes
    }

    return ProvenanceMask(
        mask=mask,
        shape=(H, W),
        counts=counts,
        percentages=percentages,
    )


def compute_quality_raster(
    provenance_mask: np.ndarray,
    distance_to_observation: Optional[np.ndarray] = None,
    kriging_variance: Optional[np.ndarray] = None,
    local_variability: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a per-pixel quality diagnostic raster (0-1 scale).

    Combines available indicators into a heuristic quality score:
    - 1.0: observed (highest confidence)
    - Based on distance to observation, kriging variance, local variability

    This is a DIAGNOSTIC, not a calibrated uncertainty estimate.

    Args:
        provenance_mask: (H, W) string array with provenance classes.
        distance_to_observation: (H, W) distance to nearest valid pixel.
        kriging_variance: (H, W) kriging variance (if available).
        local_variability: (H, W) local std dev of valid neighbors.

    Returns:
        (H, W) float64 quality raster, 0-1 (higher = better quality),
        NaN where provenance is 'nodata' or 'excluded'.
    """
    H, W = provenance_mask.shape
    quality = np.full((H, W), np.nan, dtype=float)

    # Observed pixels: highest quality
    obs_mask = provenance_mask == "observed"
    quality[obs_mask] = 1.0

    # Interpolated pixels: quality based on available diagnostics
    interp_mask = provenance_mask == "interpolated"
    if np.any(interp_mask):
        # Start with a base quality for interpolated pixels
        interp_quality = np.full((H, W), 0.7, dtype=float)

        # Distance to observation: closer = higher quality
        if distance_to_observation is not None:
            dist = distance_to_observation[interp_mask]
            if np.any(np.isfinite(dist)):
                # Exponential decay: quality drops with distance
                dist_quality = np.exp(-dist / 5.0)  # 5-pixel decay scale
                dist_quality = np.clip(dist_quality, 0.1, 1.0)
                interp_quality[interp_mask] = dist_quality

        # Kriging variance: lower variance = higher quality
        if kriging_variance is not None:
            kv = kriging_variance[interp_mask]
            kv_finite = kv[np.isfinite(kv)]
            if len(kv_finite) > 0:
                kv_max = np.percentile(kv_finite, 95)
                if kv_max > 0:
                    kv_quality = np.clip(1.0 - kv / kv_max, 0.1, 1.0)
                    # Combine: take min (conservative)
                    if distance_to_observation is not None:
                        interp_quality[interp_mask] = np.minimum(
                            interp_quality[interp_mask], kv_quality
                        )
                    else:
                        interp_quality[interp_mask] = kv_quality

        # Local variability: higher variability = lower confidence
        if local_variability is not None:
            lv = local_variability[interp_mask]
            lv_finite = lv[np.isfinite(lv)]
            if len(lv_finite) > 0:
                lv_max = np.percentile(lv_finite, 95)
                if lv_max > 0:
                    lv_quality = np.clip(1.0 - lv / lv_max, 0.1, 1.0)
                    interp_quality[interp_mask] = np.minimum(
                        interp_quality[interp_mask], lv_quality
                    )

        quality[interp_mask] = np.clip(interp_quality[interp_mask], 0.0, 1.0)

    # Excluded and nodata: NaN (no quality)
    quality[provenance_mask == "excluded"] = np.nan
    quality[provenance_mask == "nodata"] = np.nan

    return quality


def save_provenance_geotiff(
    *,
    output_path: str,
    reconstructed_array: np.ndarray,
    provenance_mask: np.ndarray,
    quality_raster: Optional[np.ndarray],
    crs: Optional[str],
    transform,
    nodata: float = -9999.0,
    dtype: str = "float32",
    input_metadata: Optional[dict] = None,
    processing_metadata: Optional[dict] = None,
) -> dict:
    """Save reconstructed raster as GeoTIFF with provenance and quality bands.

    Band layout:
      Band 1: Reconstructed values (NaN -> nodata)
      Band 2: Provenance class (encoded as integer: 1=observed, 2=interpolated, 3=nodata, 4=excluded)
      Band 3: Quality score (0-1, NaN -> nodata)

    Does NOT overwrite the input file.

    Returns:
        Dict with output file info and band descriptions.
    """
    H, W = reconstructed_array.shape

    # Encode provenance as integer
    prov_encoding = {"observed": 1, "interpolated": 2, "nodata": 3, "excluded": 4}
    prov_band = np.full((H, W), 3, dtype=np.float32)  # default = nodata
    for cls, code in prov_encoding.items():
        prov_band[provenance_mask == cls] = code

    # Quality band
    if quality_raster is not None:
        qual_band = np.asarray(quality_raster, dtype=np.float32)
    else:
        qual_band = np.full((H, W), np.nan, dtype=np.float32)

    # Replace NaN with nodata for output
    recon_out = np.where(np.isfinite(reconstructed_array), reconstructed_array, nodata)
    prov_out = prov_band.astype(np.float32)
    qual_out = np.where(np.isfinite(qual_band), qual_band, nodata)

    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 3,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }

    # Write to temp then move (avoid partial writes)
    output_path = str(output_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(output_path) or ".")
    os.close(fd)

    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(recon_out.astype(np.float32), 1)
            dst.write(prov_out, 2)
            dst.write(qual_out, 3)
            dst.set_band_description(1, "reconstructed_values")
            dst.set_band_description(2, "provenance_mask")
            dst.set_band_description(3, "quality_score")
            dst.update_tags(
                SOURCE="HeraOculus Raster Interpolation Laboratory",
                METHOD=processing_metadata.get("method", "unknown") if processing_metadata else "unknown",
                PROVENANCE_ENCODING=json.dumps(prov_encoding),
            )

        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    return {
        "output_path": output_path,
        "bands": [
            {"index": 1, "name": "reconstructed_values", "dtype": dtype, "nodata": nodata},
            {"index": 2, "name": "provenance_mask", "dtype": "float32", "nodata": nodata,
             "encoding": prov_encoding},
            {"index": 3, "name": "quality_score", "dtype": "float32", "nodata": nodata},
        ],
        "crs": str(crs) if crs else None,
        "transform": {
            "a": float(transform.a), "b": float(transform.b), "c": float(transform.c),
            "d": float(transform.d), "e": float(transform.e), "f": float(transform.f),
        } if transform else None,
    }


def create_provenance_record(
    *,
    operation_id: str,
    method: str,
    raster_type: str,
    config: dict,
    input_metadata: dict,
    output_metadata: dict,
    provenance_mask: ProvenanceMask,
    quality_report: Optional[dict],
    processing_duration_ms: float,
    source_file: Optional[str] = None,
    output_file: Optional[str] = None,
) -> InterpolationProvenanceRecord:
    """Create a complete provenance record."""
    return InterpolationProvenanceRecord(
        operation_id=operation_id,
        method=method,
        raster_type=raster_type,
        config=config,
        input_raster_metadata=input_metadata,
        output_raster_metadata=output_metadata,
        provenance_mask=provenance_mask,
        quality_report=quality_report,
        processing_duration_ms=processing_duration_ms,
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
        source_file=source_file,
        output_file=output_file,
    )


def provenance_mask_to_geotiff(
    *,
    output_path: str,
    provenance_mask: ProvenanceMask,
    crs: Optional[str],
    transform,
    nodata: float = -9999.0,
) -> dict:
    """Save a standalone provenance mask as GeoTIFF."""
    H, W = provenance_mask.shape
    prov_encoding = {"observed": 1, "interpolated": 2, "nodata": 3, "excluded": 4}
    band = np.full((H, W), 3, dtype=np.float32)
    for cls, code in prov_encoding.items():
        band[provenance_mask.mask == cls] = code

    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }

    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(output_path) or ".")
    os.close(fd)

    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(band, 1)
            dst.set_band_description(1, "provenance_mask")
            dst.update_tags(ENCODING=json.dumps(prov_encoding))
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    return {
        "output_path": output_path,
        "bands": [{"index": 1, "name": "provenance_mask", "dtype": "float32", "nodata": nodata,
                   "encoding": prov_encoding}],
    }
