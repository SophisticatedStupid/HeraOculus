"""Shared raster I/O, alignment, pixel-area and rendering helpers.

This module holds the low-level utilities shared by the DEM, SAR and LULC
sub-modules.  It has no knowledge of any specific change-detection algorithm.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import rasterio
from PIL import Image
from pyproj import CRS as pyCRS, Geod
from rasterio.warp import Resampling, reproject


def _read_raster_file(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a raster from a file path, returning (array, validity_mask, metadata).

    The array is float64 with NaN where nodata / missing values occur.  The
    validity mask is ``True`` where the pixel is finite (valid).  This separates
    raster values from their validity so downstream resampling cannot
    contaminate valid pixels with invalid source cells.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        valid = np.isfinite(arr)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "res": src.res,
            "nodata": nodata,
        }
        return arr, valid, meta


def _read_raster_bytes(raw_bytes: bytes, filename: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a raster from in-memory bytes via a temporary file (backward-compat wrapper)."""
    import tempfile
    import os as _os
    suffix = Path(filename).suffix.lower() or ".tif"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with _os.fdopen(fd, "wb") as handle:
            handle.write(raw_bytes)
        return _read_raster_file(tmp_path)
    finally:
        try:
            _os.remove(tmp_path)
        except OSError:
            pass


def _crs_equivalent(crs1, crs2) -> bool:
    """Check if two CRS definitions are equivalent."""
    if crs1 is None or crs2 is None:
        return False
    try:
        return crs1.to_epsg() == crs2.to_epsg()
    except Exception:
        try:
            return crs1.to_wkt() == crs2.to_wkt()
        except Exception:
            return False


def _validate_alignment(ref_arr: np.ndarray, ref_meta: dict, src_arr: np.ndarray, src_meta: dict) -> bool:
    """Validate that two rasters have identical dimensions and affine transform."""
    return (
        ref_arr.shape == src_arr.shape
        and ref_meta.get("transform") == src_meta.get("transform")
    )


def _reproject_to_grid(
    src_arr: np.ndarray,
    src_meta: dict,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    *,
    is_categorical: bool,
) -> np.ndarray:
    """Reproject source raster to exact destination grid, preserving NaN nodata."""
    dst = np.full(dst_shape, np.nan, dtype=float)
    resampling = Resampling.nearest if is_categorical else Resampling.bilinear
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def _pixel_grid_resize(
    ref_arr: np.ndarray,
    src_arr: np.ndarray,
    *,
    is_categorical: bool,
    src_valid: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize source array to reference shape using PIL (pixel-grid-only mode).

    Returns ``(resized_array, resized_valid_mask)``.  Invalid source cells are
    excluded from interpolation (continuous) or resampling (categorical) and do
    not contaminate valid output cells.  The output validity mask is reconstructed
    from the source validity mask after resizing.

    WARNING: This ignores georeferencing. Only use when both rasters lack CRS
    and you explicitly accept pixel-grid comparison.
    """
    warnings.warn(
        "Pixel-grid-only alignment: ignoring georeferencing. "
        "Use only for non-georeferenced rasters with identical spatial extent.",
        UserWarning,
        stacklevel=2,
    )
    from PIL import Image

    if src_valid is None:
        src_valid = np.isfinite(src_arr)

    # Build two-channel image: [values, valid_mask]
    # For categorical: use nearest-neighbor; invalid cells get a sentinel value
    # For continuous: use bilinear with zero-filled invalid cells, then mask output
    H_ref, W_ref = ref_arr.shape
    H_src, W_src = src_arr.shape

    if is_categorical:
        # Categorical: nearest-neighbor. Invalid cells are kept as NaN (sentinel)
        # so they don't introduce a fake class 0.
        filled = np.where(src_valid, src_arr, np.nan)
        img = Image.fromarray(filled.astype(np.float32), mode="F")
        resized = img.resize((W_ref, H_ref), Image.Resampling.NEAREST)
        resized_arr = np.asarray(resized, dtype=float)
        resized_valid = np.isfinite(resized_arr)
        return resized_arr, resized_valid
    else:
        # Continuous: bilinear. Zero-fill invalid cells for interpolation, then
        # mask the output so invalid source cells don't contaminate valid results.
        filled = np.where(src_valid, src_arr, 0.0)
        img = Image.fromarray(filled.astype(np.float32), mode="F")
        resized = img.resize((W_ref, H_ref), Image.Resampling.BILINEAR)
        resized_arr = np.asarray(resized, dtype=float)

        # Build validity mask: dilate source validity to match interpolation kernel
        # For bilinear, a valid output pixel can be influenced by up to 4 source pixels.
        # We use a conservative approach: output is valid if ALL contributing source
        # pixels are valid (no interpolation across invalid boundaries).
        from scipy.ndimage import binary_erosion
        src_valid_padded = np.pad(src_valid, 1, mode='constant', constant_values=False)
        # For each output pixel, check if its source window is fully valid
        # Simple approach: use binary_erosion with a 2x2 kernel on source validity,
        # then resize that mask with nearest (conservative).
        eroded = binary_erosion(src_valid, structure=np.ones((2, 2)))
        eroded_img = Image.fromarray(eroded.astype(np.float32), mode="F")
        resized_eroded = eroded_img.resize((W_ref, H_ref), Image.Resampling.NEAREST)
        resized_valid = np.asarray(resized_eroded, dtype=bool)

        # Mark invalid cells as NaN in the output array
        resized_arr[~resized_valid] = np.nan
        return resized_arr, resized_valid


def _align_second_raster(
    ref_arr: np.ndarray,
    ref_meta: dict,
    src_arr: np.ndarray,
    src_meta: dict,
    *,
    is_categorical: bool = False,
    pixel_grid_only: bool = False,
    src_valid: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject or resample second raster to match first raster's exact grid.

    Returns ``(aligned_array, aligned_valid_mask)`` where ``aligned_valid_mask``
    is ``True`` where the aligned pixel is finite (valid).  Nodata metadata from
    both rasters is preserved as NaN/False so invalid pixels never become
    legitimate measurements.

    Args:
        ref_arr: Reference raster array
        ref_meta: Reference raster metadata (crs, transform, width, height, res)
        src_arr: Source raster array to align
        src_meta: Source raster metadata
        is_categorical: Use nearest-neighbor for categorical (LULC), bilinear for continuous
        pixel_grid_only: If True, allow pixel-grid resize when CRS missing. Default False.
        src_valid: Validity mask of source array (True where valid). If None,
            derived from np.isfinite(src_arr).

    Returns:
        (aligned_array, aligned_valid_mask) matching ref_arr shape and grid.

    Raises:
        ValueError: If CRS mismatch and pixel_grid_only=False, or if one has CRS and other doesn't.
    """
    if src_valid is None:
        src_valid = np.isfinite(src_arr)

    ref_crs = ref_meta.get("crs")
    src_crs = src_meta.get("crs")
    grids_match = _validate_alignment(ref_arr, ref_meta, src_arr, src_meta)

    # Fast path: identical grid AND equivalent CRS -> copy unchanged.
    if grids_match and ref_crs is not None and src_crs is not None and _crs_equivalent(ref_crs, src_crs):
        return src_arr.copy(), src_valid.copy()

    # Both have CRS: reproject onto reference grid (covers same-CRS-different-grid
    # and different-CRS cases).
    if ref_crs is not None and src_crs is not None:
        aligned_arr = _reproject_to_grid(
            src_arr, src_meta,
            ref_arr.shape, ref_meta["transform"], ref_crs,
            is_categorical=is_categorical,
        )
        # Reproject already preserves NaN nodata; rebuild validity mask from array.
        aligned_valid = np.isfinite(aligned_arr)
        return aligned_arr, aligned_valid

    # One has CRS, the other does not: reject (never mix georeferenced with raw).
    if (ref_crs is None) != (src_crs is None):
        raise ValueError(
            "CRS mismatch: one raster has a coordinate reference system and the other does not. "
            "Both rasters must either have identical CRS definitions or both lack CRS metadata. "
            "Use pixel_grid_only=True only if you explicitly accept non-georeferenced comparison."
        )

    # Neither has CRS: do not silently assume geographic compatibility.
    if pixel_grid_only:
        if not grids_match:
            return _pixel_grid_resize(ref_arr, src_arr, is_categorical=is_categorical, src_valid=src_valid)
        return src_arr.copy(), src_valid.copy()

    raise ValueError(
        "Both rasters lack CRS metadata. Georeferenced comparison requires CRS. "
        "Provide GeoTIFFs with valid CRS, or set pixel_grid_only=True for pixel-grid-only mode "
        "(not recommended for scientific analysis)."
    )


# ---------------------------------------------------------------------------
# Pixel-area & physical-unit helpers
# ---------------------------------------------------------------------------


def _as_pyproj_crs(crs):
    """Coerce a rasterio CRS (or other) into a pyproj CRS for linear-unit queries."""
    if crs is None:
        return None
    try:
        return pyCRS(crs) if hasattr(crs, "to_wkt") else pyCRS.from_user_input(crs)
    except Exception:
        return None


def _get_crs_linear_units_to_meters(crs) -> float:
    """Linear unit conversion factor (metres per CRS unit) for a projected CRS."""
    if crs is None:
        raise ValueError("CRS is required for linear-unit conversion.")
    pcrs = _as_pyproj_crs(crs)
    try:
        factor = float(pcrs.axis_info[0].unit_conversion_factor)
        if np.isfinite(factor) and factor > 0:
            return factor
    except Exception:
        pass
    return 1.0


def _determine_area_kind(crs) -> str:
    """Classify CRS for pixel-area methodology. Returns 'geographic' or 'projected'."""
    if crs is None:
        raise ValueError("CRS is required to determine pixel-area methodology.")
    pcrs = _as_pyproj_crs(crs)
    try:
        return "geographic" if pcrs.is_geographic else "projected"
    except Exception:
        return "projected"


def _compute_pixel_areas(
    arr: np.ndarray,
    transform,
    crs,
    *,
    area_kind: str,
) -> np.ndarray:
    """Compute per-pixel areas in square metres.

    Projected CRS: area = |a*e - b*d| * (unit_to_m)^2 (constant across grid; handles
    non-square pixels and rotated affine transforms).

    Geographic CRS: per-cell geodesic polygon area (latitude-dependent).

    Returns a float array of shape ``arr.shape`` with areas in m^2.
    """
    n_rows, n_cols = arr.shape
    a, b, c, d, e, f = (
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f,
    )

    if area_kind == "geographic":
        geod = Geod(ellps="WGS84")
        areas = np.zeros((n_rows, n_cols), dtype=float)
        for r in range(n_rows):
            for cc in range(n_cols):
                lons = np.array([
                    a * cc + b * r + c,
                    a * (cc + 1) + b * r + c,
                    a * (cc + 1) + b * (r + 1) + c,
                    a * cc + b * (r + 1) + c,
                ])
                lats = np.array([
                    d * cc + e * r + f,
                    d * (cc + 1) + e * r + f,
                    d * (cc + 1) + e * (r + 1) + f,
                    d * cc + e * (r + 1) + f,
                ])
                area, _ = geod.polygon_area_perimeter(lons.tolist(), lats.tolist())
                areas[r, cc] = abs(area)
        return areas

    # projected CRS: constant area across grid
    det = abs(a * e - b * d)
    unit_factor = _get_crs_linear_units_to_meters(crs)
    if not np.isfinite(unit_factor) or unit_factor <= 0:
        raise ValueError(
            f"Unsupported or ambiguous CRS linear unit definition for {crs}. "
            "Cannot convert CRS units to metres."
        )
    return np.full((n_rows, n_cols), det * (unit_factor ** 2), dtype=float)


def _pixel_area_metadata(crs, areas: np.ndarray) -> dict:
    """Explicit metadata describing the pixel-area methodology and units."""
    if crs is None:
        return {
            "methodology": "synthetic_demo_constant_square_pixel",
            "area_units": "m2",
            "crs": None,
            "linear_units": None,
            "area_constant": True,
        }
    kind = _determine_area_kind(crs)
    unit_factor = _get_crs_linear_units_to_meters(crs) if kind == "projected" else None
    return {
        "methodology": "geodesic_per_cell" if kind == "geographic" else "affine_determinant",
        "area_units": "m2",
        "crs": str(crs),
        "linear_units": (
            "metre" if unit_factor is not None and abs(unit_factor - 1.0) < 1e-9
            else f"{unit_factor} m per CRS unit" if unit_factor is not None
            else None
        ),
        "area_constant": bool(areas.size > 0 and np.allclose(areas, areas.flat[0])),
    }


def _encode_png(rgba: np.ndarray, target_size: tuple[int, int] = (360, 280)) -> str:
    img = Image.fromarray(rgba, mode="RGBA").resize(target_size, Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _render_scalar_png(arr: np.ndarray, palette: Literal["terrain", "diverging", "sar", "mask"]) -> str:
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return _encode_png(rgba)

    if palette == "diverging":
        vmax = max(float(np.percentile(np.abs(np.nan_to_num(arr[valid], nan=0.0, posinf=0.0, neginf=0.0)), 98)), 1e-3)
        norm = np.clip(np.nan_to_num(arr, nan=0.0) / vmax, -1.0, 1.0)
        rgba[:, :, 0] = np.where(norm >= 0, 35 + 220 * norm, 35 * (1 + norm)).astype(np.uint8)
        rgba[:, :, 1] = np.where(norm >= 0, 110 + 90 * (1 - norm), 130 + 95 * (-norm)).astype(np.uint8)
        rgba[:, :, 2] = np.where(norm >= 0, 130 * (1 - norm), 175 + 80 * (-norm)).astype(np.uint8)
        rgba[:, :, 3] = np.where(valid, 245, 0)
    elif palette == "mask":
        # +1 deposition/increase (cyan), -1 erosion/flood drop (amber/red), 0 unchanged
        rgba[:, :, 0] = np.where(arr < -0.5, 244, np.where(arr > 0.5, 56, 20)).astype(np.uint8)
        rgba[:, :, 1] = np.where(arr < -0.5, 88, np.where(arr > 0.5, 189, 30)).astype(np.uint8)
        rgba[:, :, 2] = np.where(arr < -0.5, 66, np.where(arr > 0.5, 248, 48)).astype(np.uint8)
        rgba[:, :, 3] = np.where(valid, 245, 0)
    else:
        lo, hi = np.percentile(arr[valid], [2, 98])
        span = max(float(hi - lo), 1e-6)
        u = np.clip((np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo) - lo) / span, 0.0, 1.0)
        if palette == "sar":
            val = (20 + 235 * u).astype(np.uint8)
            rgba[:, :, 0] = val
            rgba[:, :, 1] = (val * 0.96).astype(np.uint8)
            rgba[:, :, 2] = np.clip(val * 1.05, 0, 255).astype(np.uint8)
        else:
            # Terrain tint
            rgba[:, :, 0] = (32 + 185 * (u ** 1.3)).astype(np.uint8)
            rgba[:, :, 1] = (58 + 150 * np.sin(u * np.pi * 0.85)).astype(np.uint8)
            rgba[:, :, 2] = (75 + 145 * (1.0 - 0.5 * u)).astype(np.uint8)
        rgba[:, :, 3] = np.where(valid, 250, 0)

    return _encode_png(rgba)
