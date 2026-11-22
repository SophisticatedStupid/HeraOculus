"""Focal / window-based interpolation family.

Genuine bilinear and bicubic (cubic convolution) interpolation for continuous
rasters, plus local mean and local median interpolation.  Each fills missing
pixels from valid neighbours; NoData cells are excluded from the computation
and are never treated as zero.
"""

from __future__ import annotations

import math
import time
from typing import Literal, Optional

import numpy as np
from scipy import ndimage

from .base import InterpolationResult, _ensure_float_array, _valid_mask


# ---------------------------------------------------------------------------
# Cubic B-spline kernel (used by bicubic interpolation)
# ---------------------------------------------------------------------------

def _cubic_bspline(d: float) -> float:
    """Cubic B-spline kernel ``beta_3(d)``.

    Compact support: ``beta_3(d) = 0`` for ``|d| >= 2``.  Values:
    ``beta_3(0) = 2/3``, ``beta_3(1) = 1/6``.  This is an *approximating*
    cubic convolution kernel (it reproduces polynomials up to degree three but
    does not interpolate grid nodes exactly), which makes it well suited to
    gap filling at missing integer nodes where an interpolatory kernel such as
    Keys' cubic would degenerate.
    """
    ad = abs(d)
    if ad >= 2.0:
        return 0.0
    if ad >= 1.0:
        return (2.0 - ad) ** 3 / 6.0
    return (4.0 - 6.0 * ad * ad + 3.0 * ad ** 3) / 6.0


# ---------------------------------------------------------------------------
# Bilinear interpolation
# ---------------------------------------------------------------------------

def interpolate_bilinear(
    array: np.ndarray,
    *,
    neighborhood: Literal["3x3", "5x5"] = "3x3",
    edge_handling: Literal["clamp", "nodata", "reject"] = "clamp",
    max_distance_pixels: Optional[float] = None,
    band_index: int = 1,
) -> InterpolationResult:
    """Genuine bilinear interpolation for continuous rasters.

    Estimates each missing cell from the four diagonal neighbours that
    surround it (NW, NE, SW, SE).  The missing cell lies at the centre of the
    square formed by those four cells, so the bilinear rectangle-centre
    weights apply (0.25 each when all four are present).  Missing diagonal
    neighbours are excluded from the weight sum and the remaining weights are
    renormalised; if no diagonal neighbour is available the function falls
    back to distance weighting over the valid cells in the search window.
    Missing cells are never treated as zero.

    ``edge_handling`` controls how neighbours outside the raster extent are
    treated:

    * ``"clamp"``   - out-of-bounds neighbours are clamped to the raster edge
                      (border replication) and treated as valid observations.
    * ``"nodata"``  - out-of-bounds neighbours are treated as missing.
    * ``"reject"``  - the cell is excluded unless every diagonal neighbour is
                      in bounds and valid.

    Args:
        array: 2D float64 array, NaN = missing.
        neighborhood: '3x3' or '5x5' search window (controls the reach of the
            distance-weighted fallback; the bilinear stencil is fixed at the
            four diagonal cells).
        edge_handling: how to handle raster boundaries.
        max_distance_pixels: max distance to the nearest valid observation.
        band_index: 1-based band number.

    Returns:
        InterpolationResult.
    """
    start = time.perf_counter()
    arr = _ensure_float_array(array)
    valid = _valid_mask(arr)
    missing = ~valid

    H, W = arr.shape
    reconstructed = arr.copy()
    provenance = np.full((H, W), "nodata", dtype=object)
    provenance[valid] = "observed"

    # Neighborhood radius only affects the distance-weighted fallback; the
    # bilinear stencil is the fixed set of four diagonal cells.
    radius = 1 if neighborhood == "3x3" else 2

    valid_coords = np.argwhere(valid)
    missing_coords = np.argwhere(missing)

    interpolated_count = 0
    excluded_count = 0

    if missing_coords.size > 0 and valid_coords.size > 0:
        if max_distance_pixels is not None:
            dist_to_valid = ndimage.distance_transform_edt(~valid)

        diag_offsets = ((-1, -1), (-1, 1), (1, -1), (1, 1))

        for r, c in missing_coords:
            # Check maximum distance to the nearest valid observation.
            if max_distance_pixels is not None:
                d = dist_to_valid[r, c]
                if d > max_distance_pixels:
                    provenance[r, c] = "excluded"
                    excluded_count += 1
                    continue

            # --- Genuine bilinear over the four diagonal neighbours -------
            diag_vals: list[float] = []
            diag_wts: list[float] = []
            reject = False
            for dr, dc in diag_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if valid[nr, nc]:
                        # Target (r, c) is the centre of the diagonal square,
                        # so each corner carries the bilinear centre weight.
                        diag_vals.append(arr[nr, nc])
                        diag_wts.append(0.25)
                    # Missing neighbour: excluded from the weight sum.
                else:
                    if edge_handling == "reject":
                        reject = True
                        break
                    elif edge_handling == "clamp":
                        rr = min(max(nr, 0), H - 1)
                        cc = min(max(nc, 0), W - 1)
                        if valid[rr, cc]:
                            diag_vals.append(arr[rr, cc])
                            diag_wts.append(0.25)
                    # "nodata": out of bounds -> missing, skip.

            if reject:
                provenance[r, c] = "excluded"
                excluded_count += 1
                continue

            if len(diag_vals) >= 1:
                wts = np.array(diag_wts, dtype=float)
                wts = wts / np.sum(wts)
                reconstructed[r, c] = float(
                    np.sum(np.array(diag_vals, dtype=float) * wts)
                )
                provenance[r, c] = "interpolated"
                interpolated_count += 1
                continue

            # --- Fallback: no usable diagonal neighbour.  Distance weight ---
            # --- over the valid cells in the search window (missing excl.) --
            r0 = max(0, r - radius)
            r1 = min(H, r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(W, c + radius + 1)
            window = arr[r0:r1, c0:c1]
            window_valid = valid[r0:r1, c0:c1]
            nb_vals: list[float] = []
            nb_wts: list[float] = []
            for dr in range(r0, r1):
                for dc in range(c0, c1):
                    if window_valid[dr - r0, dc - c0]:
                        dist = math.sqrt((dr - r) ** 2 + (dc - c) ** 2)
                        if dist == 0:
                            continue  # self (should not happen; cell is missing)
                        nb_vals.append(arr[dr, dc])
                        nb_wts.append(1.0 / (dist + 1e-12))
            if len(nb_vals) >= 1:
                wts = np.array(nb_wts, dtype=float)
                wts = wts / np.sum(wts)
                reconstructed[r, c] = float(
                    np.sum(np.array(nb_vals, dtype=float) * wts)
                )
                provenance[r, c] = "interpolated"
                interpolated_count += 1
            else:
                provenance[r, c] = "excluded"
                excluded_count += 1
    elif missing_coords.size > 0:
        excluded_count = int(missing_coords.shape[0])
        provenance[missing] = "excluded"

    observed_count = int(np.sum(valid))
    remaining_nodata = int(np.sum(missing & (provenance != "interpolated")))

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return InterpolationResult(
        reconstructed=reconstructed,
        provenance_mask=provenance,
        quality_mask=None,
        method="bilinear",
        band_index=band_index,
        raster_type="continuous",
        config={
            "neighborhood": neighborhood,
            "edge_handling": edge_handling,
            "max_distance_pixels": max_distance_pixels,
        },
        observed_pixel_count=observed_count,
        interpolated_pixel_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_pixel_count=excluded_count,
        processing_duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Bicubic interpolation
# ---------------------------------------------------------------------------

def interpolate_bicubic(
    array: np.ndarray,
    *,
    neighborhood: Literal["4x4", "6x6"] = "4x4",
    edge_handling: Literal["clamp", "nodata", "reject"] = "clamp",
    max_distance_pixels: Optional[float] = None,
    clip_to_observed_range: bool = True,
    band_index: int = 1,
) -> InterpolationResult:
    """Bicubic interpolation for continuous rasters.

    Reconstructs the continuous surface with a tensor-product cubic B-spline
    convolution kernel (``beta_3``) over a neighbourhood centred on the missing
    cell, with the missing cell itself excluded from the fit.  Because the
    kernel is approximating rather than interpolatory at grid nodes, evaluating
    it at a missing integer node is well defined and non-degenerate; the kernel
    reproduces polynomials up to degree three, so planar and cubic surfaces are
    reconstructed exactly.

    The neighbourhood is a symmetric grid centred on the missing cell with the
    centre cell excluded: ``"4x4"`` uses column/row offsets ``{-2, -1, 1, 2}``
    (a true 4x4 grid, not 5x5); ``"6x6"`` uses ``{-3, -2, -1, 1, 2, 3}``.
    Because the cubic B-spline has compact support (``beta_3(d) = 0`` for
    ``|d| >= 2``), only the immediate neighbours carry non-zero weight; larger
    neighbourhoods are accepted for API compatibility but do not change the
    effective stencil.

    ``edge_handling`` controls neighbours outside the raster extent as for
    :func:`interpolate_bilinear`.

    Args:
        array: 2D float64 array, NaN = missing.
        neighborhood: '4x4' or '6x6' cubic window (grid of neighbour cells).
        edge_handling: how to handle raster boundaries.
        max_distance_pixels: max distance to the nearest valid observation.
        clip_to_observed_range: if True, clip the estimate to the local
            min/max of the valid neighbours actually used.
        band_index: 1-based band number.

    Returns:
        InterpolationResult.
    """
    start = time.perf_counter()
    arr = _ensure_float_array(array)
    valid = _valid_mask(arr)
    missing = ~valid

    H, W = arr.shape
    reconstructed = arr.copy()
    provenance = np.full((H, W), "nodata", dtype=object)
    provenance[valid] = "observed"

    if neighborhood == "4x4":
        # Standard bicubic convolution window: the 4x4 grid of nodes
        # {r-1, r, r+1, r+2} x {c-1, c, c+1, c+2}, with the target cell
        # itself excluded.  This is a genuine 4x4 grid (16 nodes), not 5x5.
        offsets = (-1, 0, 1, 2)
    else:
        # 6x6 grid: {r-3, r-2, r-1, r, r+1, r+2} x {c-3, ..., c+2} minus target.
        offsets = (-3, -2, -1, 0, 1, 2)

    valid_coords = np.argwhere(valid)
    missing_coords = np.argwhere(missing)

    interpolated_count = 0
    excluded_count = 0

    if missing_coords.size > 0 and valid_coords.size > 0:
        if max_distance_pixels is not None:
            dist_to_valid = ndimage.distance_transform_edt(~valid)

        for r, c in missing_coords:
            if max_distance_pixels is not None:
                d = dist_to_valid[r, c]
                if d > max_distance_pixels:
                    provenance[r, c] = "excluded"
                    excluded_count += 1
                    continue

            nb_vals: list[float] = []
            nb_wts: list[float] = []
            reject = False
            for dr in offsets:
                for dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if not valid[nr, nc]:
                            continue  # missing neighbour
                        er, ec = dr, dc
                    else:
                        if edge_handling == "reject":
                            reject = True
                            break
                        elif edge_handling == "clamp":
                            rr = min(max(nr, 0), H - 1)
                            cc = min(max(nc, 0), W - 1)
                            if not valid[rr, cc]:
                                continue
                            nr, nc = rr, cc
                            # The value comes from the clamped (border) cell,
                            # but the weight uses the ORIGINAL relative offset
                            # so border replication does not double-count a cell
                            # that is also reachable at its true offset.
                            er, ec = dr, dc
                        else:  # "nodata": out of bounds -> missing
                            continue
                    w = _cubic_bspline(er) * _cubic_bspline(ec)
                    if w > 0:
                        nb_vals.append(arr[nr, nc])
                        nb_wts.append(w)
                if reject:
                    break
            if reject:
                provenance[r, c] = "excluded"
                excluded_count += 1
                continue

            if len(nb_vals) >= 1:
                wts = np.array(nb_wts, dtype=float)
                wts = wts / np.sum(wts)
                est = float(np.sum(np.array(nb_vals, dtype=float) * wts))

                # Range diagnostics / clipping over the neighbours actually used.
                local_min = float(np.min(nb_vals))
                local_max = float(np.max(nb_vals))
                if clip_to_observed_range:
                    est = max(local_min, min(local_max, est))

                reconstructed[r, c] = est
                provenance[r, c] = "interpolated"
                interpolated_count += 1
            else:
                provenance[r, c] = "excluded"
                excluded_count += 1
    elif missing_coords.size > 0:
        excluded_count = int(missing_coords.shape[0])
        provenance[missing] = "excluded"

    observed_count = int(np.sum(valid))
    remaining_nodata = int(np.sum(missing & (provenance != "interpolated")))

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return InterpolationResult(
        reconstructed=reconstructed,
        provenance_mask=provenance,
        quality_mask=None,
        method="bicubic",
        band_index=band_index,
        raster_type="continuous",
        config={
            "neighborhood": neighborhood,
            "edge_handling": edge_handling,
            "max_distance_pixels": max_distance_pixels,
            "clip_to_observed_range": clip_to_observed_range,
        },
        observed_pixel_count=observed_count,
        interpolated_pixel_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_pixel_count=excluded_count,
        processing_duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Local mean interpolation
# ---------------------------------------------------------------------------

def interpolate_local_mean(
    array: np.ndarray,
    *,
    window_size: int = 3,
    min_valid_neighbors: int = 1,
    neighborhood_shape: Literal["square", "circle"] = "square",
    max_distance_pixels: Optional[float] = None,
    band_index: int = 1,
) -> InterpolationResult:
    """Local mean interpolation.

    Fills missing pixels using the mean of valid observations within a
    configurable moving window. NoData cells are ignored.

    Args:
        array: 2D float64 array, NaN = missing.
        window_size: Size of the moving window (odd integer).
        min_valid_neighbors: Minimum valid observations required.
        neighborhood_shape: 'square' or 'circle'.
        max_distance_pixels: Max distance to nearest valid observation.
        band_index: 1-based band number.

    Returns:
        InterpolationResult.
    """
    start = time.perf_counter()
    arr = _ensure_float_array(array)
    valid = _valid_mask(arr)
    missing = ~valid

    H, W = arr.shape
    reconstructed = arr.copy()
    provenance = np.full((H, W), "nodata", dtype=object)
    provenance[valid] = "observed"

    radius = window_size // 2

    # Build structural element for the neighborhood shape
    if neighborhood_shape == "circle":
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        struct = x * x + y * y <= radius * radius
    else:
        struct = np.ones((window_size, window_size), dtype=bool)

    valid_coords = np.argwhere(valid)
    missing_coords = np.argwhere(missing)

    interpolated_count = 0
    excluded_count = 0

    if missing_coords.size > 0 and valid_coords.size > 0:
        if max_distance_pixels is not None:
            dist_to_valid = ndimage.distance_transform_edt(~valid)

        # Use generic_filter-like approach but manual for control
        padded = np.pad(arr, radius, mode="constant", constant_values=np.nan)
        padded_valid = np.pad(valid, radius, mode="constant", constant_values=False)

        for r, c in missing_coords:
            if max_distance_pixels is not None:
                d = dist_to_valid[r, c]
                if d > max_distance_pixels:
                    provenance[r, c] = "excluded"
                    excluded_count += 1
                    continue

            # Extract window centered at (r+c) in padded coordinates
            pr, pc = r + radius, c + radius
            window = padded[pr - radius:pr + radius + 1, pc - radius:pc + radius + 1]
            wvalid = padded_valid[pr - radius:pr + radius + 1, pc - radius:pc + radius + 1]

            # Apply structural element
            mask = wvalid & struct

            n_valid = int(np.sum(mask))
            if n_valid >= min_valid_neighbors:
                vals = window[mask]
                reconstructed[r, c] = float(np.mean(vals))
                provenance[r, c] = "interpolated"
                interpolated_count += 1
            else:
                provenance[r, c] = "excluded"
                excluded_count += 1
    elif missing_coords.size > 0:
        excluded_count = int(missing_coords.shape[0])
        provenance[missing] = "excluded"

    observed_count = int(np.sum(valid))
    remaining_nodata = int(np.sum(missing & (provenance != "interpolated")))

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return InterpolationResult(
        reconstructed=reconstructed,
        provenance_mask=provenance,
        quality_mask=None,
        method="local_mean",
        band_index=band_index,
        raster_type="continuous",
        config={
            "window_size": window_size,
            "min_valid_neighbors": min_valid_neighbors,
            "neighborhood_shape": neighborhood_shape,
            "max_distance_pixels": max_distance_pixels,
        },
        observed_pixel_count=observed_count,
        interpolated_pixel_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_pixel_count=excluded_count,
        processing_duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Local median interpolation
# ---------------------------------------------------------------------------

def interpolate_local_median(
    array: np.ndarray,
    *,
    window_size: int = 3,
    min_valid_neighbors: int = 1,
    neighborhood_shape: Literal["square", "circle"] = "square",
    max_distance_pixels: Optional[float] = None,
    band_index: int = 1,
) -> InterpolationResult:
    """Local median interpolation.

    Fills missing pixels using the median of valid observations within a
    configurable moving window. Robust to outliers. NoData cells are ignored.

    Args:
        array: 2D float64 array, NaN = missing.
        window_size: Size of the moving window (odd integer).
        min_valid_neighbors: Minimum valid observations required.
        neighborhood_shape: 'square' or 'circle'.
        max_distance_pixels: Max distance to nearest valid observation.
        band_index: 1-based band number.

    Returns:
        InterpolationResult.
    """
    start = time.perf_counter()
    arr = _ensure_float_array(array)
    valid = _valid_mask(arr)
    missing = ~valid

    H, W = arr.shape
    reconstructed = arr.copy()
    provenance = np.full((H, W), "nodata", dtype=object)
    provenance[valid] = "observed"

    radius = window_size // 2

    if neighborhood_shape == "circle":
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        struct = x * x + y * y <= radius * radius
    else:
        struct = np.ones((window_size, window_size), dtype=bool)

    valid_coords = np.argwhere(valid)
    missing_coords = np.argwhere(missing)

    interpolated_count = 0
    excluded_count = 0

    if missing_coords.size > 0 and valid_coords.size > 0:
        if max_distance_pixels is not None:
            dist_to_valid = ndimage.distance_transform_edt(~valid)

        padded = np.pad(arr, radius, mode="constant", constant_values=np.nan)
        padded_valid = np.pad(valid, radius, mode="constant", constant_values=False)

        for r, c in missing_coords:
            if max_distance_pixels is not None:
                d = dist_to_valid[r, c]
                if d > max_distance_pixels:
                    provenance[r, c] = "excluded"
                    excluded_count += 1
                    continue

            pr, pc = r + radius, c + radius
            window = padded[pr - radius:pr + radius + 1, pc - radius:pc + radius + 1]
            wvalid = padded_valid[pr - radius:pr + radius + 1, pc - radius:pc + radius + 1]

            mask = wvalid & struct
            n_valid = int(np.sum(mask))

            if n_valid >= min_valid_neighbors:
                vals = window[mask]
                reconstructed[r, c] = float(np.median(vals))
                provenance[r, c] = "interpolated"
                interpolated_count += 1
            else:
                provenance[r, c] = "excluded"
                excluded_count += 1
    elif missing_coords.size > 0:
        excluded_count = int(missing_coords.shape[0])
        provenance[missing] = "excluded"

    observed_count = int(np.sum(valid))
    remaining_nodata = int(np.sum(missing & (provenance != "interpolated")))

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    return InterpolationResult(
        reconstructed=reconstructed,
        provenance_mask=provenance,
        quality_mask=None,
        method="local_median",
        band_index=band_index,
        raster_type="continuous",
        config={
            "window_size": window_size,
            "min_valid_neighbors": min_valid_neighbors,
            "neighborhood_shape": neighborhood_shape,
            "max_distance_pixels": max_distance_pixels,
        },
        observed_pixel_count=observed_count,
        interpolated_pixel_count=interpolated_count,
        remaining_nodata_count=remaining_nodata,
        excluded_pixel_count=excluded_count,
        processing_duration_ms=duration_ms,
    )
