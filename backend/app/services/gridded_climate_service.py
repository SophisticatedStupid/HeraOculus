"""Gridded climate / reanalysis rainfall service.

Provides normalized monthly precipitation time series from public gridded
products:

* CHIRPS v3 (Climate Hazards Center) — keyless HTTP GeoTIFF, 0.05°, mm/month
* GPM IMERG Final (GPM_3IMERGM V07B, GES DISC) — NASA Earthdata login, 0.1°, mm/hr
* ERA5-Land (Copernicus CDS / ARCO Zarr mirror) — CDS key, 0.1°, metres

Each function returns a normalized :class:`GriddedSeries`:

    {
        "dates": ["YYYY-MM", ...],
        "values": [float | None, ...],   # None = missing / fill / out of coverage
        "units": "mm/month",
        "dataset": "CHIRPS v3",
        "version": "3.0",
        "spatial_resolution": "0.05 deg",
        "temporal_resolution": "monthly",
        "provenance": "satellite_observation" | "reanalysis_model",
        "source_url": "...",
        "coverage_note": "...",
    }

Design rules
------------
* Point extraction uses the nearest grid cell (no fabrication across cells).
* Area-mean extraction averages all valid (non-fill, finite) cells whose
  cell-centre falls inside the requested bounding box; if no valid cell falls
  inside, the month is reported as missing (None) rather than interpolated.
* Synthetic data never masquerades as live: every series originates from a
  live upstream product, missing credentials or an out-of-coverage query yield
  an explicit ``unavailable`` result (all values None / empty dates) rather
  than a fabricated series.
* Upstream requests are cached on disk with a TTL so repeat requests do not
  hammer the provider, and per-dataset rate-limit guidance is respected.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import requests

from .auth_store import auth_store
from .config import settings

# ── Cache ───────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(
    os.getenv("GEOIRIS_CACHE_DIR", str(Path(__file__).resolve().parents[2] / ".cache" / "gridded"))
)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default TTL: 24 hours.  Monthly products change at most once per month, so
# a day is short enough to stay fresh while sparing upstream on repeats.
DEFAULT_TTL_SECONDS = 24 * 60 * 60


def _cache_key(dataset: str, payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{dataset}_{digest}.bin"


def _cache_get(dataset: str, payload: str, ttl_seconds: int = DEFAULT_TTL_SECONDS):
    path = _CACHE_DIR / _cache_key(dataset, payload)
    if not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > ttl_seconds:
            return None
        return path.read_bytes()
    except OSError:
        return None


def _cache_set(dataset: str, payload: str, data: bytes) -> None:
    path = _CACHE_DIR / _cache_key(dataset, payload)
    try:
        path.write_bytes(data)
    except OSError:
        # Cache write failure must never fail the request.
        pass


# ── Shared helpers ──────────────────────────────────────────────────────────

def _month_iter(
    start_year: int, start_month: int, end_year: int, end_month: int
):
    """Yield (year, month) pairs inclusive, chronological."""
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def _validate_month_range(
    start_year: int, start_month: int, end_year: int, end_month: int
) -> None:
    if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
        raise ValueError("month must be in 1..12")
    if (start_year, start_month) > (end_year, end_month):
        raise ValueError("start_date must be <= end_date")
    # Cap at ~40 years to bound a single request.
    span_months = (end_year - start_year) * 12 + (end_month - start_month) + 1
    if span_months > 40 * 12:
        raise ValueError("date range must not exceed 40 years")


def _coords_to_grid_index(
    lon: float, lat: float, n_lon: int, n_lat: int, res: float
) -> tuple[int, int]:
    """Nearest cell (col, row) for a lon/lat on a global -180..180, 90..-90 grid."""
    col = int(round((lon + 180.0) / res))
    row = int(round((90.0 - lat) / res))
    col = max(0, min(n_lon - 1, col))
    row = max(0, min(n_lat - 1, row))
    return col, row


def _to_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


# ── Exceptions ──────────────────────────────────────────────────────────────

class GriddedUnavailable(Exception):
    """Raised when a dataset cannot be served (missing credentials or
    out-of-coverage).  Routers map this to an explicit 'unavailable'
    response — never synthetic data."""


class GriddedUpstreamError(Exception):
    """Upstream request failed (network / HTTP error / malformed payload)."""


# ── Normalized result ───────────────────────────────────────────────────────

@dataclass
class GriddedSeries:
    dates: list[str]
    values: list[Optional[float]]
    units: str
    dataset: str
    version: str
    spatial_resolution: str
    temporal_resolution: str
    provenance: str
    source_url: str
    coverage_note: str
    valid_count: int = field(init=False)
    missing_count: int = field(init=False)

    def __post_init__(self):
        self.valid_count = sum(1 for v in self.values if v is not None)
        self.missing_count = sum(1 for v in self.values if v is None)

    def to_dict(self) -> dict:
        return {
            "dates": self.dates,
            "values": self.values,
            "units": self.units,
            "dataset": self.dataset,
            "version": self.version,
            "spatial_resolution": self.spatial_resolution,
            "temporal_resolution": self.temporal_resolution,
            "provenance": self.provenance,
            "source_url": self.source_url,
            "coverage_note": self.coverage_note,
            "valid_count": self.valid_count,
            "missing_count": self.missing_count,
        }


# ── CHIRPS v3 (Climate Hazards Center) ──────────────────────────────────────

CHIRPS_BASE_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/"
)
CHIRPS_FILENAME_PATTERN = "chirps-v3.0.{year:04d}.{month:02d}.tif"
CHIRPS_RES_DEG = 0.05
CHIRPS_N_LON = int(360.0 / CHIRPS_RES_DEG)  # 7200
CHIRPS_N_LAT = int(120.0 / CHIRPS_RES_DEG)  # 2400 (60°S..60°N)
CHIRPS_LAT_MIN = -60.0
CHIRPS_LAT_MAX = 60.0
CHIRPS_NODATA = -9999.0


def _chirps_url(year: int, month: int) -> str:
    return CHIRPS_BASE_URL + CHIRPS_FILENAME_PATTERN.format(year=year, month=month)


def _fetch_chirps_geotiff(year: int, month: int) -> bytes:
    """Download one CHIRPS monthly GeoTIFF (keyless).  Cached on disk with TTL."""
    payload = f"chirps_v3_monthly_{year:04d}_{month:02d}"
    cached = _cache_get("chirps", payload)
    if cached is not None:
        return cached

    url = _chirps_url(year, month)
    try:
        resp = requests.get(
            url,
            timeout=settings.http_timeout_seconds,
            headers={"User-Agent": "Geo-Iris/2.0 (gridded-climate-service; contact: ops@example.com)"},
        )
    except requests.Timeout as exc:
        raise GriddedUpstreamError(f"CHIRPS request timed out for {year}-{month:02d}: {exc}") from exc
    except requests.RequestException as exc:
        raise GriddedUpstreamError(f"CHIRPS request failed for {year}-{month:02d}: {exc}") from exc

    if resp.status_code == 404:
        # File not yet published (near-real-time lag) — treat as missing month.
        return b""
    if resp.status_code >= 400:
        raise GriddedUpstreamError(
            f"CHIRPS returned HTTP {resp.status_code} for {year}-{month:02d}"
        )

    data = resp.content
    _cache_set("chirps", payload, data)
    return data


def _read_chirps_cell(
    data: bytes, col: int, row: int
) -> Optional[float]:
    """Read a single cell from CHIRPS GeoTIFF bytes; None if fill/nodata."""
    if not data:
        return None
    import io

    buf = io.BytesIO(data)
    try:
        with rasterio.open(buf) as src:
            band = src.read(1, masked=True)
            nodata = src.nodata
    except Exception as exc:
        raise GriddedUpstreamError(f"Failed to decode CHIRPS GeoTIFF: {exc}") from exc

    try:
        cell = band[row, col]
        if np.ma.is_masked(cell):
            return None
        val = float(cell)
    except IndexError:
        return None

    if nodata is not None and val == float(nodata):
        return None
    if val == CHIRPS_NODATA or not math.isfinite(val):
        return None
    return val


def _read_chirps_area_mean(
    data: bytes, min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> Optional[float]:
    """Mean of valid CHIRPS cells whose cell-centre falls inside the bbox."""
    if not data:
        return None
    import io

    buf = io.BytesIO(data)
    try:
        with rasterio.open(buf) as src:
            band = src.read(1, masked=True)
            transform = src.transform
            nodata = src.nodata
    except Exception as exc:
        raise GriddedUpstreamError(f"Failed to decode CHIRPS GeoTIFF: {exc}") from exc

    # Convert bbox to pixel window, then mask to cell-centres inside bbox.
    # rasterio window: (row_off, col_off) from top-left; lat decreases downward.
    try:
        window = rasterio.windows.from_bounds(
            min_lon, min_lat, max_lon, max_lat, transform
        )
    except Exception:
        return None

    row_off, col_off = int(window.row_off), int(window.col_off)
    row_size, col_size = int(window.height), int(window.width)
    if row_size <= 0 or col_size <= 0:
        return None

    sub = band[
        row_off : row_off + row_size,
        col_off : col_off + col_size,
    ]

    valid_vals = []
    for r in range(sub.shape[0]):
        for c in range(sub.shape[1]):
            cell = sub[r, c]
            if np.ma.is_masked(cell):
                continue
            val = float(cell)
            if nodata is not None and val == float(nodata):
                continue
            if val == CHIRPS_NODATA or not math.isfinite(val):
                continue
            valid_vals.append(val)

    if not valid_vals:
        return None
    return float(np.mean(valid_vals))


def fetch_chirps_monthly(
    *,
    lat: float,
    lon: float,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    area_mean: bool = False,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> GriddedSeries:
    """Fetch CHIRPS v3 monthly precipitation for a point or area.

    Args:
        lat, lon: Query point (used for point extraction; for area-mean the
            centre of the requested region is reported but the bbox governs).
        start_year, start_month, end_year, end_month: inclusive month range.
        area_mean: if True, average all valid cells within ``bbox``;
            otherwise nearest-cell to (lat, lon).
        bbox: (min_lon, min_lat, max_lon, max_lat) required when area_mean=True.

    Returns:
        GriddedSeries with one value per month (None = missing/fill).

    Raises:
        GriddedUnavailable: if (lat, lon) is outside CHIRPS coverage
            (60°S–60°N).
    """
    _validate_month_range(start_year, start_month, end_year, end_month)

    if not (CHIRPS_LAT_MIN <= lat <= CHIRPS_LAT_MAX and -180.0 <= lon <= 180.0):
        raise GriddedUnavailable(
            f"CHIRPS v3 covers 60°S–60°N (all longitudes); requested "
            f"({lat}, {lon}) is outside its footprint."
        )

    col, row = _coords_to_grid_index(lon, lat, CHIRPS_N_LON, CHIRPS_N_LAT, CHIRPS_RES_DEG)

    dates: list[str] = []
    values: list[Optional[float]] = []
    fetched_bytes = 0

    for year, month in _month_iter(
        start_year, start_month, end_year, end_month
    ):
        dates.append(f"{year:04d}-{month:02d}")
        data = _fetch_chirps_geotiff(year, month)
        if data:
            fetched_bytes += len(data)
        if area_mean:
            if bbox is None:
                raise ValueError("bbox is required when area_mean=True")
            val = _read_chirps_area_mean(data, *bbox)
        else:
            val = _read_chirps_cell(data, col, row)
        values.append(val)

    source_url = (
        f"{CHIRPS_BASE_URL.rstrip('/')}/chirps-v3.0.{{YYYY}}.{{MM}}.tif "
        f"(example {_chirps_url(2024, 6)})"
    )

    return GriddedSeries(
        dates=dates,
        values=values,
        units="mm/month",
        dataset="CHIRPS v3",
        version="3.0",
        spatial_resolution="0.05 deg",
        temporal_resolution="monthly",
        provenance="satellite_observation",
        source_url=source_url,
        coverage_note=(
            "Quasi-global land rainfall: 60°S–60°N, all longitudes, 0.05° "
            "grid, monthly totals in mm. Blends CHIRP satellite estimates with "
            "in-situ gauge stations — not a pure satellite product. "
            "Point extraction = nearest grid cell; area-mean = mean of valid "
            "cells whose cell-centre falls inside the bbox. Final month lands "
            "~third week of the following month."
        ),
    )
