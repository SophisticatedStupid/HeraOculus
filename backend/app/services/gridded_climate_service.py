"""Gridded climate / reanalysis rainfall service.



Provides normalized monthly time series from public gridded products:



* CHIRPS v3 (Climate Hazards Center) â€” keyless HTTP GeoTIFF, 0.05Â°, mm/month

* GPM IMERG Final (GPM_3IMERGM V07B, GES DISC) â€” NASA Earthdata login, 0.1Â°, mm/hr

* ERA5-Land (Copernicus CDS / ARCO Zarr mirror) â€” CDS key, 0.1Â°, metres

* SMAP L4 (SPL4SMGP V008, NSIDC / Earthdata Cloud) â€” NASA Earthdata login,

  9 km EASE-Grid 2.0, mÂ³/mÂ³ volumetric soil moisture



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
import io
import json

import logging

import math

import os


import re

import tempfile

import time

from dataclasses import dataclass, field

from datetime import date

from pathlib import Path

from typing import Optional



logger = logging.getLogger(__name__)



import numpy as np

import rasterio

import requests

from pyproj import CRS, Transformer

from rasterio.features import geometry_mask

from rasterio.transform import from_origin, rowcol

from shapely.geometry import box



from .auth_store import auth_store

from ..config import settings



# ── Cache ───────────────────────────────────────────────────────────────────

# Resolved lazily on first cache write so importing this module never crashes
# on a read-only filesystem.  None => not yet resolved; caching disabled if no
# writable location can be found anywhere.
_CACHE_DIR: Optional[Path] = None
_CACHE_DISABLED = False

# Cache format version.  Bumped when the on-disk format changes.
# Old files without this prefix are ignored/deleted on first access.
_CACHE_VERSION = "v2"

# Cache envelope magic bytes: "HRC2" = HeraOculus Cache v2
_CACHE_MAGIC = b"HRC2"
# Envelope type codes
_CACHE_TYPE_RAW = b"b"   # raw binary blob (GeoTIFF, HDF5, etc.)
_CACHE_TYPE_NPY = b"n"   # numpy array saved with np.save(allow_pickle=False)
_CACHE_TYPE_JSON = b"j"  # JSON-serializable object (dict/list)

import threading
_CACHE_LOCK = threading.Lock()


def _resolve_cache_dir() -> Optional[Path]:
    """Resolve a writable cache directory lazily on first cache write.

    Candidate locations, in order:
      1. ``HERAOCULUS_CACHE_DIR`` env var, if set
      2. ``<repo>/.cache/gridded`` next to this package
      3. ``tempfile.gettempdir()/heraoculus_cache``

    The first location that can be created *and* written to wins.  If none can,
    caching is disabled entirely and a warning is logged; the service keeps
    serving requests without caching rather than crashing on a read-only
    filesystem.

    Also purges stale pre-version cache files (*.bin without v2 prefix) on
    first successful resolution.
    """
    global _CACHE_DIR, _CACHE_DISABLED
    if _CACHE_DISABLED:
        return None
    if _CACHE_DIR is not None:
        return _CACHE_DIR

    env_dir = os.getenv("HERAOCULUS_CACHE_DIR", "").strip()
    candidates = [
        Path(env_dir) if env_dir else None,
        Path(__file__).resolve().parents[2] / ".cache" / "gridded",
        Path(tempfile.gettempdir()) / "heraoculus_cache",
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_bytes(b"")
            probe.unlink()
            _CACHE_DIR = candidate
            # Purge old-format files on first successful resolution.
            _purge_stale_cache_files(_CACHE_DIR)
            return _CACHE_DIR
        except OSError:
            continue

    _CACHE_DISABLED = True
    logger.warning(
        "Could not create a writable cache directory under %s; "
        "gridded-climate upstream caching is disabled (requests still served "
        "uncached).",
        candidates,
    )
    return None


def _purge_stale_cache_files(cache_dir: Path) -> None:
    """Delete pre-version cache files (*.bin without v2 prefix)."""
    try:
        for f in cache_dir.glob("*.bin"):
            if not f.name.startswith(_CACHE_VERSION + "_"):
                try:
                    f.unlink()
                except OSError:
                    pass
    except OSError:
        pass


def _cache_key(dataset: str, payload: str) -> str:
    """Return a versioned cache filename for the given dataset/payload."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{_CACHE_VERSION}_{dataset}_{digest}.bin"


def _envelope_wrap(typ: bytes, payload: bytes) -> bytes:
    """Wrap payload in a versioned envelope with magic + type + length + data."""
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("payload too large for cache envelope")
    length = len(payload).to_bytes(4, "big")
    return _CACHE_MAGIC + typ + length + payload


def _envelope_unwrap(data: bytes) -> Optional[bytes]:
    """Unwrap envelope, returning payload or None if corrupt/wrong version."""
    if len(data) < 1 + 4 + 4:  # magic(4) + type(1) + len(4) = 9 minimum
        return None
    if data[:4] != _CACHE_MAGIC:
        return None
    typ = data[4:5]
    length = int.from_bytes(data[5:9], "big")
    payload = data[9:9 + length]
    if len(payload) != length:
        return None
    # Optionally verify type is known
    if typ not in (_CACHE_TYPE_RAW, _CACHE_TYPE_NPY, _CACHE_TYPE_JSON):
        return None
    return payload


def _enforce_size_cap(cache_dir: Path) -> None:
    """Enforce total cache size cap by evicting LRU and expired files.

    Reads HERAOCULUS_CACHE_MAX_MB (default 512 MB).  Deletes expired (TTL)
    files first, then evicts least-recently-used files until under cap.
    """
    try:
        max_mb = int(os.getenv("HERAOCULUS_CACHE_MAX_MB", "512"))
    except ValueError:
        max_mb = 512
    max_bytes = max_mb * 1024 * 1024

    now = time.time()
    files: list[tuple[float, Path, int]] = []  # (mtime, path, size)
    total = 0
    try:
        for f in cache_dir.glob(f"{_CACHE_VERSION}_*.bin"):
            try:
                st = f.stat()
                total += st.st_size
                files.append((st.st_mtime, f, st.st_size))
            except OSError:
                # File vanished or unreadable; ignore.
                pass
    except OSError:
        return

    if total <= max_bytes:
        return

    # Sort by mtime ascending (oldest first = LRU)
    files.sort(key=lambda x: x[0])

    # First pass: delete expired files (older than DEFAULT_TTL_SECONDS).
    # Use a reasonable TTL — 7 days default, but we just evict by size anyway.
    # We'll just evict LRU until under cap.
    for mtime, f, size in files:
        if total <= max_bytes:
            break
        try:
            f.unlink()
            total -= size
        except OSError:
            pass


# Default TTL: 24 hours.  Monthly products change at most once per month, so
# a day is short enough to stay fresh while sparing upstream on repeats.

DEFAULT_TTL_SECONDS = 24 * 60 * 60


def _cache_get(dataset: str, payload: str, ttl_seconds: int = DEFAULT_TTL_SECONDS):
    """Retrieve a cached value.

    Returns the deserialized value: ``bytes`` for raw blobs, ``np.ndarray``
    for cached NumPy arrays, or ``dict``/``list`` for JSON payloads.
    Returns ``None`` on miss, expiry, corruption, or if caching is disabled.
    A corrupt or unreadable cache file is treated as a cache miss (logged at
    warning and deleted), never as a request failure.
    """
    cache_dir = _resolve_cache_dir()
    if cache_dir is None:
        return None

    path = cache_dir / _cache_key(dataset, payload)
    if not path.exists():
        return None

    try:
        st = path.stat()
        if time.time() - st.st_mtime > ttl_seconds:
            # Expired — delete to free space.
            try:
                path.unlink()
            except OSError:
                pass
            return None

        raw = path.read_bytes()
    except OSError:
        return None

    payload = _envelope_unwrap(raw)
    if payload is None:
        # Corrupt or old format — treat as miss and delete.
        logger.warning(
            "Cache file %s is corrupt or old format; deleting.", path
        )
        try:
            path.unlink()
        except OSError:
            pass
        return None

    typ = raw[4:5]
    try:
        if typ == _CACHE_TYPE_RAW:
            return payload
        elif typ == _CACHE_TYPE_NPY:
            return np.load(io.BytesIO(payload), allow_pickle=False)
        elif typ == _CACHE_TYPE_JSON:
            return json.loads(payload.decode("utf-8"))
    except Exception as exc:  # corrupt payload inside valid envelope
        logger.warning(
            "Cache file %s failed to deserialize (%s); deleting.", path, exc
        )
        try:
            path.unlink()
        except OSError:
            pass
    return None


def _cache_set(dataset: str, payload: str, data) -> None:
    """Store a value in the cache atomically.

    Accepts ``bytes`` (raw blob), ``np.ndarray`` (saved via np.save with
    allow_pickle=False), or ``dict``/``list`` (serialized via JSON).
    Write is atomic via a temporary file + os.replace.  After writing, the
    size cap is enforced.  Cache write failures never propagate to the
    caller.
    """
    cache_dir = _resolve_cache_dir()
    if cache_dir is None:
        return

    path = cache_dir / _cache_key(dataset, payload)

    try:
        if isinstance(data, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, data, allow_pickle=False)
            payload_bytes = buf.getvalue()
            typ = _CACHE_TYPE_NPY
        elif isinstance(data, (dict, list)):
            payload_bytes = json.dumps(data).encode("utf-8")
            typ = _CACHE_TYPE_JSON
        elif isinstance(data, (bytes, bytearray)):
            payload_bytes = bytes(data)
            typ = _CACHE_TYPE_RAW
        else:
            # Fallback: treat as raw bytes via bytes().
            payload_bytes = bytes(data)
            typ = _CACHE_TYPE_RAW

        envelope = _envelope_wrap(typ, payload_bytes)
    except Exception:
        # Serialization failure must not fail the request.
        return

    try:
        # Atomic write: temp file in same dir, then replace.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_dir,
            prefix=".tmp_",
            suffix=".bin",
            delete=False,
        ) as tmp:
            tmp.write(envelope)
            tmp_path = tmp.name
        # os.replace is atomic on POSIX and Windows.
        os.replace(tmp_path, path)
    except OSError:
        # Clean up temp file on failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return
    # Enforce size cap (with lock to avoid concurrent scans).
    with _CACHE_LOCK:
        _enforce_size_cap(cache_dir)


# ── Shared helpers ──────────────────────────────────────────────────────────
# â”€â”€ Shared helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



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





# â”€â”€ Exceptions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



class GriddedUnavailable(Exception):

    """Raised when a dataset cannot be served (missing credentials or

    out-of-coverage).  Routers map this to an explicit 'unavailable'

    response â€” never synthetic data."""





class GriddedUpstreamError(Exception):

    """Upstream request failed (network / HTTP error / malformed payload)."""





# â”€â”€ Normalized result â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



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





# â”€â”€ CHIRPS v3 (Climate Hazards Center) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



CHIRPS_BASE_URL = (

    "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/"

)

CHIRPS_FILENAME_PATTERN = "chirps-v3.0.{year:04d}.{month:02d}.tif"

CHIRPS_RES_DEG = 0.05

CHIRPS_N_LON = int(360.0 / CHIRPS_RES_DEG)  # 7200

CHIRPS_N_LAT = int(120.0 / CHIRPS_RES_DEG)  # 2400 (60Â°S..60Â°N)

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

            headers={"User-Agent": "HeraOculus/2.0 (gridded-climate-service; contact: ops@example.com)"},

        )

    except requests.Timeout as exc:

        raise GriddedUpstreamError(f"CHIRPS request timed out for {year}-{month:02d}: {exc}") from exc

    except requests.RequestException as exc:

        raise GriddedUpstreamError(f"CHIRPS request failed for {year}-{month:02d}: {exc}") from exc



    if resp.status_code == 404:

        # File not yet published (near-real-time lag) â€” treat as missing month.

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

            (60Â°Sâ€“60Â°N).

    """

    _validate_month_range(start_year, start_month, end_year, end_month)



    if area_mean:

        if bbox is None:

            raise ValueError("bbox is required when area_mean=True")

        min_lon, min_lat, max_lon, max_lat = bbox

        if not (

            CHIRPS_LAT_MIN <= min_lat

            and max_lat <= CHIRPS_LAT_MAX

            and -180.0 <= min_lon

            and max_lon <= 180.0

        ):

            raise GriddedUnavailable(

                f"CHIRPS v3 covers 60Â°Sâ€“60Â°N (all longitudes); requested bbox "

                f"latitude extent [{min_lat}, {max_lat}] is outside its footprint."

            )

    else:

        if lat is None or lon is None:

            raise ValueError("lat and lon are required for point extraction.")

        if not (CHIRPS_LAT_MIN <= lat <= CHIRPS_LAT_MAX and -180.0 <= lon <= 180.0):

            raise GriddedUnavailable(

                f"CHIRPS v3 covers 60Â°Sâ€“60Â°N (all longitudes); requested "

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

            "Quasi-global land rainfall: 60Â°Sâ€“60Â°N, all longitudes, 0.05Â° "

            "grid, monthly totals in mm. Blends CHIRP satellite estimates with "

            "in-situ gauge stations â€” not a pure satellite product. "

            "Point extraction = nearest grid cell; area-mean = mean of valid "

            "cells whose cell-centre falls inside the bbox. Final month lands "

            "~third week of the following month."

        ),

    )





# â”€â”€ GPM IMERG Final (GPM_3IMERGM V07B, GES DISC) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



IMERG_BASE_URL = (

    "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/"

)

IMERG_FILENAME_PATTERN = (

    "3B-MO.MS.MRG.3IMERG.{year:04d}{month:02d}01-S000000-E235959.01.V07B.HDF5"

)

IMERG_RES_DEG = 0.1

IMERG_N_LON = int(360.0 / IMERG_RES_DEG)  # 3600

IMERG_N_LAT = int(180.0 / IMERG_RES_DEG)  # 1800 (90Â°S..90Â°N)

IMERG_NODATA = -9999.0





def _imerg_earthdata_credentials() -> Optional[tuple[str, str]]:

    """Return (username, password) for NASA Earthdata if configured."""

    user, pw = auth_store.get_earthdata_credentials()

    if user or pw:

        return user, pw

    token = auth_store.get_earthdata_token()

    if token:

        # Bearer token: requests accepts it as the password with a dummy user.

        return "token", token

    return None





def _fetch_imerg_hdf5(year: int, month: int) -> bytes:

    """Download one IMERG Final monthly HDF5 (NASA Earthdata login required).



    Credentials come from the process-wide auth store only; no per-user

    secrets are accepted.  Cached on disk with TTL.

    """

    creds = _imerg_earthdata_credentials()

    if creds is None:

        raise GriddedUnavailable(

            "GPM IMERG Final requires NASA Earthdata Login credentials "

            "(EARTHDATA_USERNAME/EARTHDATA_PASSWORD or EARTHDATA_TOKEN) which "

            "are not configured on this backend."

        )



    payload = f"imerg_v07b_monthly_{year:04d}_{month:02d}"

    cached = _cache_get("imerg", payload)

    if cached is not None:

        return cached



    url = IMERG_BASE_URL + f"{year:04d}/{month:02d}/" + IMERG_FILENAME_PATTERN.format(

        year=year, month=month

    )

    user, pw = creds

    extra_headers = {

        "User-Agent": "HeraOculus/2.0 (gridded-climate-service; contact: ops@example.com)"

    }

    if pw and user == "token":

        extra_headers["Authorization"] = f"Bearer {pw}"

    try:

        resp = requests.get(

            url,

            timeout=settings.http_timeout_seconds,

            auth=(user, pw) if pw else None,

            headers=extra_headers,

        )

    except requests.Timeout as exc:

        raise GriddedUpstreamError(f"IMERG request timed out for {year}-{month:02d}: {exc}") from exc

    except requests.RequestException as exc:

        raise GriddedUpstreamError(f"IMERG request failed for {year}-{month:02d}: {exc}") from exc



    if resp.status_code == 401 or resp.status_code == 403:

        raise GriddedUnavailable(

            f"IMERG Earthdata credentials rejected (HTTP {resp.status_code}); "

            f"verify EARTHDATA_USERNAME/EARTHDATA_PASSWORD or EARTHDATA_TOKEN."

        )

    if resp.status_code == 404:

        return b""

    if resp.status_code >= 400:

        raise GriddedUpstreamError(

            f"IMERG returned HTTP {resp.status_code} for {year}-{month:02d}"

        )



    data = resp.content

    _cache_set("imerg", payload, data)

    return data





def _read_imerg_precipitation(data: bytes) -> np.ndarray:

    """Read the ``precipitation`` field (mm/hr) from an IMERG HDF5 byte blob.



    Writes the blob to a temp file and opens it with netCDF4.  IMERG HDF5 is

    not guaranteed to be netCDF4-formatted, so a failure here is surfaced as

    GriddedUpstreamError (install h5py/xarray to read pure HDF5 IMERG files).

    """

    import tempfile



    if not data:

        raise GriddedUpstreamError("Empty IMERG payload (file not published yet).")

    try:

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:

            tmp.write(data)

            tmp_path = tmp.name

        try:

            ds = netCDF4.Dataset(tmp_path, "r")

        except Exception as exc:

            raise GriddedUpstreamError(

                f"Could not open IMERG HDF5 with netCDF4: {exc}. "

                f"Install h5py/xarray to read pure-HDF5 IMERG files."

            ) from exc

        try:

            # IMERG V07 stores the monthly rate under 'precipitation'.

            precip_var = None

            for name in ("precipitation", "precipitationCal"):

                if name in ds.variables:

                    precip_var = ds.variables[name]

                    break

            if precip_var is None:

                raise GriddedUpstreamError(

                    "IMERG HDF5 has no 'precipitation' / 'precipitationCal' field."

                )

            arr = np.array(precip_var[:], dtype=np.float64)

            fill = getattr(precip_var, "_FillValue", None)

            if fill is not None:

                arr = np.where(arr == float(fill), np.nan, arr)

            return arr

        finally:

            ds.close()

    finally:

        try:

            os.unlink(tmp_path)

        except OSError:

            pass





def _coords_to_imerg_index(lon: float, lat: float) -> tuple[int, int]:

    """Nearest IMERG cell (col, row) on the 0.1Â° global grid."""

    col = int(round((lon + 180.0) / IMERG_RES_DEG))

    row = int(round((90.0 - lat) / IMERG_RES_DEG))

    col = max(0, min(IMERG_N_LON - 1, col))

    row = max(0, min(IMERG_N_LAT - 1, row))

    return col, row





def _imerg_cell_value(arr: np.ndarray, col: int, row: int) -> Optional[float]:

    try:

        val = float(arr[row, col])

    except IndexError:

        return None

    if not math.isfinite(val) or val == IMERG_NODATA:

        return None

    return val





def _imerg_area_mean(

    arr: np.ndarray, min_lon: float, min_lat: float, max_lon: float, max_lat: float

) -> Optional[float]:

    """Mean of valid IMERG cells whose cell-centre falls inside the bbox."""

    c0 = int(math.floor((min_lon + 180.0) / IMERG_RES_DEG))

    c1 = int(math.ceil((max_lon + 180.0) / IMERG_RES_DEG))

    r0 = int(math.floor((90.0 - max_lat) / IMERG_RES_DEG))

    r1 = int(math.ceil((90.0 - min_lat) / IMERG_RES_DEG))

    c0 = max(0, c0); c1 = min(IMERG_N_LON, c1)

    r0 = max(0, r0); r1 = min(IMERG_N_LAT, r1)

    if c1 <= c0 or r1 <= r0:

        return None

    sub = arr[r0:r1, c0:c1]

    valid = sub[np.isfinite(sub)]

    if valid.size == 0:

        return None

    return float(np.mean(valid))





def fetch_imerg_monthly(

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

    """Fetch GPM IMERG Final V07B monthly precipitation for a point or area.



    Returns values in **mm/month** (IMERG stores mm/hr; multiplied by the

    number of hours in each month).  Raises GriddedUnavailable when Earthdata

    credentials are not configured.

    """

    _validate_month_range(start_year, start_month, end_year, end_month)



    if area_mean:

        if bbox is None:

            raise ValueError("bbox is required when area_mean=True")

        min_lon, min_lat, max_lon, max_lat = bbox

        if not (

            -90.0 <= min_lat and max_lat <= 90.0

            and -180.0 <= min_lon and max_lon <= 180.0

        ):

            raise GriddedUnavailable(

                "IMERG covers 90Â°Sâ€“90Â°N (all longitudes); requested bbox "

                f"latitude extent [{min_lat}, {max_lat}] is outside its footprint."

            )

    else:

        if lat is None or lon is None:

            raise ValueError("lat and lon are required for point extraction.")

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):

            raise GriddedUnavailable(

                f"IMERG covers 90Â°Sâ€“90Â°N (all longitudes); requested "

                f"({lat}, {lon}) is outside its footprint."

            )



    dates: list[str] = []

    values: list[Optional[float]] = []



    for year, month in _month_iter(start_year, start_month, end_year, end_month):

        dates.append(f"{year:04d}-{month:02d}")

        data = _fetch_imerg_hdf5(year, month)

        try:

            arr = _read_imerg_precipitation(data)

        except GriddedUpstreamError:

            values.append(None)

            continue

        hours_in_month = 24 * __import__("calendar").monthrange(year, month)[1]

        if area_mean:

            val = _imerg_area_mean(arr, *bbox)

        else:

            col, row = _coords_to_imerg_index(lon, lat)

            val = _imerg_cell_value(arr, col, row)

        if val is not None:

            val = round(val * hours_in_month, 4)  # mm/hr -> mm/month

        values.append(val)



    source_url = (

        f"{IMERG_BASE_URL.rstrip('/')}/{{YYYY}}/{{MM}}/3B-MO.MS.MRG.3IMERG.{{YYYY}}{{MM}}01-..."

        f".V07B.HDF5  (example {IMERG_BASE_URL}2024/06/3B-MO.MS.MRG.3IMERG.20240601-S000000-E235959.01.V07B.HDF5)"

    )



    return GriddedSeries(

        dates=dates,

        values=values,

        units="mm/month",

        dataset="GPM IMERG Final",

        version="V07B",

        spatial_resolution="0.1 deg",

        temporal_resolution="monthly",

        provenance="satellite_observation",

        source_url=source_url,

        coverage_note=(

            "Global satelliteâ€“gauge precipitation, 0.1Â°Ã—0.1Â° grid, monthly. "

            "Stored as a rate (mm/hr) and converted to mm/month (Ã— hours in "

            "the month). Requires NASA Earthdata Login (free). "

            "Point extraction = nearest grid cell; area-mean = mean of valid "

            "cells whose cell-centre falls inside the bbox. Final product has "

            "~3.5â€“4 month latency; V07 Final ran through 2025-09."

        ),

    )





# ── ERA5-Land (Copernicus CDS / ARCO Zarr mirror) ───────────────────────────

ERA5_LAND_ARCO_ZARR = (
    "https://arco.datastores.ecmwf.int/cadl-arco-geo-009/"
    "arco/reanalysis_era5_land/sfc-pressure-precipitation/geoChunked.zarr"
)

ERA5_LAND_RES_DEG = 0.1
ERA5_LAND_M_TO_MM = 1000.0

_EARTH_FULL_CIRCLE = 360.0
_EARTH_HALF_CIRCLE = 180.0


def _convert_lon_0to360(lon: float) -> float:
    """Convert −180°..180° longitude to 0°..360° for Zarr selection."""
    return (lon + _EARTH_FULL_CIRCLE) % _EARTH_FULL_CIRCLE


def _convert_lon_360to0(lon: float) -> float:
    """Convert 0°..360° longitude back to −180°..180° for output."""
    return (lon + _EARTH_HALF_CIRCLE) % _EARTH_FULL_CIRCLE - _EARTH_HALF_CIRCLE


def _bbox_slices_0to360(
    min_lon_0: float, min_lat: float, max_lon_0: float, max_lat: float,
) -> dict:
    """Return xarray .sel() slice dict for a bbox in 0°..360° longitude.

    Handles antimeridian crossing: if the bbox crosses lon=0/360, the
    longitude slice is split into two parts.
    """
    min_lon = min_lon_0 % _EARTH_FULL_CIRCLE
    max_lon = max_lon_0 % _EARTH_FULL_CIRCLE

    crosses_antimeridian = (
        min_lon_0 < -_EARTH_HALF_CIRCLE and max_lon_0 > _EARTH_HALF_CIRCLE
    ) or (min_lon_0 > _EARTH_HALF_CIRCLE and max_lon_0 < -_EARTH_HALF_CIRCLE) or (
        min_lon_0 <= max_lon_0 and not (min_lon <= max_lon)
    ) or (min_lon_0 > max_lon_0 and not (max_lon >= min_lon))

    if crosses_antimeridian:
        min_ln = min(min_lon, max_lon)
        max_ln = max(min_lon, max_lon)
        return {
            "lat": slice(min_lat, max_lat),
            "lon": [slice(min_ln, _EARTH_FULL_CIRCLE), slice(0, max_ln)],
        }
    else:
        if min_lon <= max_lon:
            return {"lat": slice(min_lat, max_lat), "lon": slice(min_lon, max_lon)}
        else:
            return {
                "lat": slice(min_lat, max_lat),
                "lon": [slice(min_lon, _EARTH_FULL_CIRCLE), slice(0, max_lon)],
            }


def _era5_land_credentials() -> Optional[str]:
    """Return a CDS API key if configured (CDS_API_KEY or CDSAPI_KEY)."""
    return auth_store.get_cds_credentials()


def _fetch_era5_land_tp(year: int, month: int) -> np.ndarray:
    """Fetch ERA5-Land total precipitation for one month.

    Returns a 2-D numpy array of monthly total precipitation (metres) with
    shape (n_lat, n_lon), computed by summing the de-accumulated hourly
    `total_precipitation` over the month.

    Raises GriddedUnavailable if no CDS API key is configured, or
    GriddedUpstreamError if the Zarr read fails.
    """
    token = _era5_land_credentials()
    if token is None:
        raise GriddedUnavailable(
            "ERA5-Land requires a Copernicus CDS API key (CDS_API_KEY or "
            "CDSAPI_KEY) which is not configured on this backend. "
            "Set the CDS_API_KEY environment variable, or choose a keyless "
            "dataset. Only CDS API key authentication is valid for the "
            "ARCO Zarr mirror."
        )

    cache_key = f"era5_land_monthly_{year}_{month:02d}"
    cached = _cache_get("era5-land", cache_key)
    if cached is not None:
        return cached


    try:
        import xarray as xr  # noqa: import out here to keep import light
    except ImportError as exc:
        raise GriddedUpstreamError(
            "xarray is required to read ERA5-Land but is not installed."
        ) from exc

    storage_options = {"headers": {"Authorization": f"Bearer {token}"}}

    try:
        ds = xr.open_zarr(
            ERA5_LAND_ARCO_ZARR,
            consolidated=True,
            storage_options=storage_options,
        )
    except Exception as exc:
        raise GriddedUpstreamError(
            f"ERA5-Land ARCO Zarr read failed: {exc}. "
            f"Install 'zarr' and 'fsspec' to read the ARCO mirror, "
            f"or configure CDS_API_KEY."
        ) from exc

    try:
        if "total_precipitation" not in ds.variables:
            raise GriddedUpstreamError(
                "ERA5-Land Zarr has no 'total_precipitation' variable. "
                "The 'tp' variable is not used; the ARCO Zarr stores "
                "de-accumulated hourly total_precipitation in metres."
            )

        time_sel = ds["total_precipitation"].sel(
            time=f"{year:04d}-{month:02d}", method="nearest"
        )
        arr = np.asarray(time_sel.values, dtype=np.float64)
        # Sum over the time axis to obtain monthly total per cell (metres).
        if arr.ndim >= 3:
            arr = np.nansum(arr, axis=-1)
        # Ensure monotonic latitude (south-to-north: -90 -> 90).
        lat_vals = np.asarray(time_sel.coords["latitude"])
        if lat_vals.size > 1 and lat_vals[0] > lat_vals[-1]:
            arr = arr[::-1, ...]

        try:
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            _cache_set("era5-land", cache_key, buf.getvalue())
        except OSError:
            pass  # cache write failure must never fail the request

        return arr

    finally:
        try:
            ds.close()
        except Exception:
            pass


def _era5_area_mean_indices(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
) -> tuple[slice, list[slice]]:
    """Return (row_slice, col_slices) selecting cells inside a lon/lat bbox.

    Row slices follow the latitude index (90° at top). Column slices follow
    the 0°..360° longitude index; if the bbox crosses lon=0/360, two column
    slices are returned.
    """
    row_max = int(round((90.0 - min_lat) / ERA5_LAND_RES_DEG))
    row_min = int(round((90.0 - max_lat) / ERA5_LAND_RES_DEG))
    row_min = max(0, row_min)
    row_max = max(row_min, row_max)
    row_slice = slice(row_min, row_max + 1)

    lon_min_0 = _convert_lon_0to360(min_lon)
    lon_max_0 = _convert_lon_0to360(max_lon)

    if lon_min_0 <= lon_max_0:
        col_min = max(0, int(round(lon_min_0 / ERA5_LAND_RES_DEG)))
        col_max = int(round(lon_max_0 / ERA5_LAND_RES_DEG))
        col_slices = [slice(col_min, col_max + 1)]
    else:
        # Crosses antimeridian (0/360).
        col_min1 = max(0, int(round(lon_min_0 / ERA5_LAND_RES_DEG)))
        col_slices = [slice(col_min1, None), slice(0, int(round(lon_max_0 / ERA5_LAND_RES_DEG)) + 1)]

    return row_slice, col_slices


def fetch_era5_land_monthly(
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
    """Fetch ERA5-Land total precipitation for a point or area.

    Returns values in **mm/month** (ERA5-Land stores metres; the ARCO Zarr
    `total_precipitation` variable is de-accumulated to hourly resolution;
    monthly total = sum of hourly values × 1000).

    Point extraction uses the nearest grid cell to (lat, lon). Area-mean averages
    valid cells whose cell-centre falls inside the bbox. Longitude convention:
    query coords −180°..180° are converted to 0°..360° for Zarr selection;
    results are converted back to −180°..180°.

    Raises GriddedUnavailable when no CDS API key is configured, or when the
    query is outside ERA5-Land coverage (90°S–90°N land only).
    ERA5-Land ARCO access is a Beta service — rate-limited and subject to change.
    """
    _validate_month_range(start_year, start_month, end_year, end_month)

    if area_mean:
        if bbox is None:
            raise ValueError("bbox is required when area_mean=True")
        min_lon, min_lat, max_lon, max_lat = bbox
        if not (
            -90.0 <= min_lat and max_lat <= 90.0
            and -180.0 <= min_lon and max_lon <= 180.0
        ):
            raise GriddedUnavailable(
                "ERA5-Land covers 90°S–90°N (land only); requested bbox "
                f"latitude extent [{min_lat}, {max_lat}] is outside its footprint."
            )
    else:
        if lat is None or lon is None:
            raise ValueError("lat and lon are required for point extraction.")
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise GriddedUnavailable(
                f"ERA5-Land covers 90°S–90°N (all longitudes); requested "
                f"({lat}, {lon}) is outside its footprint."
            )

    dates: list[str] = []
    values: list[Optional[float]] = []

    # Date-range cap: ERA5-Land ARCO Zarr per-month fetch is expensive globally.
    # Cap at 24 months (2 years) to bound upstream cost.
    span_months = (end_year - start_year) * 12 + (end_month - start_month) + 1
    if span_months > 24:
        raise ValueError(
            "ERA5-Land date range exceeds 24-month cap "
            f"({span_months} months). "
            f"Requested {start_year:04d}-{start_month:02d} to "
            f"{end_year:04d}-{end_month:02d}. Reduce the date range and "
            "resubmit."
        )

    for year, month in _month_iter(start_year, start_month, end_year, end_month):
        dates.append(f"{year:04d}-{month:02d}")
        try:
            arr = _fetch_era5_land_tp(year, month)
        except GriddedUnavailable:
            raise
        except GriddedUpstreamError:
            values.append(None)
            continue

        if area_mean:
            assert bbox is not None
            row_slice, col_slices = _era5_area_mean_indices(
                min_lon, min_lat, max_lon, max_lat
            )
            subs = [arr[row_slice, cs] for cs in col_slices]
            sub = np.concatenate(subs, axis=1) if len(subs) > 1 else subs[0]
            valid = sub[np.isfinite(sub)]
            if valid.size == 0:
                val = None
            else:
                val = round(float(np.mean(valid)) * ERA5_LAND_M_TO_MM, 4)
        else:
            # Point extraction: nearest grid cell to (lat, lon)
            col, row = _coords_to_grid_index(
                lon, lat, arr.shape[1], arr.shape[0], ERA5_LAND_RES_DEG,
            )
            cell_metres = arr[row, col]
            if not np.isfinite(cell_metres):
                val = None
            else:
                val = round(float(cell_metres) * ERA5_LAND_M_TO_MM, 4)

        values.append(val)

    source_url = (
        f"{ERA5_LAND_ARCO_ZARR}  (ARCO Zarr mirror; CDS API key authenticated; "
        f"CDS API fallback: https://cds.climate.copernicus.eu/datasets/"
        f"reanalysis-era5-land)"
    )

    return GriddedSeries(
        dates=dates,
        values=values,
        units="mm/month",
        dataset="ERA5-Land",
        version="1.0",
        spatial_resolution="0.1 deg",
        temporal_resolution="monthly",
        provenance="reanalysis_model",
        source_url=source_url,
        coverage_note=(
            "ECMWF reanalysis land variables, 0.1° grid, monthly totals. "
            "`total_precipitation` in the ARCO Zarr is de-accumulated to hourly "
            "resolution; monthly total = sum of hourly values × 1000 (m → mm). "
            "Land-only; ocean cells are fill and reported as missing. Updated "
            "daily, ~5 days behind real time. Requires a CDS API key "
            "(CDS_API_KEY environment variable). ARCO access is Beta — rate-limited "
            "and subject to change. Point extraction = nearest grid cell; "
            "area-mean = mean of valid cells whose cell-centre falls inside the "
            "bbox."
        ),
    )

# â”€â”€ SMAP L4 (SPL4SMGP V008, NSIDC / Earthdata Cloud) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



SMAP_BASE_URL = (

    "https://n5eil02u.ecs.nsidc.org/smap/SPL4SMGP.008/"

)

SMAP_RES_DEG_APPROX = 0.1  # 9 km EASE-Grid â‰ˆ 0.1Â° at equator (used for coverage note only)

SMAP_LAT_MIN = -85.044

SMAP_LAT_MAX = 85.044

SMAP_NODATA = -9999.0

# SMAP L4 volumetric soil moisture is already mÂ³/mÂ³ (0..1); no conversion needed.

SMAP_M3M3_TO_M3M3 = 1.0





def _smap_earthdata_credentials() -> Optional[tuple[str, str]]:

    """Return (username, password) or (token_user, token) for NASA Earthdata."""

    user, pw = auth_store.get_earthdata_credentials()

    if user or pw:

        return user, pw

    token = auth_store.get_earthdata_token()

    if token:

        return "token", token

    return None





def _smap_granule_urls(

    year: int, month: int

) -> list[tuple[str, str]]:

    """Return [(datetime_iso, url)] for every 3-hourly SMAP L4 granule in a month.



    SMAP L4 granules are named ``SPL4SMGP.L4_RootZone_9km.YYYYMMDDThhmmss.v8.0.h5``

    and published under the NSIDC HTTPS File System at

    ``n5eil02u.ecs.nsidc.org/smap/SPL4SMGP.008/YYYY.MM.DD/``.  There are 8

    granules per day (00:30, 03:30, â€¦, 21:30).

    """

    import calendar



    urls: list[tuple[str, str]] = []

    days_in_month = calendar.monthrange(year, month)[1]

    for day in range(1, days_in_month + 1):

        date_dir = f"{year:04d}.{month:02d}.{day:02d}"

        date_str = f"{year:04d}{month:02d}{day:02d}"

        for hour in range(0, 24, 3):

            dt_iso = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:30:00Z"

            fname = (

                f"SPL4SMGP.L4_RootZone_9km.{date_str}T{hour:02d}3000.v8.0.h5"

            )

            url = f"{SMAP_BASE_URL}{date_dir}/{fname}"

            urls.append((dt_iso, url))

    return urls





def _fetch_smap_hdf5(url: str) -> bytes:

    """Download one SMAP L4 HDF5 granule (NASA Earthdata login required).



    Credentials come from the process-wide auth store only.  Cached on disk

    with TTL.

    """

    creds = _smap_earthdata_credentials()

    if creds is None:

        raise GriddedUnavailable(

            "SMAP L4 requires NASA Earthdata Login credentials "

            "(EARTHDATA_USERNAME/EARTHDATA_PASSWORD or EARTHDATA_TOKEN) which "

            "are not configured on this backend."

        )



    payload = f"smap_l4_monthly_{url}"

    cached = _cache_get("smap", payload)

    if cached is not None:

        return cached



    user, pw = creds

    extra_headers = {

        "User-Agent": "HeraOculus/2.0 (gridded-climate-service; contact: ops@example.com)"

    }

    if pw and user == "token":

        extra_headers["Authorization"] = f"Bearer {pw}"

    try:

        resp = requests.get(

            url,

            timeout=settings.http_timeout_seconds,

            auth=(user, pw) if pw else None,

            headers=extra_headers,

        )

    except requests.Timeout as exc:

        raise GriddedUpstreamError(f"SMAP request timed out for {url}: {exc}") from exc

    except requests.RequestException as exc:

        raise GriddedUpstreamError(f"SMAP request failed for {url}: {exc}") from exc



    if resp.status_code == 401 or resp.status_code == 403:

        raise GriddedUnavailable(

            f"SMAP Earthdata credentials rejected (HTTP {resp.status_code}); "

            f"verify EARTHDATA_USERNAME/EARTHDATA_PASSWORD or EARTHDATA_TOKEN."

        )

    if resp.status_code == 404:

        return b""

    if resp.status_code >= 400:

        raise GriddedUpstreamError(f"SMAP returned HTTP {resp.status_code} for {url}")



    data = resp.content

    _cache_set("smap", payload, data)

    return data





def _read_smap_hdf5(

    data: bytes, variable: str

) -> dict[str, np.ndarray]:

    """Read SMAP L4 HDF5 granule.



    Returns dict with ``sm`` (2-D float array of the requested variable),

    ``cell_lat`` and ``cell_lon`` (2-D EASE-grid cell-centre coordinates).

    Raises GriddedUpstreamError if the granule cannot be decoded.

    """

    import h5py

    import tempfile



    if not data:

        raise GriddedUpstreamError("Empty SMAP payload (granule not published yet).")

    try:

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:

            tmp.write(data)

            tmp_path = tmp.name

        try:

            with h5py.File(tmp_path, "r") as ds:

                geo = ds["Geophysical_Data"]

                if variable not in geo:

                    raise GriddedUpstreamError(

                        f"SMAP HDF5 has no Geophysical_Data/{variable} field."

                    )

                sm = np.array(geo[variable][:], dtype=np.float64)

                cell_lat = np.array(geo["cell_lat"][:], dtype=np.float64)

                cell_lon = np.array(geo["cell_lon"][:], dtype=np.float64)

                fill = geo[variable].attrs.get("_FillValue")

                if fill is not None:

                    sm = np.where(sm == float(fill), np.nan, sm)

                return {"sm": sm, "cell_lat": cell_lat, "cell_lon": cell_lon}

        finally:

            try:

                os.unlink(tmp_path)

            except OSError:

                pass

    except GriddedUnavailable:

        raise

    except Exception as exc:  # h5py / file errors

        raise GriddedUpstreamError(f"Failed to decode SMAP HDF5: {exc}") from exc





def _smap_nearest_cell_index(

    cell_lat: np.ndarray,

    cell_lon: np.ndarray,

    lat: float,

    lon: float,

) -> Optional[tuple[int, int]]:

    """Nearest EASE-grid cell (row, col) to a lat/lon query point."""

    valid = np.isfinite(cell_lat) & np.isfinite(cell_lon)

    if not np.any(valid):

        return None

    dlat = cell_lat - lat

    dlon = cell_lon - lon

    dist2 = np.where(valid, dlat * dlat + dlon * dlon, np.inf)

    flat_idx = int(np.argmin(dist2))

    if not np.isfinite(dist2.flat[flat_idx]):

        return None

    return int(flat_idx // cell_lat.shape[1]), int(flat_idx % cell_lat.shape[1])





def _smap_cell_value(

    sm: np.ndarray, row: int, col: int

) -> Optional[float]:

    try:

        val = float(sm[row, col])

    except IndexError:

        return None

    if not math.isfinite(val) or val == SMAP_NODATA:

        return None

    # SMAP volumetric soil moisture is physically bounded 0..1.

    if val < 0.0 or val > 1.0:

        return None

    return val





def _smap_area_mean(

    sm: np.ndarray,

    cell_lat: np.ndarray,

    cell_lon: np.ndarray,

    min_lon: float,

    min_lat: float,

    max_lon: float,

    max_lat: float,

) -> Optional[float]:

    """Mean of valid EASE-grid cells whose cell-centre falls inside the bbox."""

    in_box = (

        np.isfinite(sm)

        & np.isfinite(cell_lat)

        & np.isfinite(cell_lon)

        & (cell_lat >= min_lat)

        & (cell_lat <= max_lat)

        & (cell_lon >= min_lon)

        & (cell_lon <= max_lon)

        & (sm >= 0.0)

        & (sm <= 1.0)

    )

    valid = sm[in_box]

    if valid.size == 0:

        return None

    return float(np.mean(valid))





def fetch_smap_monthly(

    *,

    lat: float,

    lon: float,

    start_year: int,

    start_month: int,

    end_year: int,

    end_month: int,

    variable: str = "sm_rootzone",

    area_mean: bool = False,

    bbox: Optional[tuple[float, float, float, float]] = None,

) -> GriddedSeries:

    """Fetch SMAP L4 volumetric soil moisture for a point or area.



    Args:

        lat, lon: Query point (point extraction uses the nearest EASE-grid

            cell to this coordinate).

        start_year, start_month, end_year, end_month: inclusive month range.

        variable: ``sm_rootzone`` (0â€“100 cm), ``sm_surface`` (0â€“5 cm), or

            ``sm_profile`` (total profile).

        area_mean: if True, average all valid EASE-grid cells whose cell-centre

            falls inside ``bbox``; otherwise nearest-cell to (lat, lon).

        bbox: (min_lon, min_lat, max_lon, max_lat) required when area_mean=True.



    Returns:

        GriddedSeries with one value per month (None = missing/fill).

        Values are unitless volumetric soil moisture (mÂ³/mÂ³).



    Raises:

        GriddedUnavailable: if Earthdata credentials are not configured, the

            variable is unknown, or the query is outside SMAP coverage

            (85Â°Sâ€“85Â°N, land).

    """

    _validate_month_range(start_year, start_month, end_year, end_month)



    if variable not in ("sm_rootzone", "sm_surface", "sm_profile"):

        raise GriddedUnavailable(

            f"SMAP variable '{variable}' not supported. "

            f"Allowed: sm_rootzone, sm_surface, sm_profile."

        )



    if area_mean:

        if bbox is None:

            raise ValueError("bbox is required when area_mean=True")

        min_lon, min_lat, max_lon, max_lat = bbox

        if not (

            SMAP_LAT_MIN <= min_lat and max_lat <= SMAP_LAT_MAX

            and -180.0 <= min_lon and max_lon <= 180.0

        ):

            raise GriddedUnavailable(

                "SMAP L4 covers 85Â°Sâ€“85Â°N (global land); requested bbox "

                f"latitude extent [{min_lat}, {max_lat}] is outside its footprint."

            )

    else:

        if lat is None or lon is None:

            raise ValueError("lat and lon are required for point extraction.")

        if not (SMAP_LAT_MIN <= lat <= SMAP_LAT_MAX and -180.0 <= lon <= 180.0):

            raise GriddedUnavailable(

                f"SMAP L4 covers 85Â°Sâ€“85Â°N (global land); requested "

                f"({lat}, {lon}) is outside its footprint."

            )



    dates: list[str] = []

    values: list[Optional[float]] = []



    for year, month in _month_iter(start_year, start_month, end_year, end_month):

        dates.append(f"{year:04d}-{month:02d}")

        granule_vals: list[float] = []

        try:

            urls = _smap_granule_urls(year, month)

        except GriddedUnavailable:

            raise

        for _dt_iso, url in urls:

            try:

                data = _fetch_smap_hdf5(url)

                if not data:

                    continue

                arrays = _read_smap_hdf5(data, variable)

            except GriddedUpstreamError:

                continue

            sm = arrays["sm"]

            cell_lat = arrays["cell_lat"]

            cell_lon = arrays["cell_lon"]

            if area_mean:

                val = _smap_area_mean(

                    sm, cell_lat, cell_lon, min_lon, min_lat, max_lon, max_lat

                )

            else:

                idx = _smap_nearest_cell_index(cell_lat, cell_lon, lat, lon)

                if idx is None:

                    val = None

                else:

                    val = _smap_cell_value(sm, idx[0], idx[1])

            if val is not None:

                granule_vals.append(round(val * SMAP_M3M3_TO_M3M3, 4))

        if granule_vals:

            values.append(round(float(np.mean(granule_vals)), 4))

        else:

            values.append(None)



    variable_label = {

        "sm_rootzone": "Root-zone soil moisture (0â€“100 cm)",

        "sm_surface": "Surface soil moisture (0â€“5 cm)",

        "sm_profile": "Profile soil moisture",

    }[variable]



    source_url = (

        f"{SMAP_BASE_URL.rstrip('/')}/{ '{}'.format('YYYY.MM.DD')}/"

        f"SPL4SMGP.L4_RootZone_9km.YYYYMMDDThhmmss.v8.0.h5  "

        f"(NSIDC HTTPS File System; example granule "

        f"{SMAP_BASE_URL}2015.03.31/SPL4SMGP.L4_RootZone_9km.20150331T003000.v8.0.h5)"

    )



    return GriddedSeries(

        dates=dates,

        values=values,

        units="m3/m3",

        dataset="SMAP L4",

        version="V008",

        spatial_resolution="9 km EASE-Grid 2.0",

        temporal_resolution="monthly",

        provenance="satellite_observation",

        source_url=source_url,

        coverage_note=(

            "SMAP Level-4 surface/root-zone soil moisture, 9 km EASE-Grid 2.0, "

            "global land (85Â°Sâ€“85Â°N), 3-hourly, 2015â€“present (~7-day latency "

            "for L4). Values are volumetric soil moisture (mÂ³/mÂ³). "

            "Root-zone SM is a GEOS land-model estimate assimilating SMAP "

            "brightness temperatures, not a direct observation. Monthly value = "

            "mean of valid 3-hourly granules in the month. Point extraction = "

            "nearest EASE-grid cell; area-mean = mean of valid cells whose "

            "cell-centre falls inside the bbox. Requires NASA Earthdata Login."

        ),

    )





# â”€â”€ MODIS LST Night (MOD11A1 V061, LAADS / NASA Earthdata) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



MODIS_BASE_URL = (

    "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD11A1/"

)

MODIS_RES_DEG_APPROX = 0.1  # 1 km at equator (~0.009Â°) â€” noted for coverage

MODIS_LAT_MIN = -90.0

MODIS_LAT_MAX = 90.0

MODIS_NODATA = 0

MODIS_LST_NIGHT_SCALE = 0.02  # LST_K = raw * 0.02

MODIS_LST_NIGHT_VALID_MIN = 7500

MODIS_LST_NIGHT_VALID_MAX = 65535

# MODIS sinusoidal tile grid: 36 columns (h00-h35), 18 rows (v00-v17),

# each tile ~10 deg x 10 deg at the equator.

MODIS_H_COLS = 36

MODIS_V_ROWS = 18

MODIS_LST_NIGHT_SDS = "LST_Night_1km"



# MODIS Sinusoidal (SIN) projection parameters (authoritative, NASA):

#   +proj=sinu +lon_0=0 +x_0=0 +y_0=0

#   sphere radius a = b = 6371007.181 m, units = m

MODIS_SINU_PROJ4 = (

    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 "

    "+b=6371007.181 +units=m +no_defs"

)

MODIS_SINU_CRS = CRS.from_proj4(MODIS_SINU_PROJ4)

MODIS_SINU_R = 6371007.181  # metres (MODIS sphere)

MODIS_PI = math.pi

# Projected extent of the full sinusoidal globe:

#   x = R * lon_rad        (lon_0 = 0)  -> [-R*pi, R*pi] for [-180, 180]

#   y = R * lat_rad        -> [-R*pi/2, R*pi/2] for [-90, 90]

MODIS_X_MIN = -MODIS_SINU_R * MODIS_PI

MODIS_X_MAX = MODIS_SINU_R * MODIS_PI

MODIS_Y_MIN = -MODIS_SINU_R * MODIS_PI / 2.0

MODIS_Y_MAX = MODIS_SINU_R * MODIS_PI / 2.0

# Tiles are equal squares in projected space (not in geographic space;

# geographic tile shape distorts with latitude).

MODIS_TILE_W = (MODIS_X_MAX - MODIS_X_MIN) / MODIS_H_COLS

MODIS_TILE_H = (MODIS_Y_MAX - MODIS_Y_MIN) / MODIS_V_ROWS





def _modis_tile_indices(lat: float, lon: float) -> tuple[str, str]:

    """Return (h_tile, v_tile) zero-padded MODIS sinusoidal tile indices.



    The geographic coordinate is transformed into the MODIS Sinusoidal

    projected coordinate system (authoritative NASA grid) and the tile is

    located in projected space.  This correctly accounts for projection

    distortion: tiles are equal squares in projected metres but have

    different geographic shapes at different latitudes (they are ~10 deg x

    10 deg only at the equator and narrow toward the poles).



    h00 covers the westernmost column, h35 the easternmost; v00 covers the

    northernmost row, v17 the southernmost.

    """

    # lon/lat (EPSG:4326) -> MODIS sinusoidal projected (x, y) in metres.

    transformer = Transformer.from_crs("EPSG:4326", MODIS_SINU_CRS, always_xy=True)

    x, y = transformer.transform(lon, lat)



    # Clamp to the valid projected globe extent (poles / antimeridian edge).

    x = min(max(x, MODIS_X_MIN), MODIS_X_MAX)

    y = min(max(y, MODIS_Y_MIN), MODIS_Y_MAX)



    h = int(math.floor((x - MODIS_X_MIN) / MODIS_TILE_W))

    v = int(math.floor((MODIS_Y_MAX - y) / MODIS_TILE_H))

    h = max(0, min(MODIS_H_COLS - 1, h))

    v = max(0, min(MODIS_V_ROWS - 1, v))

    return f"h{h:02d}", f"v{v:02d}"





def _modis_tile_transform(

    h_tile: str, v_tile: str, n_rows: int, n_cols: int

) -> tuple:

    """Return the affine transform and CRS for a MODIS sinusoidal tile.



    The tile's projected extent is derived from its (h, v) indices in the

    authoritative sinusoidal grid; the pixel size follows from the array

    dimensions.  North-up, no rotation.

    """

    h_idx = int(h_tile[1:])

    v_idx = int(v_tile[1:])

    x_min = MODIS_X_MIN + h_idx * MODIS_TILE_W

    y_max = MODIS_Y_MAX - v_idx * MODIS_TILE_H

    pixel_w = MODIS_TILE_W / n_cols

    pixel_h = MODIS_TILE_H / n_rows

    transform = from_origin(x_min, y_max, pixel_w, pixel_h)

    return transform, MODIS_SINU_CRS





def _proj_bounds_of_bbox(

    min_lon: float, min_lat: float, max_lon: float, max_lat: float, crs

) -> tuple[float, float, float, float]:

    """Projected (xmin, ymin, xmax, ymax) bounds of a lon/lat AOI bbox.



    The AOI edges are densely sampled before transformation so that projection

    curvature (e.g. the sinusoidal collapse of meridians at the poles) is

    captured.  Transforming only the four corners is insufficient: at high

    latitude all corners can project to nearly the same x while the box edges

    span the full projected width.

    """

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    n = 65

    lon_edge = np.linspace(min_lon, max_lon, n)

    lat_edge = np.linspace(min_lat, max_lat, n)

    xs: list[float] = []

    ys: list[float] = []

    for lon in lon_edge:

        xs.append(lon)

        xs.append(lon)

        ys.append(min_lat)

        ys.append(max_lat)

    for lat in lat_edge:

        xs.append(min_lon)

        xs.append(max_lon)

        ys.append(lat)

        ys.append(lat)

    tx, ty = transformer.transform(xs, ys)

    return float(min(tx)), float(min(ty)), float(max(tx)), float(max(ty))





def _modis_area_mean_masked(

    arr: np.ndarray,

    transform,

    crs,

    min_lon: float,

    min_lat: float,

    max_lon: float,

    max_lat: float,

) -> Optional[float]:

    """Mean of valid pixels whose cell-centre falls inside the lon/lat AOI.



    The AOI bounding box is transformed from EPSG:4326 into the raster CRS

    and a spatial mask is constructed (cell-centre inclusion policy), so only

    pixels intersecting the requested AOI are averaged.  Non-finite and

    NoData/masked pixels are excluded.  Returns None when no valid AOI pixel

    exists.

    """

    arr = np.asarray(arr, dtype=np.float64)



    # Transform the AOI into the raster CRS (edge-densified for curvature).

    try:

        aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y = _proj_bounds_of_bbox(

            min_lon, min_lat, max_lon, max_lat, crs

        )

    except Exception:

        return None



    # Rasterize the AOI into a cell-centre inclusion mask in raster CRS.

    try:

        aoi_geom = box(aoi_min_x, aoi_min_y, aoi_max_x, aoi_max_y)

        in_aoi = geometry_mask(

            [aoi_geom],

            transform=transform,

            out_shape=arr.shape,

            invert=True,

            all_touched=False,

        )

    except Exception:

        return None



    # Validity: finite AND inside the AOI.

    with np.errstate(invalid="ignore"):

        finite = np.isfinite(arr)

    valid = finite & in_aoi

    if not np.any(valid):

        return None

    return float(np.mean(arr[valid]))





_MODIS_GRANULE_RE = re.compile(

    r"MOD11A1\.A(\d{7})\.h\d{2}v\d{2}\.061\.(\d{13})\.hdf"

)





def _modis_daily_granule_urls(

    year: int, month: int, h_tile: str, v_tile: str

) -> list[tuple[str, str]]:

    """List MOD11A1 daily granule URLs for one month and tile.



    Fetches the LAADS directory index for each day of the month and extracts

    granule filenames matching ``MOD11A1.AYYYYDDD.hXXvYY.061.*.hdf`` for the

    requested tile.  Returns ``[(date_iso, url), ...]`` sorted chronologically.

    """

    import calendar



    urls: list[tuple[str, str]] = []

    days_in_month = calendar.monthrange(year, month)[1]

    for day in range(1, days_in_month + 1):

        doy = f"{day:03d}"

        date_str = f"{year:04d}{month:02d}{day:02d}"

        dir_url = f"{MODIS_BASE_URL}{year:04d}/{doy}/"

        try:

            resp = requests.get(

                dir_url,

                timeout=settings.http_timeout_seconds,

                headers={

                    "User-Agent": "HeraOculus/2.0 (gridded-climate-service; contact: ops@example.com)"

                },

            )

        except requests.RequestException:

            continue

        if resp.status_code != 200:

            continue

        html = resp.text

        want_tile = f".{h_tile}{v_tile}."

        for match in _MODIS_GRANULE_RE.finditer(html):

            granule_date = match.group(1)

            if granule_date != date_str:

                continue

            filename = match.group(0)

            if want_tile not in filename:

                continue

            urls.append((f"{year:04d}-{month:02d}-{day:02d}", dir_url + filename))

    urls.sort(key=lambda item: item[0])

    return urls





def _modis_earthdata_credentials() -> Optional[tuple[str, str]]:

    """Return (username, password) or (token_user, token) for NASA Earthdata."""

    user, pw = auth_store.get_earthdata_credentials()

    if user or pw:

        return user, pw

    token = auth_store.get_earthdata_token()

    if token:

        return "token", token

    return None





def _fetch_modis_hdf(url: str) -> bytes:

    """Download one MOD11A1 HDF-EOS2 granule (NASA Earthdata login required).



    Cached on disk with TTL.

    """

    creds = _modis_earthdata_credentials()

    if creds is None:

        raise GriddedUnavailable(

            "MODIS LST requires NASA Earthdata Login credentials "

            "(EARTHDATA_USERNAME/EARTHDATA_PASSWORD or EARTHDATA_TOKEN) which "

            "are not configured on this backend."

        )



    payload = f"modis_lst_night_{url}"

    cached = _cache_get("modis_lst_night", payload)

    if cached is not None:

        return cached



    user, pw = creds

    extra_headers = {

        "User-Agent": "HeraOculus/2.0 (gridded-climate-service; contact: ops@example.com)"

    }

    if pw and user == "token":

        extra_headers["Authorization"] = f"Bearer {pw}"

    try:

        resp = requests.get(

            url,

            timeout=settings.http_timeout_seconds,

            auth=(user, pw) if pw else None,

            headers=extra_headers,

        )

    except requests.Timeout as exc:

        raise GriddedUpstreamError(f"MODIS request timed out for {url}: {exc}") from exc

    except requests.RequestException as exc:

        raise GriddedUpstreamError(f"MODIS request failed for {url}: {exc}") from exc



    if resp.status_code == 401 or resp.status_code == 403:

        raise GriddedUnavailable(

            f"MODIS Earthdata credentials rejected (HTTP {resp.status_code}); "

            f"verify EARTHDATA_USERNAME/EARTHDATA_PASSWORD or EARTHDATA_TOKEN."

        )

    if resp.status_code == 404:

        return b""

    if resp.status_code >= 400:

        raise GriddedUpstreamError(f"MODIS returned HTTP {resp.status_code} for {url}")



    data = resp.content

    _cache_set("modis_lst_night", payload, data)

    return data





def _read_modis_lst_night(data: bytes) -> np.ndarray:

    """Read LST_Night_1km SDS from a MOD11A1 HDF-EOS2 granule.



    Returns a 2-D float array in Kelvin (raw * 0.02).  Invalid/fill cells are

    NaN.  Uses pyhdf to read HDF4-EOS2.

    """

    import tempfile



    from pyhdf.SD import SD, SDC



    if not data:

        raise GriddedUpstreamError("Empty MODIS payload (granule not published yet).")

    try:

        with tempfile.NamedTemporaryFile(suffix=".hdf", delete=False) as tmp:

            tmp.write(data)

            tmp_path = tmp.name

        try:

            hdf = SD(tmp_path, SDC.READ)

            try:

                if MODIS_LST_NIGHT_SDS not in hdf.datasets():

                    raise GriddedUpstreamError(

                        f"MODIS HDF-EOS2 has no SDS '{MODIS_LST_NIGHT_SDS}'."

                    )

                sds = hdf.select(MODIS_LST_NIGHT_SDS)

                raw = np.array(sds.get(), dtype=np.float64)

                attrs = sds.attributes()

                scale = float(

                    attrs.get("scale_factor", attrs.get("ScaleFactor", 1.0))

                )

                fill = attrs.get(

                    "_FillValue",

                    attrs.get("fill_value", attrs.get("_FillValue", MODIS_NODATA)),

                )

            finally:

                hdf.end()

        finally:

            try:

                os.unlink(tmp_path)

            except OSError:

                pass

    except GriddedUnavailable:

        raise

    except Exception as exc:  # pyhdf / file errors

        raise GriddedUpstreamError(f"Failed to decode MODIS HDF-EOS2: {exc}") from exc



    arr = raw * scale

    # Mask fill values and out-of-valid-range observations.

    invalid = (

        ~np.isfinite(arr)

        | (arr == float(fill))

        | (raw < MODIS_LST_NIGHT_VALID_MIN)

        | (raw > MODIS_LST_NIGHT_VALID_MAX)

    )

    arr[invalid] = np.nan

    return arr





def _modis_cell_value(

    arr: np.ndarray, row: int, col: int

) -> Optional[float]:

    try:

        val = float(arr[row, col])

    except IndexError:

        return None

    if not math.isfinite(val):

        return None

    return val





def _modis_area_mean(

    arr: np.ndarray,

    min_lon: float,

    min_lat: float,

    max_lon: float,

    max_lat: float,

    *,

    transform=None,

    crs=None,

    h_tile: Optional[str] = None,

    v_tile: Optional[str] = None,

) -> Optional[float]:

    """Mean of valid MODIS cells whose cell-centre falls inside the AOI bbox.



    The requested AOI (lon/lat) is transformed into the raster CRS and a

    spatial mask is constructed so only pixels intersecting the AOI are

    averaged.  The tile georeferencing is reconstructed from the MODIS

    sinusoidal (h, v) indices (authoritative projected grid); when no

    georeferencing can be resolved the function falls back to averaging all

    finite cells (legacy behaviour).



    NoData / fill and non-finite cells are excluded; the statistic is

    computed only over valid AOI pixels.

    """

    arr = np.asarray(arr, dtype=np.float64)

    n_rows, n_cols = arr.shape



    if transform is None:

        if h_tile is None or v_tile is None:

            # No georeferencing available: fall back to all valid cells.

            valid = arr[np.isfinite(arr)]

            if valid.size == 0:

                return None

            return float(np.mean(valid))

        transform, crs = _modis_tile_transform(h_tile, v_tile, n_rows, n_cols)

    if crs is None:

        crs = MODIS_SINU_CRS



    return _modis_area_mean_masked(arr, transform, crs, min_lon, min_lat, max_lon, max_lat)





def fetch_lst_night_monthly(

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

    """Fetch MODIS LST_Night_1km (Terra MOD11A1 V061) for a point or area.



    MOD11A1 is daily; the monthly value is the mean of valid daily

    LST_Night_1km observations in the month.  Returns values in Kelvin.



    Args:

        lat, lon: Query point (point extraction uses the nearest cell in the

            MODIS sinusoidal tile covering the point).

        start_year, start_month, end_year, end_month: inclusive month range.

        area_mean: if True, average all valid cells within ``bbox``;

            otherwise nearest-cell to (lat, lon).

        bbox: (min_lon, min_lat, max_lon, max_lat) required when area_mean=True.



    Returns:

        GriddedSeries with one value per month (None = missing/fill).

        Values are land-surface temperature in Kelvin.



    Raises:

        GriddedUnavailable: if Earthdata credentials are not configured or the

            query is outside MODIS coverage (global).

    """

    _validate_month_range(start_year, start_month, end_year, end_month)



    if area_mean:

        if bbox is None:

            raise ValueError("bbox is required when area_mean=True")

        min_lon, min_lat, max_lon, max_lat = bbox

        if not (

            MODIS_LAT_MIN <= min_lat and max_lat <= MODIS_LAT_MAX

            and -180.0 <= min_lon and max_lon <= 180.0

        ):

            raise GriddedUnavailable(

                "MODIS LST covers 90Â°Sâ€“90Â°N (all longitudes); requested bbox "

                f"latitude extent [{min_lat}, {max_lat}] is outside its footprint."

            )

    else:

        if lat is None or lon is None:

            raise ValueError("lat and lon are required for point extraction.")

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):

            raise GriddedUnavailable(

                f"MODIS LST covers 90Â°Sâ€“90Â°N (all longitudes); requested "

                f"({lat}, {lon}) is outside its footprint."

            )



    h_tile, v_tile = _modis_tile_indices(lat, lon)



    # Fail fast on missing credentials before touching the network (directory

    # listings).  SMAP checks credentials inside _fetch_smap_hdf5, but MODIS

    # needs to list directory URLs first, so we check up front here.

    if _modis_earthdata_credentials() is None:

        raise GriddedUnavailable(

            "MODIS LST requires NASA Earthdata Login credentials "

            "(EARTHDATA_USERNAME/EARTHDATA_PASSWORD or EARTHDATA_TOKEN) which "

            "are not configured on this backend."

        )



    dates: list[str] = []

    values: list[Optional[float]] = []



    for year, month in _month_iter(start_year, start_month, end_year, end_month):

        dates.append(f"{year:04d}-{month:02d}")

        granule_vals: list[float] = []

        try:

            urls = _modis_daily_granule_urls(year, month, h_tile, v_tile)

        except GriddedUnavailable:

            raise

        for _date_iso, url in urls:

            try:

                data = _fetch_modis_hdf(url)

                if not data:

                    continue

                arr = _read_modis_lst_night(data)

            except GriddedUpstreamError:

                continue

            if area_mean:

                val = _modis_area_mean(

                    arr, min_lon, min_lat, max_lon, max_lat,

                    h_tile=h_tile, v_tile=v_tile,

                )

            else:

                # Nearest cell to (lat, lon) within the tile.  Transform the

                # query point into the tile's sinusoidal projected CRS and

                # locate the containing pixel via the tile affine transform,

                # correctly handling projection distortion (tiles are not

                # uniform 10-degree blocks in geographic space).

                transform, _tile_crs = _modis_tile_transform(

                    h_tile, v_tile, arr.shape[0], arr.shape[1]

                )

                _proj = Transformer.from_crs(

                    "EPSG:4326", MODIS_SINU_CRS, always_xy=True

                )

                px, py = _proj.transform(lon, lat)

                row, col = rowcol(transform, px, py)

                row = int(max(0, min(arr.shape[0] - 1, int(row))))

                col = int(max(0, min(arr.shape[1] - 1, int(col))))

                val = _modis_cell_value(arr, row, col)

            if val is not None:

                granule_vals.append(round(val, 4))

        if granule_vals:

            values.append(round(float(np.mean(granule_vals)), 4))

        else:

            values.append(None)



    source_url = (

        f"{MODIS_BASE_URL.rstrip('/')}/{ '{}'.format('YYYY')}/{ '{}'.format('DDD')}/"

        f"MOD11A1.AYYYYDDD.hXXvYY.061.YYYYDDDHHMMSS.hdf  "

        f"(LAADS HTTPS File System; Terra MOD11A1 V061, LST_Night_1km SDS)"

    )



    return GriddedSeries(

        dates=dates,

        values=values,

        units="K",

        dataset="MODIS MOD11A1",

        version="V061",

        spatial_resolution="1 km",

        temporal_resolution="monthly",

        provenance="satellite_observation",

        source_url=source_url,

        coverage_note=(

            "MODIS/Terra Land Surface Temperature & Emissivity Daily (MOD11A1 "

            "V061), 1 km sinusoidal grid, global, 2000â€“present. "

            "LST_Night_1km is stored as scaled integers (Kelvin = raw * 0.02, "

            "fill 0, valid range 7500-65535). Clear-sky only; above 30Â° "

            "latitude multiple observations may be averaged. Monthly value = "

            "mean of valid daily LST_Night observations. Point extraction = "

            "nearest cell within the MODIS sinusoidal tile; area-mean = mean "

            "of valid cells whose cell-centre falls inside the bbox. "

            "Requires NASA Earthdata Login."

        ),

    )









