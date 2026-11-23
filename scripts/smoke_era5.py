#!/usr/bin/env python3
"""Smoke-test the ERA5-Land connector against the live ARCO Zarr mirror.

Usage
-----
    # Backend directory (after `pip install -r requirements.txt`)
    python scripts/smoke_era5.py

The script reads ``CDS_API_KEY`` (or ``CDSAPI_KEY``) from the environment
(or the backend ``.env``), then fetches ERA5-Land total precipitation for a
small bbox over Sylhet (24.86°N, 91.86°E) for a recent month and asserts:

* the result is available (no ``GriddedUnavailable``),
* the series has at least one non-None value,
* units are ``mm/month`` (m→mm de-accumulation),
* the per-cell values are non-negative.

Exit code is non-zero on any failure so CI can pick it up (though this script
is intentionally excluded from the automated pytest run).

Only runs when invoked directly; never imported by the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the backend package is importable when run from the repo root.
_REPO = Path(__file__).resolve().parents[1]
_BACKEND = _REPO / "backend"
if _BACKEND.exists():
    sys.path.insert(0, str(_BACKEND))

from app.services.auth_store import auth_store  # noqa: E402
from app.services.gridded_climate_service import (  # noqa: E402
    GriddedUnavailable,
    fetch_era5_land_monthly,
)


def main() -> int:
    cds_key = auth_store.get_cds_credentials()
    if not cds_key:
        print(
            "FAIL: no CDS_API_KEY/CDSAPI_KEY configured. "
            "Set CDS_API_KEY in the backend .env or environment.",
            file=sys.stderr,
        )
        return 2

    # Sylhet small bbox: (min_lon, min_lat, max_lon, max_lat) in degrees.
    bbox = (91.8, 24.8, 91.9, 24.9)
    lat = 24.86
    lon = 91.86
    start_year, start_month = 2023, 9
    end_year, end_month = 2023, 9

    try:
        point = fetch_era5_land_monthly(
            lat=lat, lon=lon,
            start_year=start_year, start_month=start_month,
            end_year=end_year, end_month=end_month,
        )
    except GriddedUnavailable as exc:
        print(f"FAIL: ERA5-Land unavailable: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: ERA5-Land fetch raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    try:
        area = fetch_era5_land_monthly(
            lat=lat, lon=lon,
            start_year=start_year, start_month=start_month,
            end_year=end_year, end_month=end_month,
            area_mean=True,
            bbox=bbox,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: ERA5-Land area-mean raised {type(exc).__name__}: {exc}", file=sys.stderr)
        return 5

    ok = True
    for label, series in (("point", point), ("area-mean", area)):
        if not series.dates:
            print(f"FAIL: {label} returned no dates", file=sys.stderr)
            ok = False
            continue
        if not any(v is not None for v in series.values):
            print(f"FAIL: {label} values are all None", file=sys.stderr)
            ok = False
            continue
        if series.units != "mm/month":
            print(f"FAIL: {label} units={series.units!r} (expected 'mm/month')", file=sys.stderr)
            ok = False
        for d, v in zip(series.dates, series.values):
            if v is not None and v < 0:
                print(f"FAIL: {label} negative precip at {d}: {v}", file=sys.stderr)
                ok = False

    if not ok:
        return 1

    print("OK: ERA5-Land smoke test passed")
    for label, series in (("point", point), ("area-mean", area)):
        print(f"  {label}: {list(zip(series.dates, series.values))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())