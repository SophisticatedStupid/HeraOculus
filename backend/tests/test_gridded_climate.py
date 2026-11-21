"""Tests for the gridded climate service (CHIRPS first).

Upstream is mocked — no live network. Covers:
  * point (nearest-cell) extraction returns the cell value
  * area-mean extraction matches a hand-computed small grid
  * nodata / fill cells are reported as missing (None)
  * out-of-coverage (outside 60S-60N) raises GriddedUnavailable
  * malformed upstream GeoTIFF -> GriddedUpstreamError
  * cache hit returns cached bytes without re-downloading
  * router: /api/v1/gridded/chirps available; out-of-coverage -> unavailable;
    imerg / era5-land without credentials -> 400 unavailable
"""

from __future__ import annotations

import io
import pickle
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

from app.main import app
from app.services import gridded_climate_service as gcs
from app.services.gridded_climate_service import (
    GriddedSeries,
    GriddedUnavailable,
    GriddedUpstreamError,
    fetch_chirps_monthly,
)


# ── Synthetic GeoTIFF helpers ───────────────────────────────────────────────

def _make_geotiff_bytes(
    array: np.ndarray,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    nodata: float = -9999.0,
) -> bytes:
    """Write a small single-band GeoTIFF to in-memory bytes."""
    n_rows, n_cols = array.shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, n_cols, n_rows)
    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array.astype(np.float32), 1)
    return buf.getvalue()


# CHIRPS test grid: 0.05 deg over -10..10 lon, -10..10 lat (north-up).
_CHIRPS_TEST_ARRAY = np.full((400, 400), -9999.0, dtype=np.float32)
# Constant 77.0 interior with a nodata border so area-mean / cell reads are
# robust to rasterio window edge semantics (hand-computed == 77.0 exactly).
_CHIRPS_TEST_ARRAY = np.full((400, 400), -9999.0, dtype=np.float32)
_CHIRPS_TEST_ARRAY[1:399, 1:399] = 77.0
_CHIRPS_TEST_BYTES = _make_geotiff_bytes(
    _CHIRPS_TEST_ARRAY, -10.0, -10.0, 10.0, 10.0
)


def _patch_download(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    monkeypatch.setattr(gcs, "_fetch_chirps_geotiff", lambda y, m: data)


# ── Service-level tests ─────────────────────────────────────────────────────

def test_read_chirps_cell_returns_value() -> None:
    # Interior cell nearest (-2.0, 2.0) -> col=160, row=160 -> 77.0.
    col = int(round((-2.0 - (-10.0)) / 0.05))
    row = int(round((10.0 - 2.0) / 0.05))
    val = gcs._read_chirps_cell(_CHIRPS_TEST_BYTES, col, row)
    assert val == 77.0


def test_read_chirps_cell_nodata_is_none() -> None:
    # Border cell (col=0, row=0) is nodata.
    col = int(round((-10.0 - (-10.0)) / 0.05))
    row = int(round((10.0 - 10.0) / 0.05))
    assert gcs._read_chirps_cell(_CHIRPS_TEST_BYTES, col, row) is None


def test_area_mean_matches_hand_computed_grid() -> None:
    # Bbox fully inside the constant 77.0 interior -> area-mean == 77.0 exactly,
    # regardless of which edge cells rasterio's window includes.
    min_lon, min_lat, max_lon, max_lat = -2.0, 2.0, 2.0, 6.0
    val = gcs._read_chirps_area_mean(
        _CHIRPS_TEST_BYTES, min_lon, min_lat, max_lon, max_lat
    )
    assert val is not None
    assert abs(val - 77.0) < 1e-6, f"expected 77.0, got {val}"


def test_area_mean_all_fill_is_none() -> None:
    all_nodata = np.full((400, 400), -9999.0, dtype=np.float32)
    data = _make_geotiff_bytes(all_nodata, -10.0, -10.0, 10.0, 10.0)
    assert (
        gcs._read_chirps_area_mean(data, -5.0, 5.0, -1.0, 9.0) is None
    )


def test_fetch_chirps_monthly_point_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_download(monkeypatch, _CHIRPS_TEST_BYTES)
    monkeypatch.setattr(gcs, "_read_chirps_cell", lambda *a, **k: 42.0)
    series = fetch_chirps_monthly(
        lat=9.975,
        lon=-9.975,
        start_year=2024,
        start_month=1,
        end_year=2024,
        end_month=3,
    )
    assert series.dates == ["2024-01", "2024-02", "2024-03"]
    assert series.values == [42.0, 42.0, 42.0]
    assert series.units == "mm/month"
    assert series.dataset == "CHIRPS v3"
    assert series.provenance == "satellite_observation"
    assert series.valid_count == 3
    assert series.missing_count == 0


def test_fetch_chirps_monthly_out_of_coverage() -> None:
    with pytest.raises(GriddedUnavailable):
        fetch_chirps_monthly(
            lat=70.0,  # outside 60S-60N
            lon=0.0,
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
        )


def test_fetch_chirps_monthly_missing_month_is_none(monkeypatch) -> None:
    # 404 (empty bytes) -> missing month reported as None
    _patch_download(monkeypatch, b"")
    series = fetch_chirps_monthly(
        lat=0.0,
        lon=0.0,
        start_year=2024,
        start_month=1,
        end_year=2024,
        end_month=2,
    )
    assert series.values == [None, None]
    assert series.valid_count == 0
    assert series.missing_count == 2


def test_fetch_chirps_monthly_upstream_error(monkeypatch) -> None:
    # Force a cache miss so the live request path is exercised.
    monkeypatch.setattr(gcs, "_cache_get", lambda *a, **k: None)
    monkeypatch.setattr(
        gcs.requests,
        "get",
        MagicMock(side_effect=gcs.requests.Timeout("boom")),
    )
    with pytest.raises(GriddedUpstreamError):
        fetch_chirps_monthly(
            lat=0.0,
            lon=0.0,
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
        )


def test_cache_hit_avoids_redownload(monkeypatch) -> None:
    # Prime the real disk cache for Jan 2024.
    gcs._cache_set("chirps", "chirps_v3_monthly_2024_01", _CHIRPS_TEST_BYTES)

    # If the cache is bypassed, requests.get would perform a live HTTP call;
    # make it raise so any cache miss is surfaced as an error.
    monkeypatch.setattr(
        gcs.requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("live HTTP attempted")),
    )
    # Isolate the cache path from the (small-test-grid) cell indexing.
    monkeypatch.setattr(gcs, "_read_chirps_cell", lambda *a, **k: 0.0)

    series = fetch_chirps_monthly(
        lat=0.0,
        lon=0.0,
        start_year=2024,
        start_month=1,
        end_year=2024,
        end_month=1,
    )
    assert series.values[0] == 0.0


# ── Router-level tests ──────────────────────────────────────────────────────

client = TestClient(app)


def _post_gridded(dataset: str, body: dict):
    return client.post(f"/api/v1/gridded/{dataset}", json=body)


def test_gridded_chirps_available(monkeypatch) -> None:
    _patch_download(monkeypatch, _CHIRPS_TEST_BYTES)
    monkeypatch.setattr(gcs, "_read_chirps_cell", lambda *a, **k: 42.0)
    resp = _post_gridded(
        "chirps",
        {
            "variable": "precip",
            "start_date": "2024-01",
            "end_date": "2024-01",
            "lat": 9.975,
            "lon": -9.975,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "available"
    assert body["series"]["dates"] == ["2024-01"]
    assert body["series"]["values"] == [42.0]
    assert body["series"]["units"] == "mm/month"


def test_gridded_chirps_out_of_coverage_unavailable(monkeypatch) -> None:
    resp = _post_gridded(
        "chirps",
        {
            "variable": "precip",
            "start_date": "2024-01",
            "end_date": "2024-01",
            "lat": 70.0,
            "lon": 0.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unavailable"
    assert body["series"] is None
    assert "outside its footprint" in body["message"]


def test_gridded_chirps_area_mean(monkeypatch) -> None:
    _patch_download(monkeypatch, _CHIRPS_TEST_BYTES)
    monkeypatch.setattr(gcs, "_read_chirps_area_mean", lambda *a, **k: 199.5)
    resp = _post_gridded(
        "chirps",
        {
            "variable": "precip",
            "start_date": "2024-01",
            "end_date": "2024-01",
            "area_mean": True,
            "bbox": [-5.0, 5.0, -1.0, 9.0],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "available"
    assert abs(body["series"]["values"][0] - 199.5) < 1e-6


def test_gridded_imerg_missing_credentials_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(gcs.auth_store, "has_earthdata_auth", lambda: False)
    resp = _post_gridded(
        "imerg",
        {
            "variable": "precip",
            "start_date": "2024-01",
            "end_date": "2024-01",
            "lat": 0.0,
            "lon": 0.0,
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "Earthdata" in body["detail"]


def test_gridded_era5_land_missing_credentials_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("CDS_API_KEY", "")
    monkeypatch.setenv("CDSAPI_KEY", "")
    # Re-resolve: auth_store reads env at init; force env cleared then check route
    # reads os.getenv directly (not auth_store), so clearing env is enough.
    resp = _post_gridded(
        "era5-land",
        {
            "variable": "precip",
            "start_date": "2024-01",
            "end_date": "2024-01",
            "lat": 0.0,
            "lon": 0.0,
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "CDS" in body["detail"]


def test_gridded_unknown_dataset() -> None:
    resp = _post_gridded(
        "not-a-dataset",
        {
            "variable": "precip",
            "start_date": "2024-01",
            "end_date": "2024-01",
            "lat": 0.0,
            "lon": 0.0,
        },
    )
    assert resp.status_code == 400


# ── IMERG service-level tests ───────────────────────────────────────────────

_IMERG_TEST_ARRAY = np.full((1800, 3600), np.nan, dtype=np.float64)
# Constant 5.0 mm/hr interior with nan borders.
_IMERG_TEST_ARRAY[100:1790, 100:3590] = 5.0


def _patch_imerg_download(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    monkeypatch.setattr(gcs, "_fetch_imerg_hdf5", lambda y, m: data)


def _patch_imerg_read(monkeypatch: pytest.MonkeyPatch, arr: np.ndarray) -> None:
    monkeypatch.setattr(gcs, "_read_imerg_precipitation", lambda d: arr)


def test_imerg_cell_value_and_area_mean() -> None:
    col, row = gcs._coords_to_imerg_index(-2.0, 2.0)  # interior -> 5.0 mm/hr
    assert gcs._imerg_cell_value(_IMERG_TEST_ARRAY, col, row) == 5.0
    # Border cell is nan -> None.
    assert gcs._imerg_cell_value(_IMERG_TEST_ARRAY, 0, 0) is None
    # Area-mean over interior bbox -> 5.0 mm/hr.
    val = gcs._imerg_area_mean(_IMERG_TEST_ARRAY, -2.0, 2.0, 2.0, 6.0)
    assert val is not None
    assert abs(val - 5.0) < 1e-6


def test_fetch_imerg_monthly_mm_hr_to_mm_month(monkeypatch) -> None:
    _patch_imerg_download(monkeypatch, b"fake-hdf5")
    _patch_imerg_read(monkeypatch, _IMERG_TEST_ARRAY)
    # June 2024 has 30 days = 720 hours; 5.0 mm/hr * 720 = 3600.0 mm/month.
    series = gcs.fetch_imerg_monthly(
        lat=2.0,
        lon=-2.0,
        start_year=2024,
        start_month=6,
        end_year=2024,
        end_month=6,
    )
    assert series.dates == ["2024-06"]
    assert series.values == [3600.0]
    assert series.units == "mm/month"
    assert series.dataset == "GPM IMERG Final"
    assert series.provenance == "satellite_observation"
    assert series.valid_count == 1


def test_fetch_imerg_monthly_area_mean(monkeypatch) -> None:
    _patch_imerg_download(monkeypatch, b"fake-hdf5")
    _patch_imerg_read(monkeypatch, _IMERG_TEST_ARRAY)
    series = gcs.fetch_imerg_monthly(
        lat=0.0,
        lon=0.0,
        start_year=2024,
        start_month=1,
        end_year=2024,
        end_month=1,
        area_mean=True,
        bbox=[-2.0, 2.0, 2.0, 6.0],
    )
    # January 2024: 31 days = 744 hours; 5.0 * 744 = 3720.0 mm/month.
    assert series.values == [3720.0]


def test_fetch_imerg_monthly_out_of_coverage() -> None:
    # Lat 100 is outside IMERG's 90S-90N footprint; coverage check fires before
    # any credential/network logic.
    with pytest.raises(GriddedUnavailable):
        gcs.fetch_imerg_monthly(
            lat=100.0,
            lon=0.0,
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
        )


def test_fetch_imerg_monthly_missing_credentials(monkeypatch) -> None:
    monkeypatch.setattr(gcs, "_imerg_earthdata_credentials", lambda: None)
    with pytest.raises(GriddedUnavailable):
        gcs.fetch_imerg_monthly(
            lat=0.0,
            lon=0.0,
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
        )


# ── IMERG router-level tests ────────────────────────────────────────────────

def test_gridded_imerg_missing_credentials_400(monkeypatch) -> None:
    monkeypatch.setattr(gcs.auth_store, "has_earthdata_auth", lambda: False)
    monkeypatch.setattr(gcs.auth_store, "get_earthdata_token", lambda: None)
    monkeypatch.setattr(
        gcs.auth_store, "get_earthdata_credentials", lambda: (None, None)
    )
    resp = _post_gridded(
        "imerg",
        {
            "variable": "precip",
            "start_date": "2024-01",
            "end_date": "2024-01",
            "lat": 0.0,
            "lon": 0.0,
        },
    )
    assert resp.status_code == 400
    assert "Earthdata" in resp.json()["detail"]


def test_gridded_imerg_available(monkeypatch) -> None:
    # Credentials configured -> proceed; mock download + read.
    monkeypatch.setattr(gcs.auth_store, "has_earthdata_auth", lambda: True)
    monkeypatch.setattr(
        gcs.auth_store, "get_earthdata_credentials", lambda: ("user", "pass")
    )
    _patch_imerg_download(monkeypatch, b"fake-hdf5")
    _patch_imerg_read(monkeypatch, _IMERG_TEST_ARRAY)
    resp = _post_gridded(
        "imerg",
        {
            "variable": "precip",
            "start_date": "2024-06",
            "end_date": "2024-06",
            "lat": 2.0,
            "lon": -2.0,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "available"
    assert body["series"]["values"] == [3600.0]
    assert body["series"]["units"] == "mm/month"


def test_gridded_imerg_cache_hit_avoids_redownload(monkeypatch) -> None:
    monkeypatch.setattr(gcs.auth_store, "has_earthdata_auth", lambda: True)
    monkeypatch.setattr(
        gcs.auth_store, "get_earthdata_credentials", lambda: ("user", "pass")
    )
    gcs._cache_set("imerg", "imerg_v07b_monthly_2024_06", b"cached-hdf5")
    monkeypatch.setattr(
        gcs.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("live HTTP attempted"))
    )
    monkeypatch.setattr(gcs, "_read_imerg_precipitation", lambda d: _IMERG_TEST_ARRAY)
    series = gcs.fetch_imerg_monthly(
        lat=2.0, lon=-2.0, start_year=2024, start_month=6, end_year=2024, end_month=6
    )
    assert series.values[0] == 3600.0


# ── ERA5-Land service-level tests ───────────────────────────────────────────

_ERA5_TEST_ARRAY = np.full((1800, 3600), np.nan, dtype=np.float64)
# Constant 0.005 m interior -> 5.0 mm after x1000; nan borders.
_ERA5_TEST_ARRAY[100:1790, 100:3590] = 0.005


def _patch_era5_fetch(
    monkeypatch: pytest.MonkeyPatch, arr: np.ndarray
) -> None:
    monkeypatch.setattr(gcs, "_fetch_era5_land_tp", lambda y, m: arr)


def test_fetch_era5_land_m_to_mm(monkeypatch) -> None:
    # 0.005 m * 1000 = 5.0 mm/month.
    _patch_era5_fetch(monkeypatch, _ERA5_TEST_ARRAY)
    series = gcs.fetch_era5_land_monthly(
        lat=2.0,
        lon=-2.0,
        start_year=2024,
        start_month=1,
        end_year=2024,
        end_month=1,
    )
    assert series.dates == ["2024-01"]
    assert series.values == [5.0]
    assert series.units == "mm/month"
    assert series.dataset == "ERA5-Land"
    assert series.provenance == "reanalysis_model"
    assert series.valid_count == 1


def test_fetch_era5_land_area_mean(monkeypatch) -> None:
    _patch_era5_fetch(monkeypatch, _ERA5_TEST_ARRAY)
    series = gcs.fetch_era5_land_monthly(
        lat=0.0,
        lon=0.0,
        start_year=2024,
        start_month=2,
        end_year=2024,
        end_month=2,
        area_mean=True,
        bbox=[-2.0, 2.0, 2.0, 6.0],
    )
    assert series.values == [5.0]


def test_fetch_era5_land_missing_credentials(monkeypatch) -> None:
    monkeypatch.setattr(gcs, "_era5_land_credentials", lambda: None)
    with pytest.raises(GriddedUnavailable):
        gcs.fetch_era5_land_monthly(
            lat=0.0,
            lon=0.0,
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
        )


def test_fetch_era5_land_out_of_coverage() -> None:
    with pytest.raises(GriddedUnavailable):
        gcs.fetch_era5_land_monthly(
            lat=-100.0,
            lon=0.0,
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
        )


def test_fetch_era5_land_cache_hit(monkeypatch) -> None:
    # Credentials configured so the coverage/cred check passes.
    monkeypatch.setattr(gcs, "_era5_land_credentials", lambda: "token")
    # Prime the real disk cache for Jan 2024.
    gcs._cache_set(
        "era5-land",
        "era5_land_monthly_2024_01",
        pickle.dumps(_ERA5_TEST_ARRAY),
    )
    # If the cache is bypassed, the Zarr read / network would be attempted;
    # make requests.get raise so any cache miss is surfaced.
    monkeypatch.setattr(
        gcs.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("live HTTP attempted"))
    )
    # Do NOT monkeypatch _fetch_era5_land_tp: the real function should serve
    # from cache.
    series = gcs.fetch_era5_land_monthly(
        lat=2.0,
        lon=-2.0,
        start_year=2024,
        start_month=1,
        end_year=2024,
        end_month=1,
    )
    assert series.values[0] == 5.0


# ── ERA5-Land router-level tests ────────────────────────────────────────────

def test_gridded_era5_land_missing_credentials_400(monkeypatch) -> None:
    monkeypatch.setattr(gcs.auth_store, "get_cds_credentials", lambda: None)
    monkeypatch.setattr(gcs.auth_store, "get_destine_pat", lambda: None)
    resp = _post_gridded(
        "era5-land",
        {
            "variable": "precip",
            "start_date": "2024-01",
            "end_date": "2024-01",
            "lat": 0.0,
            "lon": 0.0,
        },
    )
    assert resp.status_code == 400
    assert "CDS" in resp.json()["detail"] or "DestinE" in resp.json()["detail"]


def test_gridded_era5_land_available(monkeypatch) -> None:
    monkeypatch.setattr(gcs.auth_store, "get_cds_credentials", lambda: "cds-key")
    monkeypatch.setattr(gcs.auth_store, "get_destine_pat", lambda: None)
    _patch_era5_fetch(monkeypatch, _ERA5_TEST_ARRAY)
    resp = _post_gridded(
        "era5-land",
        {
            "variable": "precip",
            "start_date": "2024-01",
            "end_date": "2024-01",
            "lat": 2.0,
            "lon": -2.0,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "available"
    assert body["series"]["values"] == [5.0]
    assert body["series"]["units"] == "mm/month"


# ── SMAP L4 service-level tests ─────────────────────────────────────────────

_SMAP_TEST_SM = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]], dtype=np.float64)
_SMAP_TEST_CELL_LAT = np.array(
    [[2.0, 2.0], [0.0, 0.0], [-2.0, -2.0]], dtype=np.float64
)
_SMAP_TEST_CELL_LON = np.array(
    [[-2.0, 0.0], [-2.0, 0.0], [-2.0, 0.0]], dtype=np.float64
)
_SMAP_TEST_ARRAYS = {
    "sm": _SMAP_TEST_SM,
    "cell_lat": _SMAP_TEST_CELL_LAT,
    "cell_lon": _SMAP_TEST_CELL_LON,
}


def _patch_smap_download(
    monkeypatch: pytest.MonkeyPatch, data: bytes = b"fake-hdf5"
) -> None:
    monkeypatch.setattr(gcs, "_fetch_smap_hdf5", lambda url: data)


def _patch_smap_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gcs, "_read_smap_hdf5", lambda d, v: _SMAP_TEST_ARRAYS
    )


def _patch_smap_urls(monkeypatch: pytest.MonkeyPatch, urls: list[tuple[str, str]]) -> None:
    monkeypatch.setattr(gcs, "_smap_granule_urls", lambda y, m: urls)


def test_smap_cell_value_nearest_neighbor() -> None:
    # Point (0.0, 0.0) -> nearest cell is cell_lat=0.0, cell_lon=0.0 (row=1,col=1) -> 0.5
    idx = gcs._smap_nearest_cell_index(
        _SMAP_TEST_CELL_LAT, _SMAP_TEST_CELL_LON, 0.0, 0.0
    )
    assert idx == (1, 1)
    assert gcs._smap_cell_value(_SMAP_TEST_SM, 1, 1) == 0.5
    # Out-of-bounds cell index is guarded by the indexer, not the reader.


def test_smap_area_mean_matches_hand_computed() -> None:
    # Bbox [-1, -1, 1, 1] contains only cell (row=1,col=1) lat=0,lon=0 -> 0.5
    # (row=1,col=0) has lon=-2, outside the box; rows 0/2 are outside lat range.
    val = gcs._smap_area_mean(
        _SMAP_TEST_SM,
        _SMAP_TEST_CELL_LAT,
        _SMAP_TEST_CELL_LON,
        -1.0, -1.0, 1.0, 1.0,
    )
    assert val is not None
    assert abs(val - 0.5) < 1e-6, f"expected 0.5, got {val}"


def test_smap_area_mean_all_fill_is_none() -> None:
    all_nan = np.full((3, 2), np.nan, dtype=np.float64)
    assert (
        gcs._smap_area_mean(all_nan, _SMAP_TEST_CELL_LAT, _SMAP_TEST_CELL_LON, -1.0, -1.0, 1.0, 1.0)
        is None
    )


def test_fetch_smap_monthly_point_extraction(monkeypatch) -> None:
    # Two granules in June 2024, both return cell value 0.5 at (0.0, 0.0).
    urls = [
        ("2024-06-01T00:30:00Z", "https://example.com/smap/2024.06.01/g1.h5"),
        ("2024-06-01T03:30:00Z", "https://example.com/smap/2024.06.01/g2.h5"),
    ]
    _patch_smap_urls(monkeypatch, urls)
    _patch_smap_download(monkeypatch)
    _patch_smap_read(monkeypatch)
    monkeypatch.setattr(gcs, "_smap_earthdata_credentials", lambda: ("user", "pass"))

    series = gcs.fetch_smap_monthly(
        lat=0.0,
        lon=0.0,
        start_year=2024,
        start_month=6,
        end_year=2024,
        end_month=6,
        variable="sm_rootzone",
    )
    assert series.dates == ["2024-06"]
    # Both granules yield 0.5 -> monthly mean 0.5 m3/m3
    assert series.values == [0.5]
    assert series.units == "m3/m3"
    assert series.dataset == "SMAP L4"
    assert series.provenance == "satellite_observation"
    assert series.valid_count == 1


def test_fetch_smap_monthly_area_mean(monkeypatch) -> None:
    urls = [("2024-06-01T00:30:00Z", "https://example.com/smap/g.h5")]
    _patch_smap_urls(monkeypatch, urls)
    _patch_smap_download(monkeypatch)
    _patch_smap_read(monkeypatch)
    monkeypatch.setattr(gcs, "_smap_earthdata_credentials", lambda: ("user", "pass"))

    series = gcs.fetch_smap_monthly(
        lat=0.0,
        lon=0.0,
        start_year=2024,
        start_month=6,
        end_year=2024,
        end_month=6,
        variable="sm_rootzone",
        area_mean=True,
        bbox=[-1.0, -1.0, 1.0, 1.0],
    )
    assert series.values == [0.5]


def test_fetch_smap_monthly_out_of_coverage() -> None:
    with pytest.raises(GriddedUnavailable):
        gcs.fetch_smap_monthly(
            lat=90.0,  # outside 85S-85N
            lon=0.0,
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
        )


def test_fetch_smap_monthly_missing_credentials(monkeypatch) -> None:
    monkeypatch.setattr(gcs, "_smap_earthdata_credentials", lambda: None)
    with pytest.raises(GriddedUnavailable):
        gcs.fetch_smap_monthly(
            lat=0.0,
            lon=0.0,
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
        )


def test_fetch_smap_monthly_unknown_variable() -> None:
    with pytest.raises(GriddedUnavailable):
        gcs.fetch_smap_monthly(
            lat=0.0,
            lon=0.0,
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
            variable="not_a_variable",
        )


def test_fetch_smap_monthly_cache_hit_avoids_redownload(monkeypatch) -> None:
    monkeypatch.setattr(gcs, "_smap_earthdata_credentials", lambda: ("user", "pass"))
    # Prime the real disk cache for the single June 2024 granule URL.
    gcs._cache_set(
        "smap",
        "smap_l4_monthly_https://example.com/smap/2024.06.01/g.h5",
        b"cached-hdf5",
    )
    monkeypatch.setattr(
        gcs.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("live HTTP attempted"))
    )
    _patch_smap_urls(
        monkeypatch, [("2024-06-01T00:30:00Z", "https://example.com/smap/2024.06.01/g.h5")]
    )
    _patch_smap_read(monkeypatch)
    series = gcs.fetch_smap_monthly(
        lat=0.0,
        lon=0.0,
        start_year=2024,
        start_month=6,
        end_year=2024,
        end_month=6,
    )
    assert series.values[0] == 0.5


# ── SMAP router-level tests ────────────────────────────────────────────────

def test_gridded_smap_missing_credentials_400(monkeypatch) -> None:
    monkeypatch.setattr(gcs.auth_store, "has_earthdata_auth", lambda: False)
    monkeypatch.setattr(gcs.auth_store, "get_earthdata_token", lambda: None)
    monkeypatch.setattr(
        gcs.auth_store, "get_earthdata_credentials", lambda: (None, None)
    )
    resp = _post_gridded(
        "smap",
        {
            "variable": "precip",
            "start_date": "2024-06",
            "end_date": "2024-06",
            "lat": 0.0,
            "lon": 0.0,
        },
    )
    assert resp.status_code == 400
    assert "Earthdata" in resp.json()["detail"]


def test_gridded_smap_available(monkeypatch) -> None:
    monkeypatch.setattr(gcs.auth_store, "has_earthdata_auth", lambda: True)
    monkeypatch.setattr(
        gcs.auth_store, "get_earthdata_credentials", lambda: ("user", "pass")
    )
    urls = [("2024-06-01T00:30:00Z", "https://example.com/smap/g.h5")]
    _patch_smap_urls(monkeypatch, urls)
    _patch_smap_download(monkeypatch)
    _patch_smap_read(monkeypatch)
    resp = _post_gridded(
        "smap",
        {
            "variable": "precip",
            "start_date": "2024-06",
            "end_date": "2024-06",
            "lat": 0.0,
            "lon": 0.0,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "available"
    assert body["series"]["values"] == [0.5]
    assert body["series"]["units"] == "m3/m3"


def test_gridded_smap_unknown_variable_400(monkeypatch) -> None:
    # router passes variable="sm_rootzone" always; unknown dataset is what 400s here.
    resp = _post_gridded(
        "not-a-dataset",
        {
            "variable": "precip",
            "start_date": "2024-01",
            "end_date": "2024-01",
            "lat": 0.0,
            "lon": 0.0,
        },
    )
    assert resp.status_code == 400



