"""API integration tests for Raster Interpolation & Gap-Filling endpoints."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from app.main import app
from app.security import limiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Reset slowapi storage around each test so rate-limit counters never leak
    between tests now that endpoints are rate limited."""
    limiter.reset()
    yield
    limiter.reset()


def _make_geotiff_bytes(
    array: np.ndarray,
    *,
    crs: str = "EPSG:32610",
    transform=None,
    nodata: float | None = None,
    dtype: str | None = None,
) -> bytes:
    array = np.asarray(array)
    if transform is None:
        transform = from_origin(500000.0, 5000000.0, 10.0, 10.0)
    write_dtype = dtype or array.dtype.name
    fd, path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    try:
        with rasterio.open(
            path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
            count=1, dtype=write_dtype, crs=crs, transform=transform, nodata=nodata,
        ) as dst:
            dst.write(array.astype(write_dtype), 1)
        with open(path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


def _make_hole_array() -> np.ndarray:
    arr = np.full((10, 10), 50.0, dtype=float)
    arr[3:7, 3:7] = np.nan
    return arr


def _array_to_payload(arr: np.ndarray) -> list:
    return [[None if np.isnan(v) else float(v) for v in row] for row in arr]


# ---------------------------------------------------------------------------
# Inspect endpoint
# ---------------------------------------------------------------------------

def test_inspect_endpoint():
    arr = _make_hole_array()
    b = _make_geotiff_bytes(arr, nodata=-9999.0)
    resp = client.post(
        "/api/v1/rasters/inspect",
        files={"file": ("test.tif", b, "image/tiff")},
        data={"band_index": "1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["width"] == 10
    assert data["height"] == 10
    assert data["valid_pixel_count"] == 84
    assert data["missing_pixel_count"] == 16
    assert data["missing_percentage"] == 16.0
    assert data["raster_type"] == "continuous"


def test_inspect_endpoint_invalid_file():
    resp = client.post(
        "/api/v1/rasters/inspect",
        files={"file": ("test.txt", b"not a geotiff", "text/plain")},
        data={"band_index": "1"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Compatible methods endpoint
# ---------------------------------------------------------------------------

def test_compatible_methods_endpoint():
    resp = client.get("/api/v1/rasters/interpolate/compatible-methods?raster_type=continuous")
    assert resp.status_code == 200
    data = resp.json()
    assert data["raster_type"] == "continuous"
    ids = {m["id"] for m in data["methods"] if m["compatible"]}
    assert "nearest_neighbor" in ids
    assert "bilinear" in ids
    assert "ordinary_kriging" in ids
    assert "temporal_linear" not in ids


def test_compatible_methods_categorical():
    resp = client.get("/api/v1/rasters/interpolate/compatible-methods?raster_type=categorical")
    assert resp.status_code == 200
    data = resp.json()
    ids = {m["id"] for m in data["methods"] if m["compatible"]}
    assert ids == {"nearest_neighbor"}


# ---------------------------------------------------------------------------
# Interpolate endpoint
# ---------------------------------------------------------------------------

def test_interpolate_endpoint_nearest_neighbor():
    arr = _make_hole_array()
    payload = {
        "data": _array_to_payload(arr),
        "method": "nearest_neighbor",
        "raster_type": "continuous",
        "band_index": 1,
    }
    resp = client.post("/api/v1/rasters/interpolate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "nearest_neighbor"
    assert data["interpolated_pixel_count"] == 16
    assert data["remaining_nodata_count"] == 0
    assert data["observed_pixel_count"] == 84
    # Provenance mask should have observed + interpolated classes
    prov = data["provenance_mask"]
    observed = sum(1 for row in prov for v in row if v == "observed")
    interpolated = sum(1 for row in prov for v in row if v == "interpolated")
    assert observed == 84
    assert interpolated == 16
    # Reconstructed array should have no nulls where there were holes
    recon = data["reconstructed"]
    for i in range(3, 7):
        for j in range(3, 7):
            assert recon[i][j] is not None


def test_interpolate_endpoint_idw():
    arr = _make_hole_array()
    payload = {
        "data": _array_to_payload(arr),
        "method": "idw",
        "raster_type": "continuous",
        "band_index": 1,
        "power": 2.0,
    }
    resp = client.post("/api/v1/rasters/interpolate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["interpolated_pixel_count"] == 16


def test_interpolate_endpoint_kriging():
    arr = _make_hole_array()
    payload = {
        "data": _array_to_payload(arr),
        "method": "ordinary_kriging",
        "raster_type": "continuous",
        "band_index": 1,
    }
    resp = client.post("/api/v1/rasters/interpolate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["interpolated_pixel_count"] == 16
    assert data["kriging_variance"] is not None


def test_interpolate_endpoint_invalid_method_for_raster_type():
    arr = _make_hole_array()
    payload = {
        "data": _array_to_payload(arr),
        "method": "bilinear",
        "raster_type": "categorical",
        "band_index": 1,
    }
    resp = client.post("/api/v1/rasters/interpolate", json=payload)
    assert resp.status_code == 400


def test_interpolate_endpoint_invalid_data():
    payload = {
        "data": [[1, 2], [3]],  # ragged array
        "method": "nearest_neighbor",
        "raster_type": "continuous",
    }
    resp = client.post("/api/v1/rasters/interpolate", json=payload)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Validate endpoint
# ---------------------------------------------------------------------------

def test_validate_endpoint():
    arr = _make_hole_array()
    payload = {
        "data": _array_to_payload(arr),
        "method": "nearest_neighbor",
        "raster_type": "continuous",
        "validation_strategy": "random_holdout",
        "sample_fraction": 0.1,
        "seed": 42,
    }
    resp = client.post("/api/v1/rasters/interpolate/validate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "nearest_neighbor"
    assert data["validation_strategy"] == "random_holdout"
    assert data["validation_sample_count"] > 0
    if data["mae"] is not None:
        assert data["mae"] >= 0.0


# ---------------------------------------------------------------------------
# Temporal interpolation endpoint
# ---------------------------------------------------------------------------

def test_interpolate_endpoint_temporal():
    cube = np.full((5, 10, 10), 50.0, dtype=float)
    cube[2, 3:7, 3:7] = np.nan
    payload = {
        "data": [[[None if np.isnan(v) else float(v) for v in row] for row in slice_2d] for slice_2d in cube],
        "method": "temporal_linear",
        "raster_type": "temporal",
        "band_index": 1,
        "times": [0.0, 1.0, 2.0, 3.0, 4.0],
    }
    resp = client.post("/api/v1/rasters/interpolate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["interpolated_pixel_count"] == 16
    assert data["remaining_nodata_count"] == 0


# ---------------------------------------------------------------------------
# Export endpoint
# ---------------------------------------------------------------------------

def test_export_endpoint():
    arr = _make_hole_array()
    # First interpolate
    interp_payload = {
        "data": _array_to_payload(arr),
        "method": "nearest_neighbor",
        "raster_type": "continuous",
        "band_index": 1,
    }
    interp_resp = client.post("/api/v1/rasters/interpolate", json=interp_payload)
    assert interp_resp.status_code == 200
    interp_data = interp_resp.json()

    # Then export
    export_payload = {
        "output_filename": "test_export.tif",
        "nodata": -9999.0,
        "include_provenance_band": True,
        "include_quality_band": True,
        "reconstructed_array": interp_data["reconstructed"],
        "provenance_mask": interp_data["provenance_mask"],
        "quality_mask": interp_data["quality_mask"],
        "crs": "EPSG:32610",
        "transform": {"a": 10, "b": 0, "c": 500000, "d": 0, "e": -10, "f": 5000000},
        "method": "nearest_neighbor",
    }
    resp = client.post("/api/v1/rasters/interpolate/export", json=export_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "test_export.tif"
    assert data["bands"] == 3
    assert data["crs"] == "EPSG:32610"


def test_export_endpoint_invalid_transform():
    export_payload = {
        "output_filename": "test.tif",
        "reconstructed_array": [[1.0, 2.0], [3.0, 4.0]],
        "provenance_mask": [["observed", "observed"], ["observed", "observed"]],
        "transform": {"a": 1},  # incomplete transform
        "method": "nearest_neighbor",
    }
    resp = client.post("/api/v1/rasters/interpolate/export", json=export_payload)
    assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
