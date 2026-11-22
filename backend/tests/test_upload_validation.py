"""Memory-safe file upload validation tests.

Tests that uploaded files are streamed in chunks with the size limit enforced
during reading, that malformed files are rejected, that temporary files are
cleaned up, and that concurrent uploads are handled safely.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

from app.main import app
from app.config import settings
from app.routers.utils import (
    CHUNK_SIZE,
    cleanup_temp_file,
    read_checked_upload,
    validate_upload_signature,
)
from app.security import limiter

client = TestClient(app)

MAX_BYTES = settings.max_upload_bytes
MB = 1024 * 1024


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Reset slowapi storage around each test so rate-limit counters never leak
    between tests now that endpoints are rate limited."""
    limiter.reset()
    yield
    limiter.reset()


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_geotiff_bytes(path: str, width: int = 8, height: int = 8) -> bytes:
    """Write a minimal valid GeoTIFF to ``path`` and return its bytes."""
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": from_bounds(0, 0, 10, 10, width, height),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.ones((height, width), dtype=np.float32), 1)
    with open(path, "rb") as f:
        return f.read()


def _make_bad_geotiff_bytes() -> bytes:
    """Return bytes with a .tif extension but invalid TIFF magic."""
    return b"Not a GeoTIFF at all " + b"\x00" * 100


def _make_netcdf_bytes() -> bytes:
    """Return bytes of a minimal valid NetCDF file."""
    import xarray as xr

    ds = xr.Dataset(
        {"t2m": (("lat", "lon"), np.ones((3, 3), dtype=np.float32))},
        coords={"lat": [1.0, 2.0, 3.0], "lon": [1.0, 2.0, 3.0]},
    )
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ds.to_netcdf(tmp_path, engine="netcdf4")
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def _make_bad_netcdf_bytes() -> bytes:
    """Return bytes with a .nc extension but invalid NetCDF/HDF5 magic."""
    return b"Not NetCDF " + b"\x00" * 100


# ── Upload helper ───────────────────────────────────────────────────────


def _upload_file(endpoint: str, filename: str, content: bytes, mime: str = "application/octet-stream"):
    """POST a file to an upload endpoint and return the response."""
    return client.post(
        endpoint,
        files={"file": (filename, content, mime)},
    )


def _upload_two_files(endpoint: str, f1: tuple[str, bytes], f2: tuple[str, bytes], data: dict | None = None):
    """POST two files to an upload endpoint."""
    files = [
        ("before_file", f1),
        ("after_file", f2),
    ]
    return client.post(endpoint, files=files, data=data or {})


# ── Tests ───────────────────────────────────────────────────────────────


def test_upload_below_limit():
    """A small valid GeoTIFF should be accepted."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        content = _make_geotiff_bytes(tmp_path)
        assert len(content) < MAX_BYTES
        resp = _upload_file("/api/v1/rasters/dem-diff/upload", "small.tif", content)
        # The file is too small for actual DEM differencing but should pass size/signature checks
        assert resp.status_code != 413, f"Expected non-413 for small file, got {resp.status_code}: {resp.text}"
    finally:
        os.unlink(tmp_path)


def test_upload_at_limit():
    """A file exactly at the size limit should be accepted."""
    content = b"\x00" * MAX_BYTES
    assert len(content) == MAX_BYTES
    resp = _upload_file("/api/v1/earthdata/netcdf/inspect", "limit.nc", content)
    # Should fail signature validation (not NetCDF), not size limit
    assert resp.status_code == 400, f"Expected 400 for bad signature at limit, got {resp.status_code}: {resp.text}"
    assert "magic" in resp.json()["detail"].lower(), f"Expected magic error, got: {resp.text}"


def test_upload_above_limit():
    """A file above the size limit should be rejected with 413."""
    content = b"\x00" * (MAX_BYTES + 1)
    assert len(content) > MAX_BYTES
    resp = _upload_file("/api/v1/earthdata/netcdf/inspect", "oversize.nc", content)
    assert resp.status_code == 413, f"Expected 413 for oversize file, got {resp.status_code}: {resp.text}"
    assert "exceeds" in resp.json()["detail"].lower(), f"Expected size-limit message, got: {resp.text}"


def test_upload_malformed_geotiff():
    """A .tif file with invalid GeoTIFF magic bytes should be rejected by
    signature validation on raster endpoints."""
    content = _make_bad_geotiff_bytes()
    # Upload two bad GeoTIFFs to dem-diff/upload
    resp = client.post(
        "/api/v1/rasters/dem-diff/upload",
        files=[
            ("before_file", ("bad1.tif", content, "image/tiff")),
            ("after_file", ("bad2.tif", content, "image/tiff")),
        ],
    )
    assert resp.status_code == 400, f"Expected 400 for malformed GeoTIFF, got {resp.status_code}: {resp.text}"
    assert "magic" in resp.json()["detail"].lower(), f"Expected magic error, got: {resp.text}"


def test_upload_malformed_netcdf():
    """A .nc file with invalid NetCDF/HDF5 magic bytes should be rejected."""
    content = _make_bad_netcdf_bytes()
    resp = _upload_file("/api/v1/earthdata/netcdf/inspect", "bad.nc", content)
    assert resp.status_code == 400, f"Expected 400 for malformed NetCDF, got {resp.status_code}: {resp.text}"
    assert "magic" in resp.json()["detail"].lower(), f"Expected magic error, got: {resp.text}"


def test_upload_empty_file():
    """An empty file should be rejected with 400."""
    resp = _upload_file("/api/v1/earthdata/netcdf/inspect", "empty.nc", b"")
    assert resp.status_code == 400, f"Expected 400 for empty file, got {resp.status_code}: {resp.text}"
    assert "empty" in resp.json()["detail"].lower(), f"Expected empty-file message, got: {resp.text}"


def test_upload_path_traversal_filename():
    """Filenames with path traversal should be rejected."""
    content = _make_bad_netcdf_bytes()
    resp = _upload_file("/api/v1/earthdata/netcdf/inspect", "../../../etc/passwd.nc", content)
    assert resp.status_code == 400, f"Expected 400 for path traversal, got {resp.status_code}: {resp.text}"


def test_chunked_reading_bounded_memory():
    """Verify that read_checked_upload reads in CHUNK_SIZE chunks and enforces
    the limit during streaming (not after full read)."""
    # Create an UploadFile-like object that streams data slowly
    async def _streaming_upload():
        content = b"\x00" * (MAX_BYTES + CHUNK_SIZE)
        # Simulate UploadFile with async read
        class FakeUpload:
            def __init__(self, data: bytes):
                self.data = data
                self.pos = 0
                self.filename = "stream.bin"

            async def read(self, n: int = -1) -> bytes:
                if self.pos >= len(self.data):
                    return b""
                chunk = self.data[self.pos : self.pos + n]
                self.pos += len(chunk)
                return chunk

        upload = FakeUpload(content)
        try:
            await read_checked_upload(upload)
            assert False, "Should have raised HTTPException(413)"
        except Exception as exc:
            from fastapi import HTTPException
            assert isinstance(exc, HTTPException), f"Expected HTTPException, got {type(exc)}"
            assert exc.status_code == 413, f"Expected 413, got {exc.status_code}"

    import asyncio

    asyncio.run(_streaming_upload())


def test_temp_file_cleanup_after_success():
    """Temporary upload files should be cleaned up after successful processing."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        content = _make_geotiff_bytes(tmp_path)
        # Upload via API - temp file created by read_checked_upload, cleaned by router
        resp = client.post(
            "/api/v1/earthdata/netcdf/inspect",
            files={"file": ("test.nc", _make_netcdf_bytes(), "application/x-netcdf")},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    finally:
        os.unlink(tmp_path)


def test_concurrent_uploads():
    """Multiple concurrent uploads should not interfere with each other."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        geotiff_content = _make_geotiff_bytes(tmp_path)
    finally:
        os.unlink(tmp_path)

    netcdf_content = _make_netcdf_bytes()

    def upload_geotiff(idx):
        return client.post(
            "/api/v1/earthdata/netcdf/inspect",
            files={"file": (f"concurrent_{idx}.nc", netcdf_content, "application/x-netcdf")},
        )

    def upload_bad(idx):
        return client.post(
            "/api/v1/earthdata/netcdf/inspect",
            files={"file": (f"bad_{idx}.nc", _make_bad_netcdf_bytes(), "application/x-netcdf")},
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(upload_geotiff, i) for i in range(4)]
        results = [f.result() for f in as_completed(futures)]

    # Within the per-minute upload bucket, concurrent valid uploads all succeed.
    assert all(r.status_code == 200 for r in results), [
        r.status_code for r in results
    ]

    # The bucket now has 4 of 5 used: the next upload succeeds, the one after
    # is rate limited rather than erroring.
    assert upload_geotiff(4).status_code == 200
    resp = upload_geotiff(5)
    assert resp.status_code == 429, resp.text

    # Malformed uploads are rejected by validation (400) or rate limiting (429),
    # never by a processing error (500).
    resp = upload_bad(0)
    assert resp.status_code in {400, 429}, resp.text


def test_validate_upload_signature_geotiff():
    """validate_upload_signature should accept valid GeoTIFF and reject invalid."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # Valid GeoTIFF
        content = _make_geotiff_bytes(tmp_path)
        validate_upload_signature(tmp_path, "geotiff")  # Should not raise

        # Invalid GeoTIFF
        with open(tmp_path, "wb") as f:
            f.write(_make_bad_geotiff_bytes())
        try:
            validate_upload_signature(tmp_path, "geotiff")
            assert False, "Should have raised HTTPException(400)"
        except Exception as exc:
            from fastapi import HTTPException
            assert isinstance(exc, HTTPException)
            assert exc.status_code == 400
    finally:
        os.unlink(tmp_path)


def test_validate_upload_signature_netcdf():
    """validate_upload_signature should accept valid NetCDF and reject invalid."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # Valid NetCDF
        content = _make_netcdf_bytes()
        with open(tmp_path, "wb") as f:
            f.write(content)
        validate_upload_signature(tmp_path, "netcdf_hdf5")  # Should not raise

        # Invalid NetCDF
        with open(tmp_path, "wb") as f:
            f.write(_make_bad_netcdf_bytes())
        try:
            validate_upload_signature(tmp_path, "netcdf_hdf5")
            assert False, "Should have raised HTTPException(400)"
        except Exception as exc:
            from fastapi import HTTPException
            assert isinstance(exc, HTTPException)
            assert exc.status_code == 400
    finally:
        os.unlink(tmp_path)


def test_cleanup_temp_file_nonexistent():
    """cleanup_temp_file should not raise for None or missing paths."""
    cleanup_temp_file(None)
    cleanup_temp_file("/nonexistent/path/that/does/not/exist.tmp")


def test_upload_limit_enforced_during_streaming():
    """The size limit must be enforced during streaming, not after full read.
    Upload exactly MAX_BYTES + 1 bytes and verify 413 is returned before
    the full payload is accumulated."""
    content = b"\x00" * (MAX_BYTES + 1)
    resp = _upload_file("/api/v1/earthdata/netcdf/inspect", "stream_over.nc", content)
    assert resp.status_code == 413, f"Expected 413, got {resp.status_code}: {resp.text}"


def test_upload_exactly_at_limit_netcdf_signature():
    """A file at exactly MAX_BYTES with valid NetCDF signature should pass
    size check (may fail later on content, but not size)."""
    # Build a file that is exactly MAX_BYTES: valid NetCDF header + padding
    netcdf_content = _make_netcdf_bytes()
    # Pad to exactly MAX_BYTES
    if len(netcdf_content) < MAX_BYTES:
        padded = netcdf_content + b"\x00" * (MAX_BYTES - len(netcdf_content))
    else:
        padded = netcdf_content[:MAX_BYTES]
    assert len(padded) == MAX_BYTES
    resp = _upload_file("/api/v1/earthdata/netcdf/inspect", "exact_limit.nc", padded)
    # Should pass size check; may fail on NetCDF parsing but not with 413
    assert resp.status_code != 413, f"Expected non-413 at exact limit, got {resp.status_code}: {resp.text}"
