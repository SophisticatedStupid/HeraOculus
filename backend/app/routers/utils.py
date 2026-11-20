"""Shared router helpers for memory-safe file uploads."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import HTTPException, UploadFile
from ..config import settings

# Read uploads in 64 KiB chunks so memory usage stays bounded by chunk size,
# not by the total file size.
CHUNK_SIZE = 64 * 1024

# Allowed file suffixes for uploads.  Client-supplied MIME types are never
# trusted; extension + magic-byte validation happens downstream.
_ALLOWED_SUFFIXES = {".tif", ".tiff", ".nc", ".hdf", ".h5", ".hdf5", ""}

# Magic-byte signatures for upload validation.
_GEOTIFF_MAGIC = {b"II*\x00", b"MM\x00*"}  # little-endian / big-endian TIFF
_NETCDF_MAGIC = b"CDF\x01"  # NetCDF classic / 64-bit
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"  # HDF5


def _safe_upload_name(filename: Optional[str]) -> str:
    """Return a safe basename for an uploaded file, rejecting path traversal."""
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")
    name = Path(filename).name
    if name != filename or name in (".", "..") or "/" in name or "\\" in name:
        raise HTTPException(
            status_code=400, detail=f"Unsafe filename: {filename!r}"
        )
    return name


def _safe_suffix(filename: str) -> str:
    """Return a lowercase suffix, restricted to known-safe raster/array formats."""
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        return ".bin"
    return suffix


def _read_magic(path: str, n: int = 8) -> bytes:
    """Read the first ``n`` bytes of a file for signature validation."""
    with open(path, "rb") as fh:
        return fh.read(n)


def validate_upload_signature(path: str, kind: Literal["geotiff", "netcdf_hdf5"]) -> None:
    """Validate that the file on disk matches the expected format by magic bytes.

    Raises HTTPException(400) if the signature does not match.  Client-supplied
    MIME types and extensions are never trusted.
    """
    magic = _read_magic(path, 8)
    if kind == "geotiff":
        if magic[:4] not in _GEOTIFF_MAGIC:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid GeoTIFF (bad magic bytes).",
            )
    elif kind == "netcdf_hdf5":
        if magic[:4] != _NETCDF_MAGIC and magic[:8] != _HDF5_MAGIC:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid NetCDF/HDF5 (bad magic bytes).",
            )


async def read_checked_upload(upload: UploadFile) -> tuple[str, str]:
    """Stream an uploaded file to a temp file, enforcing the size limit during
    reading.  Memory usage is bounded by CHUNK_SIZE, not by file size.

    Returns (temp_file_path, safe_filename).  Callers are responsible for
    deleting the temp file after processing (use a try/finally).
    """
    safe_name = _safe_upload_name(upload.filename)
    suffix = _safe_suffix(safe_name)

    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="geo_iris_upload_")

    total = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    mb = settings.max_upload_bytes // (1024 * 1024)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds {mb} MB limit.",
                    )
                handle.write(chunk)

        if total == 0:
            raise HTTPException(
                status_code=400, detail=f"Uploaded file '{safe_name}' is empty."
            )

        return tmp_path, safe_name
    except HTTPException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def cleanup_temp_file(path: Optional[str]) -> None:
    """Best-effort removal of a temporary upload file."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass
