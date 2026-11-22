"""Shared router helpers for memory-safe file uploads."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import queue
import tempfile
import threading
from multiprocessing.queues import Queue as MPQueue
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple

from fastapi import HTTPException, UploadFile, status

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

    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="heraoculus_upload_")

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


# ---------------------------------------------------------------------------
# Heavy task execution (process-based, terminable on timeout)
# ---------------------------------------------------------------------------

_heavy_max_concurrent_raw = os.getenv("HERAOCULUS_MAX_CONCURRENT_HEAVY_JOBS", "4").strip()
_HEAVY_MAX_CONCURRENT = int(_heavy_max_concurrent_raw) if _heavy_max_concurrent_raw else 4
_heavy_semaphores: "dict[asyncio.AbstractEventLoop, asyncio.Semaphore]" = {}
_heavy_semaphores_lock = threading.Lock()
_active_heavy: "set[mp.Process]" = set()
_active_heavy_lock = threading.Lock()


def _heavy_semaphore() -> asyncio.Semaphore:
    """Return a concurrency semaphore bound to the running event loop.

    One loop (the app) shares one semaphore so concurrent heavy jobs are capped;
    each isolated test loop gets its own so tests don't collide.
    """
    loop = asyncio.get_running_loop()
    with _heavy_semaphores_lock:
        sem = _heavy_semaphores.get(loop)
        if sem is None:
            sem = asyncio.Semaphore(_HEAVY_MAX_CONCURRENT)
            _heavy_semaphores[loop] = sem
        return sem


def _worker_run(
    queue: MPQueue,
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: dict,
) -> None:
    """Entry point for the heavy worker process."""
    try:
        result = func(*args, **kwargs)
        queue.put(("ok", result))
    except BaseException as exc:  # capture worker errors to surface in parent
        queue.put(("err", exc))


def _register_process(process: mp.Process) -> None:
    with _active_heavy_lock:
        _active_heavy.add(process)


def _unregister_process(process: mp.Process) -> None:
    with _active_heavy_lock:
        _active_heavy.discard(process)


def _terminate_process(process: mp.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


async def run_heavy_task(
    func: Callable,
    args: Tuple[Any, ...] = (),
    *,
    kwargs: Optional[dict] = None,
    timeout: float = 120.0,
) -> Any:
    """Run ``func`` in a separate process with a hard timeout and concurrency cap.

    The worker process is terminated on timeout so it cannot keep running (and
    keep reading temp files) after the HTTP request has given up.  Returns the
    worker's result on success; raises :class:`fastapi.HTTPException` 504 on
    timeout and 503 when the concurrency queue is full.

    The caller owns any temp files passed to ``func`` and must delete them only
    after this coroutine returns, at which point the worker is guaranteed to be
    finished or killed.
    """
    # Fail fast with 503 when no concurrency slot is free (non-blocking check).
    _sem = _heavy_semaphore()
    if _sem._value <= 0:  # noqa: SLF001 - Semaphore has no public try_acquire
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server busy: too many concurrent heavy jobs. Try again later.",
        )
    await _sem.acquire()

    ctx = mp.get_context("spawn")
    kwargs = kwargs or {}
    result_queue: MPQueue = ctx.Queue()
    process = ctx.Process(target=_worker_run, args=(result_queue, func, args, kwargs))
    _register_process(process)
    process.start()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + float(timeout)
    got = False
    tag: Any = None
    payload: Any = None

    try:
        while not got:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                tag, payload = result_queue.get_nowait()
                got = True
            except queue.Empty:
                if not process.is_alive():
                    # Worker died without producing a result.
                    break
                await asyncio.sleep(min(remaining, 0.2))
        if not got:
            _terminate_process(process)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Heavy processing timed out.",
            )
    finally:
        _terminate_process(process)
        _unregister_process(process)
        _heavy_semaphore().release()
        try:
            result_queue.close()
        except Exception:
            pass

    if tag == "err":
        raise payload  # surface worker exception in the caller
    return payload
