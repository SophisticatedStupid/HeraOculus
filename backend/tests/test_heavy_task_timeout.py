"""Tests for the process-based heavy task helper (timeout termination + cleanup)."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.routers.utils import _active_heavy, run_heavy_task


def _slow_job(path: str, marker: str) -> dict:
    """A job that sleeps well past any sane timeout, then marks completion."""
    time.sleep(10)
    Path(marker).write_text("done")
    return {"ok": True, "path": path}


def _identity(value: int) -> int:
    return value


def test_slow_job_returns_504_worker_killed_and_file_cleaned():
    """A slow heavy job times out with 504, the worker is killed, and the temp
    file is only removed after the worker has finished or been killed."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    marker = tempfile.mktemp(suffix=".marker")
    assert os.path.exists(path)

    try:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                run_heavy_task(_slow_job, (path, marker), timeout=1.0)
            )
        assert exc_info.value.status_code == 504, exc_info.value.detail

        # The worker must have been killed before it could mark completion.
        assert not Path(marker).exists(), "slow worker should not have completed"

        # No worker process should still be running.
        for proc in list(_active_heavy):
            assert not proc.is_alive()
        assert len(_active_heavy) == 0
    finally:
        # The caller deletes the temp file only after run_heavy_task returns,
        # at which point the worker is guaranteed finished or killed.
        if os.path.exists(path):
            os.unlink(path)

    assert not os.path.exists(path), "temp file should be cleaned up afterwards"


def test_heavy_task_returns_result_on_success():
    """A fast job completes and returns its result."""
    result = asyncio.run(run_heavy_task(_identity, (42,), timeout=10.0))
    assert result == 42
    assert len(_active_heavy) == 0


def test_503_when_concurrency_queue_is_full():
    """When all heavy slots are occupied, new heavy jobs get 503 immediately."""

    async def main() -> None:
        sem = None
        from app.routers.utils import _heavy_semaphore

        # Fill every concurrency slot in this loop.
        for _ in range(4):
            await _heavy_semaphore().acquire()
        try:
            with pytest.raises(HTTPException) as exc_info:
                await run_heavy_task(_identity, (1,), timeout=10.0)
            assert exc_info.value.status_code == 503, exc_info.value.detail
        finally:
            for _ in range(4):
                _heavy_semaphore().release()

    asyncio.run(main())
    assert len(_active_heavy) == 0
