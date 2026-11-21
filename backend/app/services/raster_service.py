"""SAR / DEM Landform Change Detection, LULC Transition Matrix & NASA ASF Proxy
(backward-compatible re-export shim).

This module has been split into smaller modules under
``backend/app/services/raster/``:

- ``base.py``   shared raster I/O, alignment, pixel-area and rendering helpers
- ``dem.py``    DEM of Difference (DoD) change detection
- ``sar.py``    SAR speckle filtering, log-ratio change detection & ASF proxy
- ``lulc.py``   LULC transition matrix & Cohen's Kappa

All names are re-exported here so existing imports such as
``from app.services import raster_service as rs`` (including private,
test-referenced helpers like ``rs._read_raster_bytes``) and
``from ..services.raster_service import ...`` keep working unchanged.
No behavior has changed.
"""

from .raster import *  # noqa: F401, F403

__all__ = [name for name in dir() if not name.startswith("__")]
