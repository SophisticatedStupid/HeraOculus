"""HeraOculus Phase 2 FastAPI application entrypoint."""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from .config import settings
from .routers import earthdata, firms, gridded, health, rasters, stats_advanced, stac
from .security import RateLimitMiddleware, limiter

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Phase 2 scientific backend providing SARIMA/ARIMA state-space forecasting, "
        "Bayesian MCMC trend & change-point inference, GEV/GPD extremes, vectorized "
        "Spatial FDR, NetCDF4/HDF5 inspection, GeoTIFF DEM/SAR/LULC change detection, "
        "and NASA FIRMS/EONET/CMR/ASF/AppEEARS proxy feeds."
    ),
)
app.state.limiter = limiter


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Consistent 429: JSON body plus a Retry-After header.

    Mirrors slowapi's header math using ``request.state.view_rate_limit``,
    which the limiter sets (limit item + [key, scope] identifiers) whenever a
    limit is evaluated, including on the failing request.
    """
    response = JSONResponse(
        {"error": f"Rate limit exceeded: {exc.detail}"}, status_code=429
    )
    app_limiter: Limiter = request.app.state.limiter
    view_rate_limit = getattr(request.state, "view_rate_limit", None)
    if view_rate_limit is not None:
        try:
            limit_item, args = view_rate_limit
            window_stats = app_limiter.limiter.get_window_stats(limit_item, *args)
            reset_in = 1 + window_stats[0]
            response.headers["Retry-After"] = str(int(reset_in - time.time()))
        except Exception:  # never let header math break the 429 response
            pass
    return response


app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)

app.include_router(health.router)
app.include_router(stats_advanced.router)
app.include_router(firms.router)
app.include_router(earthdata.router)
app.include_router(rasters.router)
app.include_router(stac.router)
app.include_router(gridded.router)


@app.get("/")
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.version,
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
    }
