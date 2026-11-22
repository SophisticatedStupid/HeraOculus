# HeraOculus

[![CI](https://github.com/SophisticatedStupid/HeraOculus/actions/workflows/ci.yml/badge.svg)](https://github.com/SophisticatedStupid/HeraOculus/actions/workflows/ci.yml)

Earth observation analytics platform: trend detection, changepoint analysis, extreme-value statistics, raster change detection, and Earthdata/NASA data access.

## Contents

- [Features](#features)
- [Data provenance](#data-provenance)
- [Backend API](#backend-api)
- [Frontend](#frontend)
- [Installation](#installation)
- [Testing & validation](#testing--validation)
- [Configuration](#configuration)
- [Assumptions & limitations](#assumptions--limitations)
- [Directory structure](#directory-structure)

## Features

### Statistical analysis
- Mann-Kendall trend test, Theil-Sen slope, OLS regression
- Seasonal Mann-Kendall with Hamed-Rao and TFPW temporal autocorrelation correction
- One-way and Two-way ANOVA
- Nonlinear regression (linear, exponential, logistic, quadratic)
- GEV block maxima and GPD peak-over-threshold extreme value analysis
- Shapiro-Wilk normality test, bootstrap confidence intervals
- Durbin-Watson lag-1 autocorrelation diagnostics
- Bayesian changepoint MCMC and Bayesian robust AR(1) with convergence diagnostics (R-hat, ESS)

### Raster analysis (user uploads)
- DEM differencing with joint validity masking, LoD thresholding, volume in m³
- SAR backscatter change detection (fixed and Otsu threshold modes), area in km²
- LULC land-cover transition matrices and change maps
- Raster geospatial alignment: CRS validation, reprojection, resampling, nodata preservation

### Earthdata / NASA connectivity
- NASA FIRMS fire detections, EONET events
- NASA CMR granule search, AppEEARS point extraction, vertical profiles, trace gases
- NetCDF4/HDF5 inspect and extract
- NASA POWER monthly time series (reanalysis: temperature, precipitation, radiation, humidity, wind, etc.)
- ORNL DAAC MODIS time series (NDVI, EVI, global land)
- **CHIRPS v3** monthly rainfall (Climate Hazards Center, keyless, 0.05°, 60°S–60°N, 1981–present)
- **GPM IMERG Final V07B** monthly rainfall (NASA GES DISC, Earthdata login, 0.1°, global, 1998–present)
- **SMAP L4** monthly root-zone/surface soil moisture (NSIDC / Earthdata Cloud, Earthdata login, 9 km EASE-Grid 2.0, global land 85°S–85°N, 2015–present)
- **ERA5-Land** monthly total precipitation (Copernicus CDS / ARCO Zarr mirror, CDS key, 0.1°, global land, 1950–present)
- **MODIS LST Night (MOD11A1 V061)** monthly nighttime land surface temperature (LAADS, Earthdata login, 1 km, global, 2000–present)
- USGS NWIS streamflow gauges (United States only)
- NOAA CO-OPS tide gauge sea level (United States coastal waters only)
- NASA GIBS WMTS map tiles (true-color, NDVI, snow, LST, SST, global)
- **STAC** scene catalog search (Earth Search / Planetary Computer, Sentinel-2, Landsat, Cloud-Optimized GeoTIFF asset URLs)

### Frontend workflows
- Interactive Explorer with region selection and variable choice
- Detective learning mode (guided case studies)
- Regional contrast mode (paired region comparison)
- Variable encyclopedia with provenance and mission info
- Custom CSV and GeoTIFF/NetCDF upload
- Data export (CSV/GeoJSON)
- **Generate GEE script** — ready-to-run Google Earth Engine Code Editor (JavaScript) and earthengine-api (Python) scripts from the current Explorer selection (region/point, date range, variable), with JS/Python tabs, copy-to-clipboard, and .js/.py download. Pure client-side code generation; no GEE credentials, backend, or network calls.

## Data provenance

Every data source carries a `provenanceType` that distinguishes live measurements from catalog metadata, reanalysis, synthetic fallback, and user uploads:

| Provenance type | Sources | Live data? | Notes |
|---|---|---|---|
| `reanalysis_model` | NASA POWER (MERRA-2 / GLDAS / CERES), ERA5-Land | Yes (reanalysis) | Global gridded reanalysis, not station measurements. ERA5-Land requires a CDS API key / DestinE PAT; CDS access is queue-throttled. |
| `satellite_observation` | MODIS (NDVI/EVI), GIBS (map tiles), CMR (catalog), STAC (scene catalog + COG asset URLs), CHIRPS v3 (rainfall), GPM IMERG Final (rainfall), SMAP L4 (soil moisture), MODIS LST Night (temperature) | Yes (catalog/tiles; STAC assets are live COG URLs) | GIBS SST layer is `GHRSST_L4_MUR` map tiles; CMR and STAC are metadata/catalog — STAC exposes Cloud-Optimized GeoTIFF asset URLs for direct access. CHIRPS blends satellite estimates with in-situ gauge stations (not pure satellite); IMERG is satellite–gauge merged. SMAP root-zone soil moisture is a GEOS land-model estimate assimilating SMAP brightness temperatures (~7-day L4 latency), not a direct observation. MODIS LST Night is clear-sky only (~5-day latency). |
| `in_situ_gauge` | USGS NWIS streamflow, NOAA CO-OPS tide gauges | Yes (gauge/station) | United States only |
| `user_upload` | DEM, SAR, LULC GeoTIFF/NetCDF uploads | N/A (user-provided) | 35 MB limit per file |
| `synthetic_demo` | Fallback when no live source covers a location | No | Calibrated demonstration data; never masquerades as live |

**Important:** NASA Sea Surface Temperature (SST) is **not** available as a NASA POWER time series. The POWER connector correctly rejects SST (`POWER_PARAM_MAP` excludes it). SST is available only as NASA GIBS map tiles (`GHRSST_L4_MUR_Sea_Surface_Temperature`). Regional providers (USGS streamflow, NOAA sea level) are limited to the United States and are never selected outside their service area.

## Backend API

Base path: `/api/v1`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check |
| `/auth/status` | GET | Authentication status |
| `/stats/sarima` | POST | ARIMA/SARIMA time-series forecasting |
| `/stats/bayesian-mcmc` | POST | Bayesian changepoint / robust AR(1) MCMC |
| `/stats/extreme-value` | POST | GEV block maxima / GPD POT extreme value |
| `/stats/spatial-fdr` | POST | Spatial Mann-Kendall with FDR correction |
| `/firms/fires` | POST | NASA FIRMS fire detections |
| `/earthdata/granules` | POST | NASA CMR granule search |
| `/earthdata/appeears-point` | POST | AppEEARS point extraction |
| `/earthdata/vertical-profile` | POST | Vertical profile extraction |
| `/earthdata/trace-gases` | POST | Trace gas column data |
| `/earthdata/netcdf/inspect` | POST | NetCDF4/HDF5 metadata inspect |
| `/earthdata/netcdf/extract` | POST | NetCDF4/HDF5 variable extract |
| `/rasters/dem-diff` | POST | DEM differencing (array input) |
| `/rasters/dem-diff/upload` | POST | DEM differencing (file upload) |
| `/rasters/sar-change` | POST | SAR change detection (array input) |
| `/rasters/sar-change/upload` | POST | SAR change detection (file upload) |
| `/rasters/asf-search` | POST | ASF SAR scene search |
| `/rasters/lulc-transitions` | POST | LULC transition analysis (array input) |
| `/rasters/lulc-transitions/upload` | POST | LULC transition analysis (file upload) |
| `/stac/search` | POST | Search public STAC catalogs (Sentinel-2, Landsat) via Earth Search / Planetary Computer proxy |
| `/gridded/{dataset}` | POST | Normalized monthly precipitation series from gridded products (CHIRPS, IMERG, ERA5-Land, SMAP soil moisture, MODIS LST Night) via backend proxy; explicit `unavailable` status when credentials/coverage missing |

**File upload limit:** 35 MB per file, enforced during streaming (not after full read). Invalid magic bytes, empty files, and path-traversal filenames are rejected with 400; oversized files return 413.

### Example: STAC search (Earth Search, keyless)

Request:

```http
POST /api/v1/stac/search
Content-Type: application/json

{
  "collections": ["sentinel-2-l2a"],
  "bbox": [-122.5, 37.0, -122.0, 37.5],
  "start_date": "2024-06-01T00:00:00Z",
  "end_date": "2024-06-10T23:59:59Z",
  "limit": 1
}
```

Normalized response:

```json
{
  "provider": "earth-search",
  "items": [
    {
      "id": "S2B_10SEF_20240607_0_L2A",
      "collection": "sentinel-2-l2a",
      "provider": "earth-search",
      "datetime": "2024-06-07T19:04:26.934000Z",
      "cloud_cover": 55.44,
      "bbox": [-122.5, 37.0, -122.0, 37.5],
      "platform": "sentinel-2b",
      "assets": {
        "red": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/10/S/EF/2024/6/S2B_10SEF_20240607_0_L2A/B04.tif",
        "nir": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../B08.tif",
        "swir": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../B11.tif",
        "scl": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../SCL.tif",
        "blue": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../B02.tif",
        "green": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../B03.tif",
        "thumbnail": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../thumbnail.jpg"
      },
      "thumbnail": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../thumbnail.jpg",
      "geometry": { "type": "Point", "coordinates": [-122.25, 37.25] }
    }
  ],
  "context": { "matched": 4, "returned": 1, "limit": 1 },
  "next_token": null
}
```

Set `STAC_PROVIDER=planetary-computer` in the backend environment to use Microsoft
Planetary Computer instead; its Azure Blob asset URLs are signed with a SAS token
via `https://planetarycomputer.microsoft.com/api/sas/v1/sign?href=...` before
being returned. No tokens or credentials are stored or logged.

## Frontend

- Framework: React + TypeScript + Vite
- Dev server: `http://localhost:3000`
- Build output: `dist/`
- Test runner: Vitest (Node environment, no browser required)
- Connector registry: `src/services/connectors/registry.ts` — geographically aware selection with deterministic fallback

## Installation

### Prerequisites

- Node.js 18+ and npm
- Python 3.10+ and pip (backend)
- `ffmpeg`, `gdal`, `rasterio`, `netCDF4`, `xarray`, `scipy`, `numpy`, `pandas`, `pyproj` (backend)

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # edit NASA_FIRMS_MAP_KEY, EARTHDATA_TOKEN, etc. as needed
python -m app.main     # or: uvicorn app.main:app --reload
```

Backend tests:

```bash
cd backend
python -m pytest tests/ -v
```

### Frontend

```bash
cd <project-root>
npm install
```

Frontend tests:

```bash
npm run test
```

### Docker (backend)

```bash
cd backend
docker build -t heraoculus-backend .
docker run -p 8000:8000 --env-file .env heraoculus-backend
```

## Testing & validation

### Backend (Python/pytest)

20 test files covering health/auth, SAR/FIRMS/Earthdata uploads, statistics, spatial FDR, raster alignment, DEM/SAR/LULC nodata handling, pixel-area audits, extreme-value temporal assumptions, Bayesian MCMC diagnostics, changepoint recovery, NASA POWER parameter-map correctness, upload size-limit streaming, STAC catalog search, and gridded climate search (CHIRPS, IMERG, ERA5-Land, SMAP, MODIS LST Night).

```bash
cd backend
python -m pytest tests/ -v
```

### Frontend (TypeScript/Vitest)

8 test files covering statistical algorithms, worker communication and main-thread fallback, connector geographic selection and error handling, data provenance, CSV ingestion and empty-data handling, Explorer workflow integration, and GEE script generation snapshots/validation.

```bash
npm run test
```

### Type check / lint

```bash
npm run typecheck     # strict tsc --noEmit (serves as lint)
npm run build         # tsc && vite build
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `BACKEND_HOST` | `0.0.0.0` | Backend bind host |
| `BACKEND_PORT` | `8000` | Backend bind port |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins |
| `NASA_FIRMS_MAP_KEY` | *(unset)* | NASA FIRMS map tile API key (optional) |
| `EARTHDATA_TOKEN` | *(unset)* | NASA Earthdata auth token (optional) |
| `EARTHDATA_USERNAME` | *(unset)* | NASA Earthdata username (optional) |
| `EARTHDATA_PASSWORD` | *(unset)* | NASA Earthdata password (optional) |
| `CDS_API_KEY` | *(unset)* | Copernicus CDS API key for ERA5-Land (optional) |
| `CDSAPI_KEY` | *(unset)* | Alternate Copernicus CDS API key name (optional) |
| `DESTINE_PAT` | *(unset)* | DestinE Earth Data Hub PAT for ERA5-Land ARCO Zarr (optional) |
| `VITE_BACKEND_URL` | `http://localhost:8000` | Frontend backend URL (Vite env, prefix `VITE_`) |
| `HERAOCULUS_API_KEY` | *(unset)* | Server-only API key. When set, every /api/v1/* route except /health requires X-API-Key. Injected by the Vercel proxy; never embedded in the frontend bundle. |
| `TRUST_PROXY_HEADERS` | `false` | Trust X-Forwarded-For for rate-limit keying when behind a trusted reverse proxy (e.g. Vercel). Set `true` only in production behind a proxy that overwrites X-Forwarded-For from the TCP connection. |
| `BACKEND_URL` | *(unset)* | Server-only backend URL used by the Vercel proxy function to forward /api/v1/* requests. Not prefixed with `VITE_`, so it is never exposed to the browser. |

Edit `.env` (backend) for server-side variables. `VITE_BACKEND_URL` is a Vite frontend variable and must be prefixed with `VITE_` in `.env`. `HERAOCULUS_API_KEY`, `TRUST_PROXY_HEADERS`, and `BACKEND_URL` are backend/server-only variables.

## API key & proxy architecture

The backend enforces an `X-API-Key` header on all `/api/v1/*` routes (except `/health`) when `HERAOCULUS_API_KEY` is set. Because the key must never be embedded in the public frontend bundle, the frontend calls same-origin `/api/v1/*` in production and a Vercel serverless function (`api/index.ts`) proxies those requests to the backend, injecting `X-API-Key` server-side:

```
Browser → https://app.vercel.app/api/v1/* → api/index.ts (adds X-API-Key, forwards X-Forwarded-For) → https://backend:8000/api/v1/*
```

This design:
- Keeps the API key out of client-side code (no `VITE_*` secret).
- Forwards the real client IP via `X-Forwarded-For` (Vercel populates it from the TCP connection), so per-client rate limiting works behind the proxy.
- Allowlists only `/api/v1/*` paths and `GET`/`POST` methods in the proxy.
- Streams request/response bodies, so multipart file uploads up to 35 MB work.

**This is not user authentication.** The API key protects the backend from direct, unauthenticated access, but every browser user shares the same key. It does not identify individual users. Set `TRUST_PROXY_HEADERS=true` on the backend in production so the rate limiter keys on the real client IP rather than the proxy's IP.

In local development the Vite dev proxy (`/api` → `http://127.0.0.1:8000`) is used and the backend typically runs without `HERAOCULUS_API_KEY` (dev mode). The runtime `localStorage` backend-URL override is disabled in production builds (`import.meta.env.DEV` only).

## Assumptions & limitations

- **NASA POWER** is global reanalysis (MERRA-2/GLDAS/CERES), not in-situ measurements. Polar-night and ocean cells return explicit fill values.
- **USGS streamflow** and **NOAA sea level** are United States only; queries outside their service area return a deterministic fallback (no connector selected), never a masquerading live source.
- **SST** is not available via NASA POWER time series; only as GIBS map tiles.
- **CHIRPS v3** covers 60°S–60°N (land); queries outside that band return an explicit unavailable result, never synthetic data.
- **GPM IMERG Final** requires NASA Earthdata Login credentials on the backend; without them it returns an explicit unavailable result. V07 Final ran through 2025-09 (~3.5–4 month latency).
- **ERA5-Land** requires a CDS API key or DestinE PAT on the backend; without them it returns an explicit unavailable result. CDS access is queue-throttled; prefer the ARCO Zarr mirror.
- **SMAP L4 (SPL4SMGP V008)** root-zone/surface soil moisture requires NASA Earthdata Login credentials on the backend; without them it returns an explicit unavailable result. Root-zone SM is a GEOS land-model estimate assimilating SMAP brightness temperatures (~7-day L4 latency), not a direct observation. Covers global land 85°S–85°N.
- **MODIS LST Night (MOD11A1 V061)** requires NASA Earthdata Login credentials on the backend; without them it returns an explicit unavailable result. Stored as scaled integers (K = raw × 0.02, fill 0, valid range 7500–65535); clear-sky only, ~5-day latency. Covers global 90°S–90°N.
- **CMR** is metadata/catalog search only; it produces no time series.
- **GIBS** is map-tile visualization only; it produces no time series.
- **Extreme value analysis** requires explicit dates and observation frequency; without dates, frequency is flagged as unknown and return-period assumptions are not silently inferred.
- **Bayesian MCMC** convergence diagnostics (R-hat, ESS) are reported per parameter; unconverged chains are flagged, not hidden.
- **Raster analysis** uses joint validity masking; nodata/NaN/Inf pixels are never zero-filled or included in volumes/areas.
- **Synthetic demo data** is used as fallback when no live source covers a query location; it is clearly labeled `isLiveNasa: false` / `provenanceType: synthetic_demo` and never presented as a successful live measurement.
- No scientific claims of validation against external reference datasets are made beyond the unit/regression tests in `tests/` and `backend/tests/`.

## Directory structure

```
Geo-Iris-main/
├── README.md
├── GEO-IRIS_COMPLETE_FEATURE_CATALOG.txt
├── package.json
├── tsconfig.json
├── vite.config.ts
├── vitest.config.ts
├── src/
│   ├── App.tsx
│   ├── components/
│   │   ├── DetectiveMode.tsx
│   │   ├── RegionalContrastView.tsx
│   │   ├── StatisticalLab.tsx
│   │   ├── VariableEncyclopedia.tsx
│   │   ├── DataExportModal.tsx
│   │   ├── CustomDataUploader.tsx
│   │   └── ...
│   ├── services/
│   │   ├── statistics/          # trend, ANOVA, nonlinear, GEV, bootstrap, PCA
│   │   ├── connectors/          # 12 EarthData connectors + registry
│   │   ├── nasaData.ts          # CSV ingestion, variable definitions
│   │   └── ...
│   ├── workers/
│   │   ├── statsWorker.ts
│   │   └── statsWorkerClient.ts # main-thread fallback
│   └── types/
│       └── earthScience.ts
├── tests/
│   ├── statistics.test.ts
│   ├── connector.test.ts
│   ├── worker.test.ts
│   ├── provenance.test.ts
│   ├── csv-ingestion.test.ts
│   └── explorer.integration.test.ts
└── backend/
    ├── app/
    │   ├── main.py
    │   ├── config.py
    │   ├── schemas.py
    │   ├── routers/
    │   │   ├── health.py
    │   │   ├── earthdata.py
    │   │   ├── firms.py
    │   │   ├── rasters.py
    │   │   ├── stac.py
    │   │   ├── gridded.py
    │   │   └── stats_advanced.py
    │   └── services/
    │       ├── earthdata_service.py
    │       ├── firms_service.py
    │       ├── raster_service.py
    │       ├── bayesian_service.py
    │       ├── extreme_service.py
    │       ├── spatial_fdr_service.py
    │       ├── forecast_service.py
    │       └── gridded_climate_service.py
    ├── tests/
    │   ├── test_health_and_auth.py
    │   ├── test_earthdata_firms.py
    │   ├── test_stats_advanced.py
    │   ├── test_rasters.py
    │   ├── test_lulc_nodata.py
    │   ├── test_auth_security.py
    │   ├── test_spatial_fdr_validation.py
    │   ├── test_pixel_area_audit.py
    │   ├── test_nasa_power_sst.py
    │   ├── test_upload_validation.py
    │   ├── test_dem_nodata.py
    │   ├── test_sar_nodata.py
    │   ├── test_sar_threshold_modes.py
    │   ├── test_raster_alignment.py
    │   ├── test_extreme_temporal.py
    │   ├── test_bayesian_diagnostics.py
    │   ├── test_changepoint_validation.py
    │   ├── test_stac_service.py
    │   ├── test_gridded_climate.py
    │   └── validation/
    ├── requirements.txt
    ├── .env.example
    └── Dockerfile
```
