# Geo-Iris

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
- USGS NWIS streamflow gauges (United States only)
- NOAA CO-OPS tide gauge sea level (United States coastal waters only)
- NASA GIBS WMTS map tiles (true-color, NDVI, snow, LST, SST, global)

### Frontend workflows
- Interactive Explorer with region selection and variable choice
- Detective learning mode (guided case studies)
- Regional contrast mode (paired region comparison)
- Variable encyclopedia with provenance and mission info
- Custom CSV and GeoTIFF/NetCDF upload
- Data export (CSV/GeoJSON)

## Data provenance

Every data source carries a `provenanceType` that distinguishes live measurements from catalog metadata, reanalysis, synthetic fallback, and user uploads:

| Provenance type | Sources | Live data? | Notes |
|---|---|---|---|
| `reanalysis_model` | NASA POWER (MERRA-2 / GLDAS / CERES) | Yes (reanalysis) | Global gridded reanalysis, not station measurements |
| `satellite_observation` | MODIS (NDVI/EVI), GIBS (map tiles), CMR (catalog) | Yes (satellite tiles/catalog) | GIBS SST layer is `GHRSST_L4_MUR` map tiles; CMR is metadata-only, no time series |
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

**File upload limit:** 35 MB per file, enforced during streaming (not after full read). Invalid magic bytes, empty files, and path-traversal filenames are rejected with 400; oversized files return 413.

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
docker build -t geo-iris-backend .
docker run -p 8000:8000 --env-file .env geo-iris-backend
```

## Testing & validation

### Backend (Python/pytest)

17 test files covering health/auth, SAR/FIRMS/Earthdata uploads, statistics, spatial FDR, raster alignment, DEM/SAR/LULC nodata handling, pixel-area audits, extreme-value temporal assumptions, Bayesian MCMC diagnostics, changepoint recovery, NASA POWER parameter-map correctness, and upload size-limit streaming.

```bash
cd backend
python -m pytest tests/ -v
```

### Frontend (TypeScript/Vitest)

6 test files covering statistical algorithms, worker communication and main-thread fallback, connector geographic selection and error handling, data provenance, CSV ingestion and empty-data handling, and Explorer workflow integration.

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
| `VITE_BACKEND_URL` | `http://localhost:8000` | Frontend backend URL (Vite env, prefix `VITE_`) |

Edit `.env` (backend) for server-side variables. `VITE_BACKEND_URL` is a Vite frontend variable and must be prefixed with `VITE_` in `.env`.

## Assumptions & limitations

- **NASA POWER** is global reanalysis (MERRA-2/GLDAS/CERES), not in-situ measurements. Polar-night and ocean cells return explicit fill values.
- **USGS streamflow** and **NOAA sea level** are United States only; queries outside their service area return a deterministic fallback (no connector selected), never a masquerading live source.
- **SST** is not available via NASA POWER time series; only as GIBS map tiles.
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
│   │   ├── connectors/          # 6 EarthData connectors + registry
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
    │   │   └── stats_advanced.py
    │   └── services/
    │       ├── earthdata_service.py
    │       ├── firms_service.py
    │       ├── raster_service.py
    │       ├── bayesian_service.py
    │       ├── extreme_service.py
    │       ├── spatial_fdr_service.py
    │       └── forecast_service.py
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
    │   └── test_changepoint_validation.py
    ├── requirements.txt
    ├── .env.example
    └── Dockerfile
```
