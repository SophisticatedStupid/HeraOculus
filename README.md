# HERAOCULUS

HERAOCULUS (also called Geo-Iris in the source code) is an interactive Earth
science change-detection and significance platform. It helps students,
researchers, and citizen scientists explore environmental time series,
understand their provenance, and distinguish persistent trends from seasonal
or short-lived variability.

The application has two complementary compute paths:

- **Browser compute** for exploration, charts, statistical analysis, custom
  CSV/JSON data, and the detective learning mode.
- **FastAPI compute** for larger or specialized workflows such as forecasting,
  Bayesian inference, extreme-value analysis, gridded spatial FDR, raster
  change detection, and NetCDF/HDF5 inspection.

> **Data and scientific-use note:** Results are exploratory and depend on the
> selected dataset, preprocessing, missing values, spatial/temporal resolution,
> and statistical assumptions. Treat demonstration data and fallback results as
> labeled demonstrations, not as observed measurements.

## What it does

### Explorer

- Select Earth variables including temperature, precipitation, NDVI, soil
  moisture, evapotranspiration, aerosol optical depth, land-surface
  temperature, ozone, and additional atmosphere, land, ocean, and hydrology
  variables.
- Explore predefined global regions or click any map coordinate.
- Retrieve supported live feeds through the connector registry and NASA POWER,
  with an explicitly labeled calibrated demonstration fallback when a live
  endpoint is unavailable.
- View raw values, deseasonalized anomalies, rolling means, trend estimates,
  confidence bands, provenance, and missing observations.
- Export the active analysis as CSV/JSON or a printable dossier.

### Statistical Lab

The browser-side lab includes descriptive and inferential statistics, OLS and
robust/nonlinear regression, ANOVA, PCA, residual diagnostics, change-point
analysis, extreme-value fitting, and autocorrelation-aware Mann-Kendall
methods. Heavy operations can run in a Web Worker with a main-thread fallback.

### Data Detective and regional contrasts

Detective Mode provides four guided investigations covering Arctic warming,
Amazon canopy loss, Aral Sea decline, and California soil-moisture stress.
Regional Contrast compares paired locations whose trends move in opposite
directions and explains the physical context.

### Earth Compute Lab

The optional backend exposes:

- SARIMA/ARIMA forecasting with diagnostics, decomposition, prediction
  intervals, and holdout metrics.
- Multi-chain Bayesian MCMC for robust AR(1) trends and change points.
- GEV block-maxima and GPD peaks-over-threshold extreme-value analysis.
- Vectorized gridded Mann-Kendall and BH, BY, Holm, or Bonferroni spatial FDR.
- DEM differencing, SAR log-ratio change detection, LULC transitions, and ASF
  catalog search.
- NASA FIRMS/EONET fire queries and Earthdata CMR, AppEEARS, atmospheric
  profile, trace-gas, and NetCDF4/HDF5 workflows.

## Architecture

```text
Browser
  React 19 + TypeScript + Vite + Tailwind CSS v4
      ├── NASA POWER and connector integrations
      ├── Client-side statistics and Web Worker
      └── /api/v1/* ── Vite proxy (development) ── FastAPI backend

Backend
  FastAPI + Pydantic
      ├── NumPy / pandas / SciPy / statsmodels / scikit-learn
      ├── xarray / netCDF4 / rasterio / shapely
      └── NASA proxy and uploaded-file workflows
```

The frontend development server runs on port `3000`; the backend defaults to
`8000`. The Vite proxy sends `/api` requests to
`http://127.0.0.1:8000`.

## Requirements

- Node.js with npm
- Python 3.11 or newer (the backend Docker images use Python 3.12)
- Optional: Docker for the backend
- Optional NASA credentials for authenticated FIRMS and Earthdata workflows

## Quick start

### 1. Install frontend dependencies

```bash
npm install
```

### 2. Start the frontend

```bash
npm run dev
```

Open <http://localhost:3000>. The browser-only explorer and statistical lab can
be used without the backend.

### 3. Start the optional backend

In a second terminal:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend documentation is available at <http://127.0.0.1:8000/docs>, and the
health endpoint is <http://127.0.0.1:8000/api/v1/health>.

### Backend with Docker

```bash
cd backend
docker build -t heraoculus-backend .
docker run --rm -p 8000:8000 --env-file .env heraoculus-backend
```

If no `.env` exists, omit `--env-file .env` or create one from
[`backend/.env.example`](backend/.env.example).

## Configuration and credentials

Copy the example file before adding optional credentials:

```bash
cp backend/.env.example backend/.env
```

Supported settings include:

| Variable | Purpose | Default |
| --- | --- | --- |
| `BACKEND_HOST` | Backend bind host | `127.0.0.1` |
| `BACKEND_PORT` | Backend port | `8000` |
| `CORS_ORIGINS` | Comma-separated browser origins | Local ports `3000` and `5173` |
| `NASA_FIRMS_MAP_KEY` | NASA FIRMS active-fire API key | Empty |
| `EARTHDATA_TOKEN` | NASA Earthdata bearer token | Empty |
| `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` | Earthdata login alternative | Empty |
| `VITE_BACKEND_URL` | Frontend backend base URL for a separately deployed API | Vite `/api` proxy |

FIRMS and Earthdata credentials can also be configured for the current
backend session through the UI/API. Secrets are masked in status responses and
are not intended to be persisted by the application. Keyless feeds such as
NASA POWER, EONET, CMR, ASF search, local uploads, and demonstration fallbacks
remain available where supported.

## Backend API

All routes below are prefixed as shown. Request and response schemas are
defined in [`backend/app/schemas.py`](backend/app/schemas.py), and interactive
OpenAPI documentation is generated at `/docs`.

| Prefix | Operations |
| --- | --- |
| `/api/v1` | `GET /health`, `GET /auth/status`, `POST /auth/configure` |
| `/api/v1/stats` | `POST /sarima`, `/bayesian-mcmc`, `/extreme-value`, `/spatial-fdr` |
| `/api/v1/firms` | `POST /fires` |
| `/api/v1/earthdata` | `POST /granules`, `/appeears-point`, `/vertical-profile`, `/trace-gases`, `/netcdf/inspect`, `/netcdf/extract` |
| `/api/v1/rasters` | `POST /dem-diff`, `/dem-diff/upload`, `/sar-change`, `/sar-change/upload`, `/asf-search`, `/lulc-transitions`, `/lulc-transitions/upload` |

Uploaded GeoTIFF and NetCDF/HDF5 requests are bounded by a 35 MB backend
safety limit. Use the generated API docs for exact multipart fields and JSON
payloads.

## Data provenance

Every active series is assigned one of these provenance tiers:

- **Satellite observation**: direct satellite or in-situ connector data.
- **NASA model / reanalysis**: gridded POWER, MERRA-2, GLDAS, or related model
  data.
- **User upload**: CSV or JSON supplied through the Data Studio.
- **Calibrated demonstration**: offline fallback data labeled in the UI.

Missing values are preserved where possible, including NASA fill values such
as `-999`. Statistical views report the usable sample and missingness rather
than silently presenting a complete series.

## Development commands

```bash
# Type-check and create the production frontend bundle
npm run build

# Serve the production bundle locally
npm run preview

# Run backend tests
cd backend
python -m pytest
```

The repository also contains focused TypeScript verification scripts in
[`tests/`](tests/) and statistical unit tests beside their implementation in
[`src/services/statistics/`](src/services/statistics/). There is currently no
dedicated npm test script; use the project’s configured TypeScript/build
checks and the backend pytest suite when validating changes.

## Deployment

The root [`vercel.json`](vercel.json) builds the Vite frontend with
`npm run build`, serves `dist`, and rewrites browser routes to
`index.html`. The backend includes a standard
[`backend/Dockerfile`](backend/Dockerfile) and a Vercel-oriented
[`backend/Dockerfile.vercel`](backend/Dockerfile.vercel).

For a separately hosted backend, set `VITE_BACKEND_URL` at frontend build time
and configure the backend’s `CORS_ORIGINS` for the deployed frontend origin.
Do not commit `backend/.env` or any API credentials.

## Repository map

```text
src/
  App.tsx                         Main application state and mode routing
  components/                     Explorer, labs, maps, charts, and UI
  services/connectors/             NASA and public-data connector registry
  services/statistics/             Browser statistical implementations
  services/phase2Api.ts            Typed FastAPI client
  workers/                         Off-thread statistical computation
  types/earthScience.ts            Shared domain and provenance types
backend/app/
  main.py                         FastAPI entrypoint and CORS
  routers/                        Health, statistics, Earthdata, FIRMS, rasters
  services/                       Scientific models and external-data adapters
  schemas.py                      Pydantic API contracts
backend/tests/                    Backend API and scientific validation tests
public/                           Browser icons and static assets
```

## License and external data

No license file is currently included in this repository. Review the terms of
each upstream provider before redistributing data or derived products,
including NASA, NOAA, USGS, ASF, and other connector sources.
