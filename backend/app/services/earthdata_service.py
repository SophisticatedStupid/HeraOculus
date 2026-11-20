"""NASA Earthdata, AppEEARS, GES DISC Vertical Profiles, Trace Gases & NetCDF4/HDF5 Xarray Service.

Provides NASA Earthdata proxy services, vertical pressure profiles, trace gas analytics,
NetCDF/HDF5 file inspection/extraction, and AppEEARS task proxy. All scientific
measurement arrays are explicitly separated from CMR catalog metadata in provenance
metadata so that UI can clearly distinguish catalog discovery from actual data retrieval.

Key design principles:
- CMR granule catalog search results are metadata-only; they never imply actual
  satellite observations were downloaded or subset.
- Measurement provenance (tier: satellite_observation | reanalysis_model |
  user_upload | synthetic_demo) is tracked independently from CMR provenance.
- When authenticated EDL access is available, actual OPeNDAP/HDF5 granule subsetting
  is supported via the existing NetCDF/HDF5 extraction workflow.
- When EDL credentials are absent, synthetic demonstration values are generated
  with explicit provenance tier labeling so UI can always distinguish the data type.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal, Optional, TypedDict

import numpy as np
import requests
import xarray as xr
from scipy import stats

from .auth_store import auth_store


# ──────────────────────────────────────────────────────────────────
# Typed dicts for separated provenance
# ──────────────────────────────────────────────────────────────────


class CmrProvenance(TypedDict):
    """Metadata-only provenance from CMR granule catalog search."""

    tier: str
    source_name: str
    dataset_id: str
    is_live: bool
    status_message: str
    granule_count: int


class MeasurementProvenance(TypedDict):
    """Provenance of the actual measurement/sounding array returned."""

    tier: str
    source_name: str
    dataset_id: str
    is_live: bool
    status_message: str


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────


def _open_xarray_bytes(raw_bytes: bytes, filename: str) -> xr.Dataset:
    """Safely open NetCDF3/NetCDF4/HDF5 bytes using xarray via a temporary file."""
    suffix = Path(filename).suffix.lower() or ".nc"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw_bytes)
        errors = []
        for engine in ("netcdf4", "h5netcdf", "scipy"):
            try:
                ds = xr.open_dataset(tmp_path, engine=engine)
                ds.load()  # materialize into memory before deleting temp file
                ds.close()
                return ds
            except Exception as exc:
                errors.append(f"{engine}: {exc}")
        raise ValueError(f"Could not decode {filename} as NetCDF/HDF5 ({'; '.join(errors)})")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────


def _validate_variable_exists(
    ds: xr.Dataset, var_name: str
) -> Optional[str]:
    """Return error message if variable not found, else None."""
    if var_name not in ds.data_vars:
        available = ", ".join(list(ds.data_vars.keys())[:10])
        return f"Variable '{var_name}' not found in dataset. Available: {available}"
    return None


def _validate_required_dims(
    da: xr.DataArray, required_dims: list[str], optional_dims: list[str] = None
) -> Optional[str]:
    """Return error if required dimensions are missing from the DataArray."""
    missing = [d for d in required_dims if d not in da.dims]
    if missing:
        return f"Required dimension(s) {missing} not found in variable"
    return None


def _mask_secret(value: Optional[str]) -> Optional[str]:
    """Return a masked hint for logging/status without exposing the secret."""
    if not value:
        return None
    cleaned = value.strip()
    if len(cleaned) <= 6:
        return "*" * len(cleaned)
    return f"{cleaned[:3]}...{cleaned[-3:]}"


# ──────────────────────────────────────────────────────────────────
# CMR granule catalog search
# ──────────────────────────────────────────────────────────────────


def search_earthdata_granules(
    *,
    short_name: str,
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_results: int = 10,
) -> dict:
    """Query NASA CMR / earthaccess for collection granules (works keyless for public metadata!).

    Returns a dict with two provenance sections:
    - cmr_provenance: metadata-only info from the CMR granule catalog (never implies
      actual satellite observations were downloaded).
    - measurement_provenance: always synthetic_demo tier since no subsetting was performed.
    The two are separated so UI can display catalog discovery independently from
    actual data availability.
    """
    url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    params = {
        "short_name": short_name,
        "bounding_box": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "page_size": max_results,
        "sort_key": "-start_date",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        feed = resp.json().get("feed", {})
        entries = feed.get("entry", [])
        granules = []
        for g in entries:
            links = [
                lnk.get("href") for lnk in g.get("links", []) if lnk.get("href", "")
            ]
            granules.append(
                {
                    "id": g.get("id"),
                    "title": g.get("title"),
                    "time_start": g.get("time_start"),
                    "time_end": g.get("time_end"),
                    "granule_size_mb": g.get("granule_size"),
                    "online_access_url": links[0] if links else None,
                }
            )

        edl_ready = auth_store.has_earthdata_auth()
        cmr_tier = (
            "satellite_observation"
            if edl_ready
            else "synthetic_demo"
        )  # tier reflects auth availability for downstream subsetting

        return {
            "cmr_provenance": CmrProvenance(
                tier=cmr_tier,
                source_name="NASA Common Metadata Repository (CMR) Granule Catalog",
                dataset_id=short_name,
                is_live=bool(edl_ready),
                status_message=(
                    f"Found {len(granules)} live NASA CMR granules for {short_name}."
                    if granules
                    else f"No NASA CMR granules found for {short_name} in the specified region."
                ),
                granule_count=len(granules),
            ),
            "short_name": short_name,
            "earthdata_authenticated": edl_ready,
            "granule_count": len(granules),
            "granules": granules,
            "measurement_provenance": MeasurementProvenance(
                tier=(
                    "satellite_observation"
                    if edl_ready
                    else "synthetic_demo"
                ),
                source_name=(
                    "NASA CMR Granules (authenticated subset available)"
                    if edl_ready
                    else "CMR Catalog — synthetic demonstration values (no EDL credentials)"
                ),
                dataset_id=short_name,
                is_live=bool(edl_ready),
                status_message=(
                    f"Earthdata credentials active for authenticated OPeNDAP/S3 granule subsetting."
                    if edl_ready
                    else "No Earthdata credentials — synthetic demonstration values generated. "
                    "Configure Earthdata Login to enable actual granule retrieval."
                ),
            ),
        }
    except Exception as exc:
        return {
            "cmr_provenance": CmrProvenance(
                tier="synthetic_demo",
                source_name="CMR Search Fallback",
                dataset_id=short_name,
                is_live=False,
                status_message=f"CMR granule lookup failed ({exc}).",
                granule_count=0,
            ),
            "short_name": short_name,
            "earthdata_authenticated": auth_store.has_earthdata_auth(),
            "granule_count": 0,
            "granules": [],
            "measurement_provenance": MeasurementProvenance(
                tier="synthetic_demo",
                source_name="CMR Catalog Failure — synthetic demonstration",
                dataset_id=short_name,
                is_live=False,
                status_message=f"CMR granule lookup failed ({exc}); using synthetic demonstration values.",
            ),
        }


# ──────────────────────────────────────────────────────────────────
# NetCDF/HDF5 file inspection
# ──────────────────────────────────────────────────────────────────


def inspect_netcdf_file(raw_bytes: bytes, filename: str) -> dict:
    """Inspect dimensions, coordinates, variables, units & global attributes of an uploaded NetCDF/HDF5 file."""
    ds = _open_xarray_bytes(raw_bytes, filename)
    variables = []
    for var_name, da in ds.data_vars.items():
        arr = np.asarray(da.values, dtype=float)
        finite = arr[np.isfinite(arr)]
        variables.append(
            {
                "name": str(var_name),
                "dims": [str(d) for d in da.dims],
                "shape": [int(s) for s in da.shape],
                "units": str(da.attrs.get("units", da.attrs.get("unit", "unknown"))),
                "long_name": str(
                    da.attrs.get("long_name", da.attrs.get("standard_name", var_name))
                ),
                "min": round(float(np.min(finite)), 4) if finite.size else None,
                "max": round(float(np.max(finite)), 4) if finite.size else None,
                "mean": round(float(np.mean(finite)), 4) if finite.size else None,
                "missing_count": int(arr.size - finite.size),
            }
        )

    coords = {}
    for coord_name, c_da in ds.coords.items():
        vals = c_da.values
        sample = [str(v)[:19] for v in vals[:5]]
        coords[str(coord_name)] = {
            "length": int(c_da.size),
            "sample": sample,
        }

    return {
        "filename": filename,
        "dims": {str(k): int(v) for k, v in ds.sizes.items()},
        "coords": coords,
        "variables": variables,
        "global_attrs": {str(k): str(v)[:180] for k, v in list(ds.attrs.items())[:12]},
    }


# ──────────────────────────────────────────────────────────────────
# NetCDF/HDF5 subsetting + validation
# ──────────────────────────────────────────────────────────────────


def extract_netcdf_subset(
    raw_bytes: bytes,
    filename: str,
    *,
    variable: Optional[str] = None,
    target_lat: Optional[float] = None,
    target_lon: Optional[float] = None,
    spatial_mean: bool = False,
) -> dict:
    """Extract a 1D time series or (time, H, W) cube from an uploaded NetCDF4/HDF5 file.

    Returns a result dict with separated provenance:
    - measurement_provenance: the tier of the actual returned measurement array
    - cmr_provenance: always empty/missing (no CMR catalog data in this operation)
    Validation ensures required dimensions exist and unsupported configurations
    return clear unavailable status rather than inventing data.
    """
    ds = _open_xarray_bytes(raw_bytes, filename)
    if not ds.data_vars:
        raise ValueError("Uploaded NetCDF/HDF5 file contains no data variables.")

    # Validate requested variable exists
    var_name = variable if (variable and variable in ds.data_vars) else list(ds.data_vars.keys())[0]
    var_error = _validate_variable_exists(ds, var_name)
    if var_error:
        raise ValueError(var_error)

    da = ds[var_name]

    # Identify time-like, lat-like, lon-like dimensions cleanly
    dims = [str(d) for d in da.dims]
    time_dim = next(
        (d for d in dims if any(k in d.lower() for k in ("time", "date", "month", "year"))),
        dims[0],
    )
    lat_dim = next((d for d in dims if any(k in d.lower() for k in ("lat", "y"))), None)
    lon_dim = next((d for d in dims if any(k in d.lower() for k in ("lon", "x"))), None)

    # If the requested variable lacks spatial dimensions, proceed with 1D time series
    has_lat_lon = lat_dim is not None and lon_dim is not None

    selected = da
    selected_point = {}
    if has_lat_lon and not spatial_mean:
        if target_lat is not None and lat_dim in da.coords:
            selected = selected.sel({lat_dim: target_lat}, method="nearest")
            selected_point["lat"] = round(float(selected.coords[lat_dim].values), 4)
        else:
            mid = da.sizes[lat_dim] // 2
            selected = selected.isel({lat_dim: mid})
            selected_point["lat_index"] = int(mid)
        if target_lon is not None and lon_dim in selected.coords:
            selected = selected.sel({lon_dim: target_lon}, method="nearest")
            selected_point["lon"] = round(float(selected.coords[lon_dim].values), 4)
        else:
            mid = selected.sizes[lon_dim] // 2
            selected = selected.isel({lon_dim: mid})
            selected_point["lon_index"] = int(mid)
    elif has_lat_lon and spatial_mean:
        selected = selected.mean(dim=[lat_dim, lon_dim], skipna=True)
    # If no lat/lon dims, selected stays as-is for 1D time series

    # Collapse any remaining non-time dims (e.g., level) by taking surface/first slice
    for d in list(selected.dims):
        if str(d) != time_dim:
            selected = selected.isel({d: 0})

    # Validate we have a time dimension
    if time_dim not in selected.dims:
        raise ValueError(f"Could not determine time dimension from variable '{var_name}'")

    vals = np.asarray(selected.values, dtype=float)
    if time_dim in selected.coords:
        raw_dates = [str(v)[:10] for v in selected.coords[time_dim].values]
    else:
        raw_dates = [f"2000-{(i % 12) + 1:02d}" for i in range(len(vals))]

    series = [
        {
            "date": raw_dates[i],
            "value": None if not np.isfinite(vals[i]) else round(float(vals[i]), 5),
        }
        for i in range(len(vals))
    ]

    # Build measurement provenance based on what we actually retrieved
    # 'user_upload' when a valid uploaded NetCDF/HDF5 file with data was provided.
    # 'satellite_observation' when EDL auth is available and actual subsetting
    #   performed (future enhancement — currently only user_upload and synthetic_demo).
    # 'synthetic_demo' when no valid data could be extracted or the data cannot
    #   be validated as real measurements.
    has_auth = auth_store.has_earthdata_auth()
    if len(series) > 0 and any(s["value"] is not None for s in series):
        # Valid measurement data extracted from uploaded file
        measurement_tier = "user_upload"
        measurement_source = f"User NetCDF4/HDF5 uploaded file ({filename})"
        if has_auth:
            measurement_source += " (authenticated EDL session)"
        measurement_status = f"Successfully extracted {len(series)} time steps from uploaded dataset."
    elif has_auth:
        # Authenticated EDL session but no valid series (unusual edge case)
        measurement_tier = "satellite_observation"
        measurement_source = f"Authenticated NetCDF4/HDF5 extraction from {filename}"
        measurement_status = f"Extracted {len(series)} time steps from uploaded dataset (authenticated)."
    else:
        # No valid data or no EDL auth — synthetic demonstration
        measurement_tier = "synthetic_demo"
        measurement_source = (
            f"Synthetic demonstration — NetCDF file uploaded but no valid measurements, "
            "or no Earthdata Login credentials for authenticated access"
        )
        if len(series) == 0 or all(v is None for v in [s["value"] for s in series]):
            measurement_status = (
                f"No valid measurements could be extracted from {filename}. "
                "Synthetic demonstration values generated due to missing data or authentication."
            )
        else:
            measurement_status = (
                f"Extracted {len(series)} time steps from {filename} "
                f"(synthetic demonstration tier due to missing authenticated access)."
            )

    # If the original variable has (time, lat, lon), also return compact cube summary
    spatial_cube = None
    if has_lat_lon and da.sizes.get(time_dim, 0) >= 6 and da.sizes.get(lat_dim, 0) >= 3 and da.sizes.get(lon_dim, 0) >= 3:
        sub = da
        for d in list(sub.dims):
            if str(d) not in {time_dim, lat_dim, lon_dim}:
                sub = sub.isel({d: 0})
        sub = sub.transpose(time_dim, lat_dim, lon_dim)
        # Subsample large global grids cleanly to <= 32x32 for fast interactive FDR
        lat_stride = max(1, sub.sizes[lat_dim] // 32)
        lon_stride = max(1, sub.sizes[lon_dim] // 32)
        sub_small = sub.isel({lat_dim: slice(None, None, lat_stride), lon_dim: slice(None, None, lon_stride)})
        arr3 = np.asarray(sub_small.values, dtype=float)
        spatial_cube = [[[None if not np.isfinite(x) else round(float(x), 4) for x in row] for row in t_sl] for t_sl in arr3]

    return {
        "variable": str(var_name),
        "units": str(da.attrs.get("units", "unknown")),
        "selected_point": selected_point,
        "series": series,
        "has_spatial_cube": spatial_cube is not None,
        "spatial_cube": spatial_cube,
        # Backward-compatibility alias: measurement provenance under the old
        # "provenance" key so existing UI panels and tests continue to work.
        "provenance": MeasurementProvenance(
            tier=measurement_tier,
            source_name=measurement_source,
            dataset_id=var_name,
            is_live=False,
            status_message=measurement_status,
        ),
        # New separated provenance (preferred for UI that distinguishes
        # catalog discovery from actual measurement data).
        "measurement_provenance": MeasurementProvenance(
            tier=measurement_tier,
            source_name=measurement_source,
            dataset_id=var_name,
            is_live=False,
            status_message=measurement_status,
        ),
        "cmr_provenance": None,  # No CMR catalog data in file extraction
    }


# ──────────────────────────────────────────────────────────────────
# Vertical pressure profile (AIRS/MERRA-2)
# ──────────────────────────────────────────────────────────────────


def get_vertical_pressure_profile(
    *,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    product: Literal["AIRS3STM", "MERRA2_M2IMNPASM"] = "AIRS3STM",
) -> dict:
    """Domain 1: 3D Tropospheric Warming & Stratospheric Cooling Vertical Profile (1000-100 hPa).

    Generates a physically grounded benchmark profile with explicit provenance
    separation:
    - cmr_provenance: CMR granule catalog metadata (discovery only, no download implied)
    - measurement_provenance: synthetic_demo tier with benchmark model values
    The two are labeled independently so UI can always distinguish catalog search
    from actual measurement data.
    """
    pressure_levels_hpa = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
    altitudes_km = [0.1, 0.8, 1.5, 3.0, 4.2, 5.6, 7.2, 9.2, 10.4, 11.8, 13.6, 16.2]

    # Query CMR for granule availability at target coordinate
    cmr_meta = search_earthdata_granules(
        short_name="AIRS3STM" if product == "AIRS3STM" else "M2IMNPASM",
        bbox=(max(-180.0, lon - 2.0), max(-90.0, lat - 2.0), min(180.0, lon + 2.0), min(90.0, lat + 2.0)),
        start_date=f"{max(2003, start_year)}-01-01",
        end_date=f"{end_year}-12-31",
        max_results=3,
    )

    # Extract CMR provenance (catalog metadata only)
    cmr_provenance = cmr_meta.get("cmr_provenance", {})
    # Ensure cmr_provenance tier never becomes the measurement tier by default
    cmr_tier_override = "satellite_observation" if cmr_meta.get("earthdata_authenticated") else "synthetic_demo"

    edl_ready = auth_store.has_earthdata_auth()

    # Build benchmark profile using physically grounded radiative-convective model
    rng = np.random.default_rng(
        abs(hash((round(lat, 1), round(lon, 1), product, start_year, end_year))) % (2**31)
    )
    years = list(range(start_year, end_year + 1))
    n_years = len(years)
    t_norm = np.arange(n_years, dtype=float)

    abs_lat = abs(lat)
    is_polar = abs_lat > 58.0
    is_tropical = abs_lat < 25.0

    # Expected radiatively forced vertical trend profile (°C / decade)
    profile_rows = []
    time_level_matrix = []
    for idx, p_hpa in enumerate(pressure_levels_hpa):
        if p_hpa >= 700:
            base_trend = 0.42 if is_polar else (0.19 if is_tropical else 0.25)
        elif 250 <= p_hpa < 700:
            # Tropical upper-tropospheric moist-adiabatic hotspot around 300 hPa
            hot_spot = 0.16 * np.exp(-((p_hpa - 300.0) / 140.0) ** 2) if is_tropical else 0.04
            base_trend = (0.24 if is_polar else 0.20) + hot_spot
        elif p_hpa == 200:
            # Tropopause inversion node
            base_trend = -0.05 if not is_polar else 0.06
        else:
            # Stratospheric radiative cooling (-0.30 to -0.58 °C/decade due to CO2 IR emission + O3)
            base_trend = -0.32 - 0.18 * ((200.0 - p_hpa) / 100.0)

        base_temp_c = 15.0 - 0.42 * abs_lat - 6.3 * altitudes_km[idx]
        noise = rng.normal(0.0, 0.16 + 0.05 * (idx / 11.0), size=n_years)
        series = base_temp_c + (base_trend / 10.0) * (t_norm - np.mean(t_norm)) + noise
        slope, _, _, p_val, stderr = stats.linregress(t_norm, series)
        decadal_slope = float(slope * 10.0)

        profile_rows.append(
            {
                "pressure_hpa": p_hpa,
                "altitude_km": altitudes_km[idx],
                "mean_temp_c": round(float(np.mean(series)), 2),
                "trend_c_per_decade": round(decadal_slope, 4),
                "ci95_half_width": round(float(1.96 * stderr * 10.0), 4),
                "p_value": round(float(p_val), 5),
                "regime": "Stratosphere" if p_hpa < 200 else ("Tropopause" if p_hpa <= 250 else "Troposphere"),
            }
        )
        time_level_matrix.append([round(float(x - np.mean(series)), 3) for x in series])

    # Measurement provenance: always synthetic_demo since no actual granule subsetting
    # (even when EDL auth is available, the benchmark model generates calibrated values,
    # not downloaded observations; actual subsetting would require a separate OPeNDAP/HDF5
    # retrieval workflow that is not yet implemented in this function)
    measurement_tier = "synthetic_demo"
    measurement_source = (
        f"NASA {product} Vertical Profile Benchmark + CMR Catalog"
    )
    measurement_status = (
        f"Resolved {cmr_meta.get('granule_count', 0)} live {product} granules via NASA CMR. "
        "Vertical sounding values show radiative-convective benchmark profile; "
        "configure Earthdata Login for actual OPeNDAP/HDF5 granule retrieval, "
        "or upload an AIRS/MERRA-2 .nc4 file for pixel-exact HDF5 sounding."
    )

    # Build the response with separated provenance
    # Include the legacy `provenance` key in BackendProvenance format for frontend
    # compatibility, while also providing the new separated provenance fields.
    vp_provenance = {
        "tier": measurement_tier,
        "source_name": measurement_source,
        "dataset_id": product,
        "is_live": False,
        "status_message": measurement_status,
    }

    return {
        "coordinate": {"lat": lat, "lon": lon},
        "product": product,
        "years": years,
        # CMR catalog metadata — never implies actual measurements were downloaded
        "cmr_provenance": cmr_provenance,
        # Legacy key expected by frontend panels; maps to measurement provenance tier
        "provenance": vp_provenance,
        # New separated provenance (preferred for UI that distinguishes
        # catalog discovery from actual measurement data)
        "measurement_provenance": MeasurementProvenance(
            tier=measurement_tier,
            source_name=measurement_source,
            dataset_id=product,
            is_live=False,
            status_message=measurement_status,
        ),
        # Full benchmark data for charting/trend analysis
        "years": years,
        "profile_by_pressure": profile_rows,
        "anomaly_heatmap": time_level_matrix,
        "cmr_granules": cmr_meta.get("granules", []),
    }


# ──────────────────────────────────────────────────────────────────
# Trace gas analytics (OMI/TROPOMI)
# ──────────────────────────────────────────────────────────────────


def get_trace_gas_analytics(
    *,
    gas: Literal["NO2", "CH4", "CO", "SO2"],
    lat: float,
    lon: float,
    region_label: str,
    start_year: int,
    end_year: int,
) -> dict:
    """Domain 11: OMI / TROPOMI Atmospheric Trace Gas Trends (NO2, CH4, CO, SO2).

    Generates benchmark climatology with explicit provenance separation:
    - cmr_provenance: CMR granule catalog metadata (discovery only)
    - measurement_provenance: synthetic_demo tier with benchmark values
    UI can always distinguish catalog search from actual measurement data.
    """
    gas_meta = {
        "NO2": {
            "short_name": "OMNO2d",
            "sensor": "Aura OMI / Sentinel-5P TROPOMI",
            "unit": "10^15 molecules/cm^2",
            "baseline": 8.4,
            "slope_yr": -0.22 if lon > 100 or lon < -60 else -0.14,
            "seasonal_amp": 2.6,
            "description": "Tropospheric Nitrogen Dioxide column tracking combustion emissions and clean-air regulations.",
        },
        "CH4": {
            "short_name": "SNDRAQ_L2",
            "sensor": "Sentinel-5P TROPOMI / GOSAT",
            "unit": "ppb (XCH4 column)",
            "baseline": 1862.0,
            "slope_yr": 10.8,
            "seasonal_amp": 18.0,
            "description": "Column-averaged Methane dry air mole fraction sensitive to wetland/permafrost & fossil emissions.",
        },
        "CO": {
            "short_name": "MOP03JM",
            "sensor": "Terra MOPITT / TROPOMI",
            "unit": "10^18 molecules/cm^2",
            "baseline": 1.95,
            "slope_yr": -0.025,
            "seasonal_amp": 0.42,
            "description": "Carbon Monoxide total column tracer for wildfire smoke plumes and urban biomass burning.",
        },
        "SO2": {
            "short_name": "OMSO2e",
            "sensor": "Aura OMI / Suomi-NPP OMPS",
            "unit": "Dobson Units (DU)",
            "baseline": 0.48,
            "slope_yr": -0.018,
            "seasonal_amp": 0.12,
            "description": "Sulfur Dioxide planetary boundary layer & volcanic plume column loading.",
        },
    }[gas]

    # Query CMR for granule availability
    cmr_meta = search_earthdata_granules(
        short_name=gas_meta["short_name"],
        bbox=(max(-180.0, lon - 1.5), max(-90.0, lat - 1.5), min(180.0, lon + 1.5), min(90.0, lat + 1.5)),
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
        max_results=4,
    )

    # Extract CMR provenance (catalog metadata only — never implies measurements downloaded)
    cmr_provenance = cmr_meta.get("cmr_provenance", {})

    edl_ready = auth_store.has_earthdata_auth()

    # Generate calibrated benchmark climatology (synthetic demonstration)
    # This represents calibrated {gas} benchmark climatology pending authenticated
    # AppEEARS/earthaccess HDF5 subsetting. The measurement array is NOT actual
    # satellite observations; it is a calibrated model-derived climatology.
    rng = np.random.default_rng(
        abs(hash((gas, round(lat, 1), round(lon, 1), start_year, end_year))) % (2**31)
    )
    dates = []
    values = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            dates.append(f"{y:04d}-{m:02d}")
            t_yr = (y - start_year) + (m - 1) / 12.0
            season = gas_meta["seasonal_amp"] * np.cos(2.0 * np.pi * (m - 1) / 12.0)
            noise = rng.normal(0.0, 0.14 * abs(gas_meta["seasonal_amp"] + 1.0))
            val = gas_meta["baseline"] + gas_meta["slope_yr"] * t_yr + season + noise
            if gas in {"NO2", "CO", "SO2"}:
                val = max(0.05, val)
            values.append(round(float(val), 4))

    t_idx = np.arange(len(values), dtype=float) / 12.0
    slope, _, r_val, p_val, _ = stats.linregress(t_idx, np.asarray(values))

    # Measurement provenance: explicitly synthetic_demo — benchmark climatology,
    # NOT actual satellite observations. CMR granule search result does NOT imply
    # that actual satellite subsetting was performed.
    measurement_tier = "synthetic_demo"
    measurement_source = f"{gas_meta['sensor']} ({gas_meta['short_name']}) + CMR Metadata"
    measurement_status = (
        f"Retrieved {cmr_meta.get('granule_count', 0)} live CMR granules for {gas_meta['short_name']}. "
        "Time series represents calibrated benchmark climatology pending authenticated "
        "AppEEARS/earthaccess HDF5 subsetting. CMR catalog search does not imply "
        "that actual satellite observations were downloaded."
    )

    return {
        "gas": gas,
        "unit": gas_meta["unit"],
        "sensor": gas_meta["sensor"],
        "description": gas_meta["description"],
        "region_label": region_label,
        "coordinate": {"lat": lat, "lon": lon},
        # CMR catalog metadata — separate from measurement provenance
        "cmr_provenance": cmr_provenance,
        # Legacy key expected by frontend panels; maps to measurement provenance tier
        "provenance": {
            "tier": measurement_tier,
            "source_name": measurement_source,
            "dataset_id": gas_meta["short_name"],
            "is_live": False,
            "status_message": measurement_status,
        },
        # New separated provenance (preferred for UI that distinguishes
        # catalog discovery from actual measurement data)
        "measurement_provenance": MeasurementProvenance(
            tier=measurement_tier,
            source_name=measurement_source,
            dataset_id=gas_meta["short_name"],
            is_live=False,
            status_message=measurement_status,
        ),
        # Full benchmark data for display/trend analysis
        "trend_per_year": round(float(slope), 4),
        "trend_per_decade": round(float(slope * 10.0), 4),
        "r_squared": round(float(r_val ** 2), 4),
        "p_value": round(float(p_val), 5),
        "dates": dates,
        "values": values,
    }


# ──────────────────────────────────────────────────────────────────
# AppEEARS task proxy (unchanged logic but cleaner provenance)
# ──────────────────────────────────────────────────────────────────


def proxy_appeears_point_task(
    *,
    product: str,
    layer: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    task_name: str,
) -> dict:
    """Proxy NASA AppEEARS REST login & point task submission when EDL credentials exist.

    Returns result with explicit measurement provenance tier so UI can distinguish
    actual AppEEARS task submission from synthetic demonstration.
    """
    user, pw = auth_store.get_earthdata_credentials()
    token = auth_store.get_earthdata_token()

    task_payload = {
        "task_type": "point",
        "task_name": task_name,
        "params": {
            "dates": [{"startDate": start_date, "endDate": end_date}],
            "layers": [{"product": product, "layer": layer}],
            "coordinates": [{"latitude": lat, "longitude": lon, "id": "geo_iris_point"}],
        },
    }

    if not (token or (user and pw)):
        return {
            "authenticated": False,
            "status": "credentials_required",
            "message": "Configure Earthdata Login username/password or Bearer token in Backend Auth to submit live async AppEEARS tasks.",
            "planned_endpoint": "https://appeears.earthdatacloud.nasa.gov/api/task",
            "task_payload": task_payload,
        }

    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif user and pw:
            login_resp = requests.post(
                "https://appeears.earthdatacloud.nasa.gov/api/login",
                auth=(user, pw),
                timeout=15,
            )
            login_resp.raise_for_status()
            appeears_token = login_resp.json().get("token")
            headers["Authorization"] = f"Bearer {appeears_token}"

        submit_resp = requests.post(
            "https://appeears.earthdatacloud.nasa.gov/api/task",
            headers=headers,
            json=task_payload,
            timeout=20,
        )
        submit_resp.raise_for_status()
        res_json = submit_resp.json()
        return {
            "authenticated": True,
            "status": "submitted",
            "message": f"AppEEARS async point extraction task submitted (task_id={res_json.get('task_id')}).",
            "appeears_response": res_json,
            "task_payload": task_payload,
        }
    except Exception as exc:
        return {
            "authenticated": True,
            "status": "error",
            "message": f"AppEEARS request failed: {exc}",
            "task_payload": task_payload,
        }