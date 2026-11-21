"""Pydantic v2 request and response schemas for Geo-Iris Phase 2 API."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

ProvenanceTier = Literal[
    "satellite_observation",
    "reanalysis_model",
    "user_upload",
    "synthetic_demo",
]


class ProvenanceMetadata(BaseModel):
    tier: ProvenanceTier
    source_name: str
    dataset_id: str
    is_live: bool
    status_message: str
    citation: Optional[str] = None


class AuthStatusResponse(BaseModel):
    earthdata_configured: bool
    earthdata_source: str
    earthdata_username_hint: Optional[str] = None
    earthdata_token_hint: Optional[str] = None
    firms_configured: bool
    firms_source: str
    firms_key_hint: Optional[str] = None
    keyless_feeds_available: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    capabilities: dict[str, bool]
    auth: AuthStatusResponse


# ---------------------------------------------------------------------------
# Advanced Statistical & Mathematical Schemas
# ---------------------------------------------------------------------------

class TimeSeriesInputPoint(BaseModel):
    date: str
    value: Optional[float] = None


class SarimaRequest(BaseModel):
    dates: list[str] = Field(min_length=24, max_length=100_000)
    values: list[Optional[float]] = Field(min_length=24, max_length=100_000)
    forecast_steps: int = Field(default=12, ge=1, le=60)
    seasonal_period: int = Field(default=12, ge=1, le=24)
    auto_order: bool = True
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 12)
    holdout_size: int = Field(default=6, ge=0, le=24)
    variable_name: Optional[str] = "Environmental Series"


class BayesianMcmcRequest(BaseModel):
    dates: list[str] = Field(min_length=12, max_length=100_000)
    values: list[Optional[float]] = Field(min_length=12, max_length=100_000)
    model_type: Literal["robust_ar1", "changepoint"] = "robust_ar1"
    chains: int = Field(default=3, ge=2, le=4)
    draws: int = Field(default=900, ge=300, le=2500)
    burn_in: int = Field(default=450, ge=150, le=1200)
    seed: int = 42


class ExtremeValueRequest(BaseModel):
    values: list[Optional[float]] = Field(min_length=10, max_length=100_000)
    dates: Optional[list[str]] = None
    mode: Literal["gev_block_maxima", "gpd_pot"] = "gev_block_maxima"
    block_size: int = Field(default=12, ge=2, le=60)
    threshold_quantile: float = Field(default=0.85, ge=0.5, le=0.98)
    return_periods: list[float] = Field(default_factory=lambda: [2, 5, 10, 20, 50, 100])
    observation_frequency: Optional[Literal["daily", "monthly", "annual", "irregular"]] = None


class SpatialFdrRequest(BaseModel):
    """Run vectorized Mann-Kendall across a 3D (T, H, W) spatio-temporal grid.

    Temporal autocorrelation correction (per pixel, independent of spatial FDR):
      - "none": no correction; standard MK assumes temporal independence.
      - "hamed_rao": Hamed & Rao (1998) variance modification using detrended
        rank autocorrelation at consecutive-year lags. Missing/invalid data and
        irregular sampling are handled explicitly.

    Spatial FDR methods (BH/BY/Holm/Bonferroni) correct multiple testing across
    pixels; they do NOT correct temporal autocorrelation within each pixel's time
    series. Use `temporal_autocorr_correction` for that purpose.
    """

    region_name: Optional[str] = "Arctic & Boreal Sub-Grid"
    center_lat: float = Field(default=68.0, ge=-90.0, le=90.0)
    center_lon: float = Field(default=20.0, ge=-180.0, le=180.0)
    grid_rows: int = Field(default=20, ge=5, le=64)
    grid_cols: int = Field(default=24, ge=5, le=64)
    n_years: int = Field(default=20, ge=8, le=45)
    alpha: float = Field(default=0.05, ge=0.001, le=0.25)
    correction_method: Literal["bh", "by", "holm", "bonferroni"] = "by"
    cube: Optional[list[list[list[Optional[float]]]]] = None
    seed: int = 42
    timestamps: Optional[list[str]] = Field(
        default=None,
        description="Explicit timestamps (e.g. 'YYYY', 'YYYY-MM', 'YYYY-MM-DD'). When supplied, converted to decimal years for the temporal slope. Must match the number of time steps.",
    )
    time_step_years: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Uniform time step between samples in years (e.g. 1/12 for monthly). If provided, temporal slope is computed on the correct decimal-year coordinate.",
    )
    temporal_autocorr_correction: Literal["none", "hamed_rao"] = Field(
        default="none",
        description="Temporal autocorrelation correction applied per-pixel before FDR. Options: 'none' (standard MK assumes independence); 'hamed_rao' (Hamed & Rao 1998 variance correction for lag-1 autocorrelation).",
    )


# ---------------------------------------------------------------------------
# NASA Proxy & Earthdata Schemas
# ---------------------------------------------------------------------------

class FirmsQueryRequest(BaseModel):
    west: float = Field(default=-125.0, ge=-180.0, le=180.0)
    south: float = Field(default=32.0, ge=-90.0, le=90.0)
    east: float = Field(default=-114.0, ge=-180.0, le=180.0)
    north: float = Field(default=43.0, ge=-90.0, le=90.0)
    source: Literal["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"] = "VIIRS_SNPP_NRT"
    day_range: int = Field(default=5, ge=1, le=10)
    region_label: Optional[str] = "California & Western Sierra"
    allow_eonet_fallback: bool = True


class VerticalProfileRequest(BaseModel):
    lat: float = Field(default=0.0, ge=-90.0, le=90.0)
    lon: float = Field(default=-60.0, ge=-180.0, le=180.0)
    start_year: int = Field(default=2003, ge=1980, le=2025)
    end_year: int = Field(default=2024, ge=1981, le=2026)
    product: Literal["AIRS3STM", "MERRA2_M2IMNPASM"] = "AIRS3STM"


class TraceGasRequest(BaseModel):
    gas: Literal["NO2", "CH4", "CO", "SO2"] = "NO2"
    lat: float = Field(default=31.2, ge=-90.0, le=90.0)
    lon: float = Field(default=121.5, ge=-180.0, le=180.0)
    region_label: Optional[str] = "Yangtze River Delta Industrial Corridor"
    start_year: int = Field(default=2018, ge=2004, le=2025)
    end_year: int = Field(default=2025, ge=2005, le=2026)


class EarthdataGranuleRequest(BaseModel):
    short_name: str = Field(default="AIRS3STM", min_length=2, max_length=64)
    bbox: tuple[float, float, float, float] = (-10.0, 40.0, 20.0, 60.0)
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    max_results: int = Field(default=10, ge=1, le=30)


class AppeearsPointRequest(BaseModel):
    product: str = Field(default="MOD13Q1.061", min_length=3, max_length=64)
    layer: str = Field(default="_250m_16_days_NDVI", min_length=2, max_length=128)
    lat: float = Field(default=-3.1, ge=-90.0, le=90.0)
    lon: float = Field(default=-60.0, ge=-180.0, le=180.0)
    start_date: str = "01-01-2020"
    end_date: str = "12-31-2024"
    task_name: str = "geo_iris_point_sample"


# ---------------------------------------------------------------------------
# Raster / Landform / SAR Schemas
# ---------------------------------------------------------------------------

class DemDiffPresetRequest(BaseModel):
    scenario: Literal[
        "mount_st_helens_lahar",
        "colorado_river_incision",
        "thwaites_glacier_thinning",
    ] = "mount_st_helens_lahar"
    lod_threshold_m: float = Field(default=0.75, ge=0.0, le=10.0)
    pixel_size_m: float = Field(default=10.0, ge=0.5, le=1000.0)


class SarChangePresetRequest(BaseModel):
    scenario: Literal[
        "amazon_seasonal_flooding",
        "rondonia_deforestation_clearcut",
        "kilauea_lava_emplacement",
    ] = "amazon_seasonal_flooding"
    speckle_filter: Literal["none", "lee", "mean"] = "lee"
    window_size: int = Field(default=5, ge=3, le=11)
    threshold_db: float = Field(default=3.0, ge=0.5, le=15.0)
    threshold_mode: Literal["fixed", "otsu"] = "fixed"
    pixel_size_m: float = Field(default=20.0, ge=1.0, le=1000.0)


class LulcPresetRequest(BaseModel):
    scenario: Literal[
        "amazon_arc_deforestation",
        "pearl_river_urbanization",
        "aral_sea_desiccation",
    ] = "amazon_arc_deforestation"


class AsfSearchRequest(BaseModel):
    lat: float = Field(default=-3.46, ge=-90.0, le=90.0)
    lon: float = Field(default=-62.21, ge=-180.0, le=180.0)
    radius_deg: float = Field(default=0.4, ge=0.05, le=3.0)
    platform: Literal["SENTINEL-1", "ALOS", "UAVSAR", "SEASAT"] = "SENTINEL-1"
    start_date: str = "2024-01-01T00:00:00Z"
    end_date: str = "2025-12-31T23:59:59Z"
    max_results: int = Field(default=12, ge=1, le=30)


# ---------------------------------------------------------------------------
# Raster Interpolation & Gap-Filling Schemas
# ---------------------------------------------------------------------------

class RasterInspectRequest(BaseModel):
    """Request to inspect a raster for NoData and missing-data diagnostics."""
    band_index: int = Field(default=1, ge=1, description="1-based band index to inspect.")
    user_confirmed_type: Optional[Literal["continuous", "categorical", "temporal", "unknown"]] = Field(
        default=None,
        description="User-confirmed raster semantics. If omitted, inferred from dtype.",
    )


class RasterInspectResponse(BaseModel):
    width: int
    height: int
    bands: int
    crs: Optional[str] = None
    transform: Optional[dict] = None
    pixel_size_x: Optional[float] = None
    pixel_size_y: Optional[float] = None
    extent: Optional[tuple[float, float, float, float]] = None
    data_type: str
    nodata_value: Optional[float] = None
    has_alpha_band: bool = False
    band_index: int

    valid_pixel_count: int
    missing_pixel_count: int
    missing_percentage: float
    valid_percentage: float

    band_stats: list[dict]

    raster_type: str

    missing_region_count: int
    missing_region_stats: list[dict]
    largest_missing_region_pixels: int
    average_missing_region_size: float
    missing_region_size_distribution: dict
    isolated_missing_pixels: int
    edge_gaps: int
    interior_gaps: int

    distance_to_valid_stats: dict
    missing_spatial_distribution: dict

    processing_duration_ms: float
    inspected_at: str


class InterpolateRequest(BaseModel):
    """Request to interpolate missing values in a raster band."""
    data: Any = Field(
        description="Raster array as nested lists (2D for spatial, 3D (T,H,W) for temporal). null = NoData."
    )
    method: Literal[
        "nearest_neighbor", "bilinear", "bicubic", "local_mean",
        "local_median", "idw", "ordinary_kriging",
        "temporal_linear", "temporal_nearest_date",
        "temporal_forward_fill", "temporal_backward_fill",
        "temporal_local_mean",
    ] = "nearest_neighbor"

    band_index: int = Field(default=1, ge=1)
    raster_type: Literal["continuous", "categorical", "temporal", "unknown"] = "continuous"

    # Spatial parameters
    max_distance_pixels: Optional[float] = Field(
        default=None, ge=0, description="Maximum search radius in pixels."
    )
    geographic_crs: bool = Field(default=False)
    pixel_size_x: Optional[float] = Field(default=None, gt=0)
    pixel_size_y: Optional[float] = Field(default=None, gt=0)

    # Nearest neighbor
    # (uses max_distance_pixels)

    # Bilinear / bicubic
    neighborhood: Optional[Literal["3x3", "5x5", "4x4", "6x6"]] = None
    edge_handling: Optional[Literal["clamp", "nearest_valid", "nodata"]] = None
    clip_to_observed_range: Optional[bool] = None

    # Local mean / median
    window_size: Optional[int] = Field(default=None, ge=3, le=21)
    min_valid_neighbors: Optional[int] = Field(default=None, ge=1, le=20)
    neighborhood_shape: Optional[Literal["square", "circle"]] = None

    # IDW
    power: Optional[float] = Field(default=None, gt=0, le=10)
    search_radius_pixels: Optional[float] = Field(default=None, ge=0)
    max_neighbors: Optional[int] = Field(default=None, ge=1, le=50)
    min_neighbors: Optional[int] = Field(default=None, ge=1, le=20)

    # Ordinary kriging
    variogram_model: Optional[Literal["spherical", "exponential", "gaussian", "linear"]] = None
    variogram_nugget: Optional[float] = Field(default=None, ge=0)
    variogram_sill: Optional[float] = Field(default=None, gt=0)
    variogram_range: Optional[float] = Field(default=None, gt=0)

    # Temporal
    max_gap_steps: Optional[int] = Field(default=None, ge=1)
    allow_extrapolation: bool = Field(default=False)

    # Temporal cube data (for temporal methods)
    times: Optional[list[float]] = Field(
        default=None,
        description="Decimal-year time coordinates for temporal interpolation (length = T)."
    )


class InterpolateResponse(BaseModel):
    method: str
    band_index: int
    raster_type: str
    config: dict

    observed_pixel_count: int
    interpolated_pixel_count: int
    remaining_nodata_count: int
    excluded_pixel_count: int
    processing_duration_ms: float

    # Provenance mask as nested list of strings
    provenance_mask: list[list[str]]

    # Reconstructed array as nested list (NaN -> null)
    reconstructed: list[list[Optional[float]]]

    # Quality raster (optional)
    quality_mask: Optional[list[list[Optional[float]]]] = None

    # Distance to observation (IDW/NN diagnostic)
    distance_to_observation: Optional[list[list[Optional[float]]]] = None

    # Kriging variance (kriging only)
    kriging_variance: Optional[list[list[Optional[float]]]] = None

    # Variogram fit info (kriging only)
    variogram_fit: Optional[dict] = None


class ValidateRequest(BaseModel):
    """Request to cross-validate an interpolation method."""
    data: list[list[Optional[float]]] = Field(
        description="2D raster array as nested lists (null = NoData)."
    )
    method: Literal[
        "nearest_neighbor", "bilinear", "bicubic", "local_mean",
        "local_median", "idw", "ordinary_kriging",
    ] = "nearest_neighbor"

    band_index: int = Field(default=1, ge=1)
    raster_type: Literal["continuous", "categorical", "temporal", "unknown"] = "continuous"

    validation_strategy: Literal["random_holdout", "spatial_block"] = "spatial_block"
    sample_fraction: float = Field(default=0.1, ge=0.01, le=0.5)
    n_blocks: int = Field(default=4, ge=2, le=16)
    block_index: Optional[int] = Field(default=None, ge=0)
    seed: int = Field(default=42)

    max_distance_pixels: Optional[float] = Field(default=None, ge=0)
    geographic_crs: bool = Field(default=False)
    pixel_size_x: Optional[float] = Field(default=None, gt=0)
    pixel_size_y: Optional[float] = Field(default=None, gt=0)

    # Method-specific parameters (same as InterpolateRequest)
    neighborhood: Optional[Literal["3x3", "5x5", "4x4", "6x6"]] = None
    edge_handling: Optional[Literal["clamp", "nearest_valid", "nodata"]] = None
    clip_to_observed_range: Optional[bool] = None
    window_size: Optional[int] = Field(default=None, ge=3, le=21)
    min_valid_neighbors: Optional[int] = Field(default=None, ge=1, le=20)
    neighborhood_shape: Optional[Literal["square", "circle"]] = None
    power: Optional[float] = Field(default=None, gt=0, le=10)
    search_radius_pixels: Optional[float] = Field(default=None, ge=0)
    max_neighbors: Optional[int] = Field(default=None, ge=1, le=50)
    min_neighbors: Optional[int] = Field(default=None, ge=1, le=20)
    variogram_model: Optional[Literal["spherical", "exponential", "gaussian", "linear"]] = None
    variogram_nugget: Optional[float] = Field(default=None, ge=0)
    variogram_sill: Optional[float] = Field(default=None, gt=0)
    variogram_range: Optional[float] = Field(default=None, gt=0)


class ValidateResponse(BaseModel):
    method: str
    config: dict
    validation_strategy: str
    validation_sample_count: int
    n_excluded_from_validation: int

    mae: Optional[float] = None
    rmse: Optional[float] = None
    bias: Optional[float] = None
    median_absolute_error: Optional[float] = None
    mape: Optional[float] = None
    r_squared: Optional[float] = None

    reconstruction_percentage: float
    remaining_nodata_percentage: float
    excluded_percentage: float

    processing_duration_ms: float
    warnings: list[str]
    limitations: list[str]

    validation_predicted: list[float]
    validation_observed: list[float]
    validation_residuals: list[float]
    validation_coords: list[list[int]]


class CompatibleMethodsResponse(BaseModel):
    raster_type: str
    methods: list[dict]


class ExportRequest(BaseModel):
    """Request to export reconstructed raster as GeoTIFF."""
    output_filename: str = Field(default="reconstructed.tif", min_length=1, max_length=255)
    nodata: float = Field(default=-9999.0)
    include_provenance_band: bool = True
    include_quality_band: bool = True

    # Raster data (sent from frontend after interpolation)
    reconstructed_array: list[list[Optional[float]]] = Field(
        description="Reconstructed 2D array as nested lists (null = NoData)."
    )
    provenance_mask: list[list[str]] = Field(
        description="Provenance class per pixel ('observed'|'interpolated'|'nodata'|'excluded')."
    )
    quality_mask: Optional[list[list[Optional[float]]]] = Field(
        default=None,
        description="Quality score per pixel (0-1, null = no quality data)."
    )
    crs: Optional[str] = Field(default=None, description="Coordinate reference system.")
    transform: Optional[dict] = Field(
        default=None,
        description="Affine transform as dict with keys a,b,c,d,e,f."
    )
    method: str = Field(default="unknown", description="Interpolation method used.")
