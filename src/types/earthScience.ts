export type VariableId = 
  | 'temp_air'
  | 'precip'
  | 'ndvi'
  | 'soil_moist'
  | 'evapo'
  | 'aod'
  | 'lst_day'
  | 'ozone'
  // Phase 1 expanded variables
  | 'pressure'
  | 'humidity'
  | 'specific_humidity'
  | 'wind_speed'
  | 'radiation_sw'
  | 'radiation_lw'
  | 'albedo'
  | 'snow_depth'
  | 'snow_precip'
  | 'ozone_total'
  | 'soil_moist_top'
  | 'soil_moist_prof'
  | 'sea_level'
  | 'sst'
  | 'streamflow'
  | 'evi';

export type VariableCategory = 'Atmosphere' | 'Land' | 'Hydrosphere' | 'Biosphere' | 'Cryosphere' | 'Ocean';

export type MeasurementType = 'Satellite Instrument' | 'Global Assimilation Model' | 'Combined Reanalysis' | 'In-Situ Gauge Network';

export type ProvenanceTier =
  | 'satellite_observation' // ORNL MODIS, satellite imagery
  | 'in_situ_gauge'         // NOAA tide gauges, USGS streamflow, in-situ sensors
  | 'reanalysis_model'      // NASA POWER (MERRA-2/GLDAS)
  | 'user_upload'           // CSV/JSON uploaded by user
  | 'synthetic_demo';       // Calibrated model fallback data

export const PROVENANCE_META: Record<ProvenanceTier, {
  label: string;
  shortLabel: string;
  icon: string;
  colorClass: string;
  description: string;
}> = {
  satellite_observation: {
    label: 'Satellite Observation',
    shortLabel: 'SAT OBS',
    icon: '🛰️',
    colorClass: 'text-silver-500',
    description: 'Direct satellite instrument measurements (ORNL MODIS, satellite imagery)',
  },
  in_situ_gauge: {
    label: 'In-Situ Gauge',
    shortLabel: 'IN-SITU',
    icon: '🌊',
    colorClass: 'text-cyan-400',
    description: 'Ground- or water-based gauge measurements (NOAA tide gauges, USGS streamflow, in-situ sensors)',
  },
  reanalysis_model: {
    label: 'NASA Model / Reanalysis',
    shortLabel: 'REANALYSIS',
    icon: '🔄',
    colorClass: 'text-gold-400',
    description: 'NASA POWER gridded reanalysis data (MERRA-2, GLDAS, CERES assimilations)',
  },
  user_upload: {
    label: 'User Upload',
    shortLabel: 'USER CSV',
    icon: '📤',
    colorClass: 'text-gold-300',
    description: 'Custom CSV or JSON data uploaded by the user',
  },
  synthetic_demo: {
    label: 'Calibrated Demonstration',
    shortLabel: 'DEMO',
    icon: '🧪',
    colorClass: 'text-silver-600',
    description: 'Synthetic time series calibrated against published observations for offline demonstration',
  },
};

export interface EarthVariable {
  id: VariableId;
  name: string;
  shortName: string;
  unit: string;
  category: VariableCategory;
  description: string;
  studentExplanation: string;
  missions: string[]; // e.g. ["MODIS (Terra/Aqua)", "MERRA-2"]
  measurementType: MeasurementType;
  typicalRange: [number, number];
  color: string;
  iconName: string;
  driverExamples: string[];
  fillValue?: number; // e.g. -999 for NASA POWER polar-night
}

export interface HotspotRegion {
  id: string;
  name: string;
  countryOrOcean: string;
  coordinates: [number, number]; // [lat, lon]
  hasCoordinates?: boolean; // False when uploaded CSV lacks lat/lon columns
  category: 'Polar' | 'Tropical Rainforest' | 'Arid / Desert' | 'Agricultural' | 'Urban' | 'Marine' | 'High Altitude' | 'Coastal' | 'Riverine';
  description: string;
  keyPhenomenon: string;
  primaryVariable: VariableId;
  oppositePairId?: string; // Links to another region with contrasting trend
  oppositeExplanation?: string;
}

export interface TimeSeriesPoint {
  date: string; // YYYY-MM
  year: number;
  month: number;
  raw: number | null;
  isMissing?: boolean;
  movingAvg?: number; // 12-month rolling mean
  anomaly?: number;   // deseasonalized baseline anomaly
  trend?: number;     // linear regression value
  ciUpper?: number;   // 95% confidence upper bound
  ciLower?: number;   // 95% confidence lower bound
}

export interface TrendAnalysisResult {
  variableId: VariableId;
  variableName: string;
  unit: string;
  regionName: string;
  coordinates: [number, number];
  startDate: string;
  endDate: string;
  sampleCount: number;
  
  // Rate of change
  slopePerYear: number;
  slopePerDecade: number;
  sensSlopePerDecade: number;
  totalChange: number;
  percentChange: number;
  direction: 'Increasing' | 'Decreasing' | 'Stable';
  
  // Regression metrics
  rSquared: number;
  olsSlope: number;
  olsIntercept: number;
  standardError: number;
  
  // Student's t-test
  tStatistic: number;
  tPValue: number;
  
  // Mann-Kendall Trend Test
  mkS: number;
  mkVarS: number;
  mkZ: number;
  mkTau: number;
  mkPValue: number;
  
  // Significance Level
  significanceLevel: 'Extremely Significant' | 'Highly Significant' | 'Significant' | 'Marginally Significant' | 'Not Significant';
  significanceStars: '***' | '**' | '*' | '†' | 'n.s.';
  significanceColor: string;
  
  // Descriptive summary
  plainEnglishVerdict: string;
  scientificImplication: string;

  // Provenance & Missingness
  provenance?: ProvenanceTier;
  provenanceSource?: string;
  statusMessage?: string;
  missingCount?: number;

  // Time range metadata from the data source
  /** Requested date range, e.g. "1995-2026" */
  requestedRange?: string;
  /** Actual date range returned by the API, e.g. "1995-2025" */
  actualRange?: string;
  /** Most recent month with a valid observation, e.g. "2025-11" */
  latestValidObservation?: string;
}

// ── Safe trend analysis result (no-throw wrapper) ────────────────────

export type TrendAnalysisState = 'loading' | 'ready' | 'unavailable' | 'empty' | 'error';

export interface TrendAnalysisResponse {
  state: TrendAnalysisState;
  /** Populated only when state === 'ready' */
  result?: TrendAnalysisResult;
  /** Validation diagnostics describing why the data was rejected */
  validation?: {
    totalPoints: number;
    validCount: number;
    missingCount: number;
    minRequired: number;
    messages: string[];
    state: 'loading' | 'ready' | 'unavailable' | 'error' | 'empty';
  };
  /** Human-readable explanation for non-ready states */
  explanation?: string;
}

// ── Connector Registry Types ──────────────────────────────────────────

export interface ConnectorCapabilities {
  timeSeries: boolean;       // Can return TimeSeriesPoint[]
  mapTiles: boolean;         // Can provide WMTS/tile overlays
  catalogSearch: boolean;    // Can search metadata/collections
  pointSubset: boolean;      // Can query by lat/lon point
  regionSubset: boolean;     // Can query by bounding box
  /** Geographic coverage of the connector's data source. */
  coverage: GeographicCoverage;
}

/** Kind of geographic coverage a connector claims. */
export type CoverageKind = 'global' | 'regional';

/**
 * Declarative geographic coverage metadata for a connector.
 * Only claim coverage that is supported by the provider's actual documentation
 * or implementation; never claim global coverage for a regional in-situ network.
 */
export interface GeographicCoverage {
  /** 'global' when the provider serves data worldwide (subject to product masks); 'regional' when limited to a specific area. */
  kind: CoverageKind;
  /** Human-readable description of the covered area and any known limitations. */
  description: string;
  /** Named regions/countries covered (only meaningful when kind === 'regional'). */
  regions?: string[];
  /**
   * For in-situ regional connectors: maximum great-circle distance (km) from a
   * station/gauge at which a query point is still considered covered.
   * Points beyond this distance return an explicit unavailable result.
   */
  maxStationDistanceKm?: number;
}

export interface ConnectorFetchOptions {
  lat: number;
  lon: number;
  variableId: VariableId;
  startYear?: number;
  endYear?: number;
  parameter?: string; // connector-specific parameter override
}

/** Scientific metadata describing the physical station behind a measurement */
export interface StationMetadata {
  /** Station / gauge identifier, e.g. "8518750" */
  stationId: string;
  /** Human-readable station name */
  stationName: string;
  /** Station coordinates [lat, lon] */
  stationCoordinates: [number, number];
  /** Great-circle distance from the query point in km */
  distanceKm: number;
  /** Measurement type, e.g. "Tide gauge", "Streamflow gauge" */
  measurementType: string;
  /** Observation period covered, e.g. "2000-2024" */
  observationPeriod: string;
  /** Units of the returned values, e.g. "m", "m³/s" */
  units: string;
  /** Source agency, e.g. "NOAA CO-OPS", "USGS NWIS" */
  sourceAgency: string;
}

export interface ConnectorFetchResult {
  points: TimeSeriesPoint[];
  provenance: ProvenanceTier;
  sourceLabel: string;
  statusMessage: string;
  /** Scientific metadata about the measurement station, when applicable */
  station?: StationMetadata;
  /** Marks a result where no geographically appropriate station was found */
  unavailable?: boolean;
  /** Requested date range, e.g. "1995-2026" */
  requestedRange?: string;
  /** Actual date range returned by the API, e.g. "1995-2025" */
  actualRange?: string;
  /** Most recent month with a valid observation, e.g. "2025-11" */
  latestValidObservation?: string;
}

export interface EarthDataConnector {
  id: string;
  name: string;
  shortName: string;
  description: string;
  baseUrl: string;
  capabilities: ConnectorCapabilities;
  supportedVariables: VariableId[];
  provenanceType: ProvenanceTier;
  requiresAuth: boolean;
  requiresProxy: boolean;

  /** Fetch time-series data for a point */
  fetchTimeSeries(options: ConnectorFetchOptions): Promise<ConnectorFetchResult>;

  /** Check if a variable is supported by this connector */
  supportsVariable(variableId: VariableId): boolean;

  /**
   * Optional point-coverage test. Regional connectors should implement this
   * (e.g. nearest-station distance check). Global connectors omit it and are
   * treated as covering every valid coordinate. When omitted for a regional
   * connector, coverage is conservatively treated as false everywhere.
   */
  coversPoint?(lat: number, lon: number): boolean;
}

// ── Statistical Lab Types ─────────────────────────────────────────────

export interface DescriptiveStats {
  count: number;
  mean: number;
  median: number;
  mode: number[];
  stdDev: number;
  variance: number;
  skewness: number;
  kurtosis: number;
  min: number;
  max: number;
  range: number;
  q1: number;
  q3: number;
  iqr: number;
  cv: number; // coefficient of variation
  sem: number; // standard error of mean
  missingCount: number;
  missingPercent: number;
}

export interface CorrelationResult {
  pearsonR: number;
  pearsonP: number;
  spearmanRho: number;
  spearmanP: number;
  kendallTau: number;
  kendallP: number;
}

export interface RegressionResult {
  coefficients: number[];
  intercept: number;
  rSquared: number;
  adjustedRSquared: number;
  fStatistic: number;
  fPValue: number;
  standardErrors: number[];
  tStatistics: number[];
  tPValues: number[];
  residuals: number[];
  predictedValues: number[];
  durbinWatson: number;
}

export interface PCAResult {
  eigenvalues: number[];
  explainedVariance: number[];
  cumulativeVariance: number[];
  loadings: number[][];
  scores: number[][];
  componentLabels: string[];
}

export interface ChangePointResult {
  method: 'Pettitt' | 'CUSUM';
  changePointIndex: number;
  changePointDate: string;
  testStatistic: number;
  pValue: number;
  significanceLevel: string;
  meanBefore: number;
  meanAfter: number;
  shift: number;
}

export interface DiagnosticResults {
  normalityTest: {
    method: string;
    statistic: number;
    pValue: number;
    isNormal: boolean;
    /**
     * Interpretation of the normality test outcome.
     * - 'no_evidence_against': p > alpha; failed to reject normality hypothesis.
     *   This is NOT proof that the data are normal; it only indicates insufficient
     *   evidence to reject the assumption.
     * - 'evidence_against': p <= alpha; rejected normality.
     * - 'insufficient_data': test could not be evaluated.
     */
    verdict: 'no_evidence_against' | 'evidence_against' | 'insufficient_data';
  };
  outliers: { index: number; value: number; method: string; zScore?: number }[];
  missingnessProfile: { total: number; missing: number; percent: number; pattern: string };
  /**
   * Autocorrelation diagnostics.
   * All fields are null when the underlying observation ordering is unavailable
   * (e.g. residuals constructed group-by-group in arbitrary order). A null value
   * indicates the diagnostic is not computable, NOT that the value is zero.
   */
  autocorrelation: {
    lag1: number | null;
    durbinWatson: number | null;
    ljungBoxQ: number | null;
    ljungBoxP: number | null;
    observationOrderAvailable?: boolean;
    warning?: string;
  };
  heteroskedasticity?: { bpStatistic: number; bpPValue: number; isHomoskedastic: boolean };
  vif?: { variable: string; vif: number }[];
}

export interface SeasonalMKResult {
  s: number;
  varS: number;
  z: number;
  pValue: number;
  tau: number;
  method: 'standard' | 'hamed_rao' | 'tfpw';
  seasonCount: number;
  /**
   * Variance correction factor applied to the S statistic.
   * For Hamed-Rao this is the variance inflation factor (n/N*), not a correlation.
   * For TFPW this is undefined (the correction is via pre-whitening, not a variance multiplier).
   * Undefined when no correction was applied.
   */
  autocorrelationCorrection?: number;
  /**
   * TFPW lag-1 autocorrelation diagnostic (mean across seasons with sufficient consecutive pairs).
   * This is a correlation coefficient, NOT a variance multiplier. Only present for TFPW.
   */
  tfpwR1?: number;
  /**
   * Per-lag rho values used in Hamed-Rao variance correction.
   * These are correlation coefficients, NOT variance multipliers. Only present for Hamed-Rao.
   */
  hamedRaoRhoDiagnostic?: number[];
  /**
   * Optional warnings about data quality or methodological assumptions.
   * E.g. irregular seasonal spacing, insufficient consecutive pairs.
   */
  warnings?: string[];
}

export interface DetectiveCase {
  id: string;
  title: string;
  mysteryTitle: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  regionId: string;
  variableId: VariableId;
  teaser: string;
  briefing: string;
  suspectDrivers: string[];
  satelliteClue: string;
  guidingQuestions: {
    question: string;
    options: string[];
    correctIndex: number;
    explanation: string;
  }[];
  verdictDebrief: string;
  realWorldImpact: string;
}

export interface RegionalContrastScenario {
  id: string;
  title: string;
  commonDriver: string;
  description: string;
  variableId: VariableId;
  regionA: {
    hotspotId: string;
    trendDescription: string;
    mechanism: string;
  };
  regionB: {
    hotspotId: string;
    trendDescription: string;
    mechanism: string;
  };
  scientificTakeaway: string;
}

export interface TwoWayAnovaEffect {
  f: number | null;
  df: number;
  ss: number;
  ms: number;
  pValue: number | null;
  estimable: boolean;
  note?: string;
}

export interface TwoWayAnovaResult {
  factorALabel: string;
  factorBLabel: string;
  type: 'I' | 'II' | 'III';
  factorA: TwoWayAnovaEffect;
  factorB: TwoWayAnovaEffect;
  interaction: TwoWayAnovaEffect;
  within: { df: number; ss: number; ms: number };
  grandMean: number;
  cellMeans: { levelA: string; levelB: string; mean: number; n: number }[];
  rank: number;
  dfError: number;
  dfModel: number;
  ssTotal: number;
  error: string | null;
  warnings: string[];
}

export interface NonlinearRegressionResult {
  model: 'exponential' | 'logarithmic' | 'power' | 'quadratic' | 'logistic';
  equation: string;
  paramLabels: string[];
  params: number[];
  rSquared: number;
  rmse: number;
  aic: number;
  predictedValues: number[];
  residuals: number[];
}

export interface GEVResult {
  shapeXi: number;
  locationMu: number;
  scaleSigma: number;
  family: 'Frechet (heavy-tailed)' | 'Gumbel (light-tailed)' | 'Weibull (bounded)';
  blockMaxima: number[];
  returnLevels: { period: number; level: number }[];
}
