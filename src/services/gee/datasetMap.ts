/**
 * HeraOculus → Google Earth Engine dataset map.
 *
 * Maps each supported VariableId to one or more candidate GEE ImageCollections
 * with the canonical band, scale factor, offset, units and native resolution,
 * taken from the official Earth Engine Data Catalog. Every entry below has been
 * cross-checked against its catalog page; deprecated datasets are flagged.
 *
 * Variables with NO good GEE equivalent map to an empty array and are excluded
 * from the UI (we never generate a broken script).
 */

import type { VariableId } from '../../types/earthScience';

export type GeeReducer = 'mean' | 'median' | 'first' | 'max' | 'sum';
export type GeeTemporalAgg = 'none' | 'monthly' | 'annual';
export type GeeExportTarget = 'chart' | 'csv-drive' | 'geotiff-drive';

export type GeeGeometry =
  | { kind: 'point'; lat: number; lon: number }
  | { kind: 'bbox'; west: number; south: number; east: number; north: number }
  | { kind: 'polygon'; rings: number[][][] };

export interface GeeDatasetConfig {
  variableId: VariableId;
  /** Human-readable label shown in the dataset selector. */
  label: string;
  /** Earth Engine ImageCollection asset ID. */
  eeCollectionId: string;
  /** Primary band name (exact, from the catalog). */
  band: string;
  /** Multiplier applied to raw stored values. */
  scale: number;
  /** Additive offset applied after scaling. */
  offset: number;
  /** Output units after scale/offset. */
  units: string;
  /** Native pixel resolution in metres. */
  nativeScaleMeters: number;
  /** Default spatial reducer for reduceRegion. */
  reducer: GeeReducer;
  /** Optional GEE expression (computed index, e.g. S2 NDVI from B8/B4). */
  expression?: string;
  /** Band name → expression variable name when using an expression. */
  expressionBands?: Record<string, string>;
  /** Apply a basic cloud-mask step (relevant for optical/S2). */
  cloudMask?: boolean;
  /** Values below this (raw, pre-scale) are treated as invalid/no-data. */
  validMin?: number;
  /** Marked true when the catalog version is deprecated/superseded. */
  deprecated?: boolean;
  /** Human notes / caveats for the generated script header. */
  notes: string;
  /** Coverage caveat shown in the UI. */
  coverageNote?: string;
}

export interface GeeScriptOptions {
  geometry: GeeGeometry;
  startDate: string; // YYYY-MM-DD
  endDate: string;   // YYYY-MM-DD
  variableId: VariableId;
  datasetIndex?: number; // which candidate dataset (default 0 = best)
  temporalAggregation: GeeTemporalAgg;
  spatialReducer: GeeReducer;
  exportTarget: GeeExportTarget;
}

/**
 * Resolve the dataset config for a variable. Returns undefined when the
 * variable has no GEE equivalent (caller must exclude it from the UI).
 */
export function resolveGeeDataset(
  variableId: VariableId,
  datasetIndex = 0
): GeeDatasetConfig | undefined {
  const cands = GEE_DATASET_MAP[variableId];
  if (!cands || cands.length === 0) return undefined;
  const idx = Math.max(0, Math.min(datasetIndex, cands.length - 1));
  return cands[idx];
}

export function listGeeDatasets(variableId: VariableId): GeeDatasetConfig[] {
  return GEE_DATASET_MAP[variableId] ?? [];
}

export function hasGeeSupport(variableId: VariableId): boolean {
  return (GEE_DATASET_MAP[variableId] ?? []).length > 0;
}

/* ------------------------------------------------------------------ */
/* Dataset map                                                        */
/* ------------------------------------------------------------------ */

export const GEE_DATASET_MAP: Record<VariableId, GeeDatasetConfig[]> = {
  /* ---------------------------------------------------------------- */
  /* NDVI — MODIS MOD13Q1 (primary) + Sentinel-2 SR (computed)       */
  /* ---------------------------------------------------------------- */
  ndvi: [
    {
      variableId: 'ndvi',
      label: 'MODIS Terra Vegetation Indices (MOD13Q1 V6.1) — NDVI',
      eeCollectionId: 'MODIS/061/MOD13Q1',
      band: 'NDVI',
      scale: 0.0001,
      offset: 0,
      units: 'index (0–1)',
      nativeScaleMeters: 250,
      reducer: 'mean',
      notes:
        '16-day composite, 250 m. Apply scale factor 0.0001. Valid raw range -2000..10000; ' +
        'values outside that range are fill/no-data. Use SummaryQA/DetailedQA for quality ' +
        'filtering if needed. V0.06 (006) was decommissioned 2023-07-31; this map uses V0.61 (061).',
    },
    {
      variableId: 'ndvi',
      label: 'Sentinel-2 MSI Level-2A SR (Harmonized) — NDVI (B8−B4)/(B8+B4)',
      eeCollectionId: 'COPERNICUS/S2_SR_HARMONIZED',
      band: 'B8',
      scale: 0.0001,
      offset: 0,
      units: 'index (0–1)',
      nativeScaleMeters: 10,
      reducer: 'mean',
      expression: '(b8 - b4) / (b8 + b4)',
      expressionBands: { b8: 'B8', b4: 'B4' },
      cloudMask: true,
      notes:
        'NDVI computed from NIR (B8) and red (B4) surface reflectance. The 0.0001 scale ' +
        'cancels in the ratio, so bands are used unscaled. Basic QA60 cloud mask applied ' +
        '(approximate; harmonized collection reconstructs QA60 from 2024-02-28). ' +
        '10 m resolution, 5-day revisit. L2 coverage 2017-2018 is not yet global.',
    },
  ],
  /* ---------------------------------------------------------------- */
  /* EVI — MODIS MOD13Q1                                              */
  /* ---------------------------------------------------------------- */
  evi: [
    {
      variableId: 'evi',
      label: 'MODIS Terra Vegetation Indices (MOD13Q1 V6.1) — EVI',
      eeCollectionId: 'MODIS/061/MOD13Q1',
      band: 'EVI',
      scale: 0.0001,
      offset: 0,
      units: 'index (0–1)',
      nativeScaleMeters: 250,
      reducer: 'mean',
      notes:
        '16-day composite, 250 m. Apply scale factor 0.0001. Valid raw range -2000..10000; ' +
        'values outside that range are fill/no-data. V0.61 (061); V0.06 decommissioned 2023-07-31.',
    },
  ],
  /* ---------------------------------------------------------------- */
  /* Precipitation — CHIRPS (primary) + GPM IMERG                     */
  /* ---------------------------------------------------------------- */
  precip: [
    {
      variableId: 'precip',
      label: 'CHIRPS Daily Precipitation (V2 Final)',
      eeCollectionId: 'UCSB-CHG/CHIRPS/DAILY',
      band: 'precipitation',
      scale: 1,
      offset: 0,
      units: 'mm/day',
      nativeScaleMeters: 5566,
      reducer: 'mean',
      notes:
        'Daily precipitation, quasi-global (60°S–60°N), ~0.05° (~5.6 km). No scale factor ' +
        '(raw values already in mm/day). For monthly totals, sum daily values; for annual, ' +
        'sum then divide by years as needed.',
    },
    {
      variableId: 'precip',
      label: 'GPM IMERG Final V07 — Precipitation',
      eeCollectionId: 'NASA/GPM_L3/IMERG_V07',
      band: 'precipitation',
      scale: 1,
      offset: 0,
      units: 'mm/hr',
      nativeScaleMeters: 11132,
      reducer: 'mean',
      notes:
        'Half-hourly, global 0.1°. Units mm/hr. Final-run products transitioned to V08 ' +
        'after 2025-09-30; check provisional/permanent status for your date range.',
      coverageNote: 'Global (90°S–90°N).',
    },
  ],
  /* ---------------------------------------------------------------- */
  /* Land Surface Temperature — MODIS MOD11A1                         */
  /* ---------------------------------------------------------------- */
  lst_day: [
    {
      variableId: 'lst_day',
      label: 'MODIS Terra LST & Emissivity (MOD11A1 V6.1) — Daytime LST',
      eeCollectionId: 'MODIS/061/MOD11A1',
      band: 'LST_Day_1km',
      scale: 0.02,
      offset: 0,
      units: 'K',
      nativeScaleMeters: 1000,
      reducer: 'mean',
      validMin: 7500,
      notes:
        'Daily, 1 km. Apply scale 0.02 → Kelvin. Raw fill value is 0; valid range 7500..65535. ' +
        'Mask raw < 7500 as no-data. °C = K − 273.15. Clear-sky only; above 30° latitude ' +
        'multiple observations may be averaged. V0.61 (061); V0.06 decommissioned 2023-07-31.',
    },
  ],
  lst_night: [
    {
      variableId: 'lst_night',
      label: 'MODIS Terra LST & Emissivity (MOD11A1 V6.1) — Nighttime LST',
      eeCollectionId: 'MODIS/061/MOD11A1',
      band: 'LST_Night_1km',
      scale: 0.02,
      offset: 0,
      units: 'K',
      nativeScaleMeters: 1000,
      reducer: 'mean',
      validMin: 7500,
      notes:
        'Daily, 1 km. Apply scale 0.02 → Kelvin. Raw fill value is 0; valid range 7500..65535. ' +
        'Mask raw < 7500 as no-data. °C = K − 273.15. Clear-sky only; above 30° latitude ' +
        'multiple observations may be averaged. V0.61 (061); V0.06 decommissioned 2023-07-31.',
    },
  ],
  /* ---------------------------------------------------------------- */
  /* Air temperature — ERA5-Land                                      */
  /* ---------------------------------------------------------------- */
  temp_air: [
    {
      variableId: 'temp_air',
      label: 'ERA5-Land Daily Aggregated — 2 m Temperature',
      eeCollectionId: 'ECMWF/ERA5_LAND/DAILY_AGGR',
      band: 'temperature_2m',
      scale: 1,
      offset: 0,
      units: 'K',
      nativeScaleMeters: 11132,
      reducer: 'mean',
      notes:
        'Daily aggregate (1950–present, ~3-month latency). Already in Kelvin, no scale factor. ' +
        '°C = K − 273.15. Land-only grid.',
      coverageNote: 'Global land (90°S–90°N).',
    },
  ],
  /* ---------------------------------------------------------------- */
  /* No good GEE equivalent — excluded from UI                        */
  /* ---------------------------------------------------------------- */
  soil_moist: [],
  soil_moist_top: [],
  soil_moist_root: [],
  soil_moist_prof: [],
  evapo: [],
  aod: [],
  ozone: [],
  ozone_total: [],
  pressure: [],
  humidity: [],
  specific_humidity: [],
  wind_speed: [],
  radiation_sw: [],
  radiation_lw: [],
  albedo: [],
  snow_depth: [],
  snow_precip: [],
  sea_level: [],
  sst: [],
  streamflow: [],
};
