/**
 * CHIRPS v3 connector — gridded monthly rainfall (Climate Hazards Center).
 *
 * Keyless, proxyed through the HeraOculus backend (`POST /api/v1/gridded/chirps`).
 * Coverage: 60°S–60°N (land), all longitudes, 0.05° grid, 1981–present.
 * Variable: precipitation (mm/month). Blends satellite estimates with in-situ
 * gauge stations — documented as satellite_observation with a gauge-blend note.
 * Synthetic data never masquerades as live: an out-of-coverage point or a
 * backend error yields an explicit 'unavailable' result.
 */

import type {
  ConnectorCapabilities,
  ConnectorFetchOptions,
  ConnectorFetchResult,
  EarthDataConnector,
  TimeSeriesPoint,
  VariableId,
} from './types';
import { regionalCoverage } from './types';

const GRIDDED_API_BASE =
  import.meta.env.VITE_API_BASE || '/api/v1';

export interface GriddedSeriesResponse {
  status: 'available' | 'unavailable';
  message: string;
  series: {
    dates: string[];
    values: (number | null)[];
    units: string;
    dataset: string;
    version: string;
    spatial_resolution: string;
    temporal_resolution: string;
    provenance: string;
    source_url: string;
    coverage_note: string;
    valid_count: number;
    missing_count: number;
  } | null;
}

function toTimeSeriesPoints(
  series: NonNullable<GriddedSeriesResponse['series']>
): TimeSeriesPoint[] {
  return series.dates.map((date, i) => {
    const [yearStr, monthStr] = date.split('-');
    const value = series.values?.[i];
    return {
      date,
      year: parseInt(yearStr, 10),
      month: parseInt(monthStr, 10),
      raw: value === null ? null : value,
      isMissing: value === null,
    };
  });
}

const chirpsCapabilities: ConnectorCapabilities = {
  timeSeries: true,
  mapTiles: false,
  catalogSearch: false,
  pointSubset: true,
  regionSubset: false,
  coverage: regionalCoverage(
    'CHIRPS v3 quasi-global land rainfall: 60°S–60°N, all longitudes, 0.05° grid, 1981–present. Monthly totals in mm; gauge-blended (not pure satellite).',
    { regions: ['60S-60N land band'] }
  ),
};

export const chirpsConnector: EarthDataConnector = {
  id: 'chirps',
  name: 'CHIRPS v3 Monthly Rainfall',
  shortName: 'CHIRPS',
  description:
    'Climate Hazards Center Infrared Precipitation with Stations (CHIRPS) v3 — 0.05° gridded monthly rainfall, 60°S–60°N, 1981–present. Satellite + gauge blended.',
  baseUrl: 'https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/',
  capabilities: chirpsCapabilities,
  supportedVariables: ['precip'],
  provenanceType: 'satellite_observation',
  requiresAuth: false,
  requiresProxy: true,

  supportsVariable(variableId: VariableId): boolean {
    return variableId === 'precip';
  },

  coversPoint(lat: number, lon: number): boolean {
    return lat >= -60 && lat <= 60 && lon >= -180 && lon <= 180;
  },

  async fetchTimeSeries(options: ConnectorFetchOptions): Promise<ConnectorFetchResult> {
    const { lat, lon, variableId } = options;
    const startYear = options.startYear ?? 1981;
    const endYear = options.endYear ?? new Date().getFullYear();

    if (!this.supportsVariable(variableId)) {
      return {
        points: [],
        provenance: 'satellite_observation',
        sourceLabel: 'CHIRPS v3',
        statusMessage: `Variable "${variableId}" is not supported by CHIRPS (only 'precip').`,
      };
    }
    if (!(this.coversPoint?.(lat, lon) ?? false)) {
      return {
        points: [],
        unavailable: true,
        provenance: 'satellite_observation',
        sourceLabel: 'CHIRPS v3',
        statusMessage: `CHIRPS does not cover (${lat.toFixed(2)}, ${lon.toFixed(2)}) — outside 60°S–60°N.`,
      };
    }

    try {
      const startDate = `${startYear}-01`;
      const endDate = `${endYear}-12`;
      const response = await fetch(`${GRIDDED_API_BASE}/gridded/chirps`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          variable: 'precip',
          start_date: startDate,
          end_date: endDate,
          lat,
          lon,
        }),
      });

      if (!response.ok) {
        const detail = await response.text().catch(() => response.statusText);
        return {
          points: [],
          provenance: 'satellite_observation',
          sourceLabel: 'CHIRPS v3',
          statusMessage: `CHIRPS request failed (HTTP ${response.status}): ${detail.slice(0, 300)}`,
        };
      }

      const json: GriddedSeriesResponse = await response.json();
      if (json.status === 'unavailable' || !json.series) {
        return {
          points: [],
          unavailable: true,
          provenance: 'satellite_observation',
          sourceLabel: 'CHIRPS v3',
          statusMessage: json.message ?? 'CHIRPS data unavailable for this location/range.',
        };
      }

      const points = toTimeSeriesPoints(json.series);
      const validCount = points.filter(p => p.raw !== null).length;
      const latest = points.slice().reverse().find(p => p.raw !== null);
      return {
        points,
        provenance: 'satellite_observation',
        sourceLabel: `CHIRPS v3 (${json.series.dataset})`,
        statusMessage:
          json.message ??
          `Retrieved ${validCount} valid month(s) of CHIRPS v3 (${json.series.missing_count} missing), ${json.series.units}.`,
        requestedRange: `${startYear}-${endYear}`,
        actualRange: `${startYear}-${endYear}`,
        latestValidObservation: latest?.date ?? undefined,
      };
    } catch (err) {
      return {
        points: [],
        provenance: 'satellite_observation',
        sourceLabel: 'CHIRPS v3',
        statusMessage: `CHIRPS request failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  },
};

export default chirpsConnector;
