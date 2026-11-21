/**
 * MODIS LST Night connector — gridded monthly land surface temperature (NASA
 * LAADS / Earthdata Cloud).
 *
 * Proxyed through the Geo-Iris backend (`POST /api/v1/gridded/modis-lst-night`).
 * The backend requires NASA Earthdata Login credentials
 * (EARTHDATA_USERNAME/EARTHDATA_PASSWORD or EARTHDATA_TOKEN); when they are not
 * configured the backend returns an explicit 'unavailable' result — the
 * connector never fabricates data.
 * Coverage: global (90°S–90°N), 1 km sinusoidal grid, 2000–present.
 * Variable: night-time land surface temperature (Kelvin). Clear-sky only;
 * LST_Night_1km is stored as scaled integers (K = raw * 0.02).
 */

import type {
  ConnectorCapabilities,
  ConnectorFetchOptions,
  ConnectorFetchResult,
  EarthDataConnector,
  TimeSeriesPoint,
  VariableId,
} from './types';
import { globalCoverage } from './types';

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

const modisLstCapabilities: ConnectorCapabilities = {
  timeSeries: true,
  mapTiles: false,
  catalogSearch: false,
  pointSubset: true,
  regionSubset: false,
  coverage: globalCoverage(
    'MODIS/Terra MOD11A1 V061 night-time land surface temperature, global (90°S–90°N), 1 km sinusoidal grid, 2000–present. Satellite-derived, clear-sky only.'
  ),
};

export const modisLstConnector: EarthDataConnector = {
  id: 'modis-lst-night',
  name: 'MODIS LST Night',
  shortName: 'MODIS LST Night',
  description:
    'NASA MODIS/Terra MOD11A1 V061 — 1 km gridded monthly night-time land surface temperature, global, 2000–present. Stored as scaled integers; backend converts to Kelvin.',
  baseUrl: 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD11A1/',
  capabilities: modisLstCapabilities,
  supportedVariables: ['lst_night'],
  provenanceType: 'satellite_observation',
  requiresAuth: false,
  requiresProxy: true,

  supportsVariable(variableId: VariableId): boolean {
    return variableId === 'lst_night';
  },

  coversPoint(lat: number, lon: number): boolean {
    return lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
  },

  async fetchTimeSeries(options: ConnectorFetchOptions): Promise<ConnectorFetchResult> {
    const { lat, lon, variableId } = options;
    const startYear = options.startYear ?? 2000;
    const endYear = options.endYear ?? new Date().getFullYear();

    if (!this.supportsVariable(variableId)) {
      return {
        points: [],
        provenance: 'satellite_observation',
        sourceLabel: 'MODIS LST Night',
        statusMessage: `Variable "${variableId}" is not supported by MODIS LST Night (only 'lst_night').`,
      };
    }
    if (!(this.coversPoint?.(lat, lon) ?? false)) {
      return {
        points: [],
        unavailable: true,
        provenance: 'satellite_observation',
        sourceLabel: 'MODIS LST Night',
        statusMessage: `MODIS LST Night does not cover (${lat.toFixed(2)}, ${lon.toFixed(2)}) — outside 90°S–90°N.`,
      };
    }

    try {
      const response = await fetch(`${GRIDDED_API_BASE}/gridded/modis-lst-night`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          variable: 'precip',
          start_date: `${startYear}-01`,
          end_date: `${endYear}-12`,
          lat,
          lon,
        }),
      });

      if (!response.ok) {
        const detail = await response.text().catch(() => response.statusText);
        return {
          points: [],
          provenance: 'satellite_observation',
          sourceLabel: 'MODIS LST Night',
          statusMessage: `MODIS LST Night request failed (HTTP ${response.status}): ${detail.slice(0, 300)}`,
        };
      }

      const json: GriddedSeriesResponse = await response.json();
      if (json.status === 'unavailable' || !json.series) {
        return {
          points: [],
          unavailable: true,
          provenance: 'satellite_observation',
          sourceLabel: 'MODIS LST Night',
          statusMessage: json.message ?? 'MODIS LST Night data unavailable for this location/range.',
        };
      }

      const points = toTimeSeriesPoints(json.series);
      const validCount = points.filter(p => p.raw !== null).length;
      return {
        points,
        provenance: 'satellite_observation',
        sourceLabel: `MODIS LST Night (${json.series.dataset})`,
        statusMessage:
          json.message ??
          `Retrieved ${validCount} valid month(s) of MODIS LST Night (${json.series.missing_count} missing), ${json.series.units}.`,
        requestedRange: `${startYear}-${endYear}`,
        actualRange: `${startYear}-${endYear}`,
        latestValidObservation:
          points
            .slice()
            .reverse()
            .find(p => p.raw !== null)?.date ?? undefined,
      };
    } catch (err) {
      return {
        points: [],
        provenance: 'satellite_observation',
        sourceLabel: 'MODIS LST Night',
        statusMessage: `MODIS LST Night request failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  },
};

export default modisLstConnector;
