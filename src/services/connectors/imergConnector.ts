/**
 * GPM IMERG Final connector — gridded monthly rainfall (NASA GES DISC).
 *
 * Proxyed through the HeraOculus backend (`POST /api/v1/gridded/imerg`).
 * The backend requires NASA Earthdata Login credentials
 * (EARTHDATA_USERNAME/EARTHDATA_PASSWORD or EARTHDATA_TOKEN); when they are not
 * configured the backend returns an explicit 'unavailable' result — the
 * connector never fabricates data.
 * Coverage: global (90°S–90°N), 0.1°×0.1°, 1998–present.
 * Variable: precipitation (mm/hr stored; converted to mm/month by the backend).
 * Final product has ~3.5–4 month latency; V07 Final ran through 2025-09.
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

const imergCapabilities: ConnectorCapabilities = {
  timeSeries: true,
  mapTiles: false,
  catalogSearch: false,
  pointSubset: true,
  regionSubset: false,
  coverage: globalCoverage(
    'GPM IMERG Final V07B: global (90°S–90°N), 0.1°×0.1°, 1998–present. Satellite–gauge merged; Final run ~3.5–4 month latency.'
  ),
};

export const imergConnector: EarthDataConnector = {
  id: 'imerg',
  name: 'GPM IMERG Final Monthly Rainfall',
  shortName: 'IMERG',
  description:
    'NASA GPM IMERG Final Run (GPM_3IMERGM V07B) — 0.1°×0.1° gridded monthly satellite–gauge precipitation, global, 1998–present. Backend converts mm/hr to mm/month.',
  baseUrl: 'https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/',
  capabilities: imergCapabilities,
  supportedVariables: ['precip'],
  provenanceType: 'satellite_observation',
  requiresAuth: false,
  requiresProxy: true,

  supportsVariable(variableId: VariableId): boolean {
    return variableId === 'precip';
  },

  coversPoint(lat: number, lon: number): boolean {
    return lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
  },

  async fetchTimeSeries(options: ConnectorFetchOptions): Promise<ConnectorFetchResult> {
    const { lat, lon, variableId } = options;
    const startYear = options.startYear ?? 1998;
    const endYear = options.endYear ?? new Date().getFullYear();

    if (!this.supportsVariable(variableId)) {
      return {
        points: [],
        provenance: 'satellite_observation',
        sourceLabel: 'GPM IMERG Final',
        statusMessage: `Variable "${variableId}" is not supported by IMERG (only 'precip').`,
      };
    }
    if (!(this.coversPoint?.(lat, lon) ?? false)) {
      return {
        points: [],
        unavailable: true,
        provenance: 'satellite_observation',
        sourceLabel: 'GPM IMERG Final',
        statusMessage: `IMERG does not cover (${lat.toFixed(2)}, ${lon.toFixed(2)}) — outside 90°S–90°N.`,
      };
    }

    try {
      const response = await fetch(`${GRIDDED_API_BASE}/gridded/imerg`, {
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
          sourceLabel: 'GPM IMERG Final',
          statusMessage: `IMERG request failed (HTTP ${response.status}): ${detail.slice(0, 300)}`,
        };
      }

      const json: GriddedSeriesResponse = await response.json();
      if (json.status === 'unavailable' || !json.series) {
        return {
          points: [],
          unavailable: true,
          provenance: 'satellite_observation',
          sourceLabel: 'GPM IMERG Final',
          statusMessage: json.message ?? 'IMERG data unavailable for this location/range.',
        };
      }

      const points = toTimeSeriesPoints(json.series);
      const validCount = points.filter(p => p.raw !== null).length;
      const latest = points.slice().reverse().find(p => p.raw !== null);
      return {
        points,
        provenance: 'satellite_observation',
        sourceLabel: `GPM IMERG Final (${json.series.dataset})`,
        statusMessage:
          json.message ??
          `Retrieved ${validCount} valid month(s) of IMERG Final (${json.series.missing_count} missing), ${json.series.units}.`,
        requestedRange: `${startYear}-${endYear}`,
        actualRange: `${startYear}-${endYear}`,
        latestValidObservation: latest?.date ?? undefined,
      };
    } catch (err) {
      return {
        points: [],
        provenance: 'satellite_observation',
        sourceLabel: 'GPM IMERG Final',
        statusMessage: `IMERG request failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  },
};

export default imergConnector;
