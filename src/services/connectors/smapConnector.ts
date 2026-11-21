/**
 * SMAP L4 connector — gridded monthly soil moisture (NASA NSIDC / Earthdata Cloud).
 *
 * Proxyed through the Geo-Iris backend (`POST /api/v1/gridded/smap`).
 * The backend requires NASA Earthdata Login credentials
 * (EARTHDATA_USERNAME/EARTHDATA_PASSWORD or EARTHDATA_TOKEN); when they are not
 * configured the backend returns an explicit 'unavailable' result — the
 * connector never fabricates data.
 * Coverage: global land (85°S–85°N), 9 km EASE-Grid 2.0, 2015–present.
 * Variable: volumetric soil moisture (m³/m³). Root-zone (0–100 cm) is a
 * land-model estimate assimilating SMAP brightness temperatures, not a direct
 * observation. L4 latency ~7 days.
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

const smapCapabilities: ConnectorCapabilities = {
  timeSeries: true,
  mapTiles: false,
  catalogSearch: false,
  pointSubset: true,
  regionSubset: false,
  coverage: regionalCoverage(
    'SMAP L4 (SPL4SMGP V008) root-zone soil moisture, global land 85°S–85°N, 9 km EASE-Grid 2.0, 2015–present. Satellite brightness temperatures assimilated into a land model.',
    { regions: ['global land 85S-85N'] }
  ),
};

export const smapConnector: EarthDataConnector = {
  id: 'smap',
  name: 'SMAP L4 Soil Moisture',
  shortName: 'SMAP',
  description:
    'NASA SMAP Level-4 (SPL4SMGP V008) — 9 km gridded monthly volumetric soil moisture, global land, 2015–present. Root-zone (0–100 cm) is a land-model estimate.',
  baseUrl: 'https://nsidc.org/data/smap/spl4smgp/versions/8',
  capabilities: smapCapabilities,
  supportedVariables: ['soil_moist_root'],
  provenanceType: 'satellite_observation',
  requiresAuth: false,
  requiresProxy: true,

  supportsVariable(variableId: VariableId): boolean {
    return variableId === 'soil_moist_root';
  },

  coversPoint(lat: number, lon: number): boolean {
    return lat >= -85.044 && lat <= 85.044 && lon >= -180 && lon <= 180;
  },

  async fetchTimeSeries(options: ConnectorFetchOptions): Promise<ConnectorFetchResult> {
    const { lat, lon, variableId } = options;
    const startYear = options.startYear ?? 2015;
    const endYear = options.endYear ?? new Date().getFullYear();

    if (!this.supportsVariable(variableId)) {
      return {
        points: [],
        provenance: 'satellite_observation',
        sourceLabel: 'SMAP L4',
        statusMessage: `Variable "${variableId}" is not supported by SMAP (only 'soil_moist_root').`,
      };
    }
    if (!(this.coversPoint?.(lat, lon) ?? false)) {
      return {
        points: [],
        unavailable: true,
        provenance: 'satellite_observation',
        sourceLabel: 'SMAP L4',
        statusMessage: `SMAP does not cover (${lat.toFixed(2)}, ${lon.toFixed(2)}) — outside 85°S–85°N (land).`,
      };
    }

    try {
      const response = await fetch(`${GRIDDED_API_BASE}/gridded/smap`, {
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
          sourceLabel: 'SMAP L4',
          statusMessage: `SMAP request failed (HTTP ${response.status}): ${detail.slice(0, 300)}`,
        };
      }

      const json: GriddedSeriesResponse = await response.json();
      if (json.status === 'unavailable' || !json.series) {
        return {
          points: [],
          unavailable: true,
          provenance: 'satellite_observation',
          sourceLabel: 'SMAP L4',
          statusMessage: json.message ?? 'SMAP data unavailable for this location/range.',
        };
      }

      const points = toTimeSeriesPoints(json.series);
      const validCount = points.filter(p => p.raw !== null).length;
      return {
        points,
        provenance: 'satellite_observation',
        sourceLabel: `SMAP L4 (${json.series.dataset})`,
        statusMessage:
          json.message ??
          `Retrieved ${validCount} valid month(s) of SMAP L4 (${json.series.missing_count} missing), ${json.series.units}.`,
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
        sourceLabel: 'SMAP L4',
        statusMessage: `SMAP request failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  },
};

export default smapConnector;
