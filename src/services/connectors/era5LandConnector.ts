/**
 * ERA5-Land connector — gridded monthly total precipitation (Copernicus CDS /
 * ARCO Zarr mirror).
 *
 * Proxyed through the HeraOculus backend (`POST /api/v1/gridded/era5-land`).
 * The backend requires a CDS API key (CDS_API_KEY/CDSAPI_KEY) or a DestinE PAT
 * (DESTINE_PAT); when none are configured it returns an explicit
 * 'unavailable' result — the connector never fabricates data.
 * Coverage: global land (90°S–90°N), 0.1° grid, 1950–present.
 * Variable: total precipitation (stored in metres; converted to mm/month by
 * the backend). Provenance: reanalysis_model. Updated daily, ~5 days behind
 * real time; CDS access is queue-throttled (prefer the ARCO Zarr mirror).
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

const era5LandCapabilities: ConnectorCapabilities = {
  timeSeries: true,
  mapTiles: false,
  catalogSearch: false,
  pointSubset: true,
  regionSubset: false,
  coverage: regionalCoverage(
    'ERA5-Land reanalysis: global land (90°S–90°N), 0.1° grid, 1950–present. Land-only; ocean cells are fill and reported missing.',
    { regions: ['global land'] }
  ),
};

export const era5LandConnector: EarthDataConnector = {
  id: 'era5-land',
  name: 'ERA5-Land Monthly Precipitation',
  shortName: 'ERA5-Land',
  description:
    'ECMWF ERA5-Land reanalysis — 0.1° gridded monthly total precipitation, global land, 1950–present. Stored in metres; backend converts to mm/month.',
  baseUrl: 'https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land',
  capabilities: era5LandCapabilities,
  supportedVariables: ['precip'],
  provenanceType: 'reanalysis_model',
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
    const startYear = options.startYear ?? 1950;
    const endYear = options.endYear ?? new Date().getFullYear();

    if (!this.supportsVariable(variableId)) {
      return {
        points: [],
        provenance: 'reanalysis_model',
        sourceLabel: 'ERA5-Land',
        statusMessage: `Variable "${variableId}" is not supported by ERA5-Land (only 'precip').`,
      };
    }
    if (!(this.coversPoint?.(lat, lon) ?? false)) {
      return {
        points: [],
        unavailable: true,
        provenance: 'reanalysis_model',
        sourceLabel: 'ERA5-Land',
        statusMessage: `ERA5-Land does not cover (${lat.toFixed(2)}, ${lon.toFixed(2)}) — outside 90°S–90°N.`,
      };
    }

    try {
      const response = await fetch(`${GRIDDED_API_BASE}/gridded/era5-land`, {
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
          provenance: 'reanalysis_model',
          sourceLabel: 'ERA5-Land',
          statusMessage: `ERA5-Land request failed (HTTP ${response.status}): ${detail.slice(0, 300)}`,
        };
      }

      const json: GriddedSeriesResponse = await response.json();
      if (json.status === 'unavailable' || !json.series) {
        return {
          points: [],
          unavailable: true,
          provenance: 'reanalysis_model',
          sourceLabel: 'ERA5-Land',
          statusMessage: json.message ?? 'ERA5-Land data unavailable for this location/range.',
        };
      }

      const points = toTimeSeriesPoints(json.series);
      const validCount = points.filter(p => p.raw !== null).length;
      const latest = points.slice().reverse().find(p => p.raw !== null);
      return {
        points,
        provenance: 'reanalysis_model',
        sourceLabel: `ERA5-Land (${json.series.dataset})`,
        statusMessage:
          json.message ??
          `Retrieved ${validCount} valid month(s) of ERA5-Land (${json.series.missing_count} missing), ${json.series.units}.`,
        requestedRange: `${startYear}-${endYear}`,
        actualRange: `${startYear}-${endYear}`,
        latestValidObservation: latest?.date ?? undefined,
      };
    } catch (err) {
      return {
        points: [],
        provenance: 'reanalysis_model',
        sourceLabel: 'ERA5-Land',
        statusMessage: `ERA5-Land request failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  },
};

export default era5LandConnector;
