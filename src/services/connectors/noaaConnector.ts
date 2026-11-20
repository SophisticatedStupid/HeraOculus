/**
 * NOAA Tides & Currents Connector — Monthly mean sea level from coastal tide gauge stations
 * Implements EarthDataConnector for sea_level variable
 */
import type {
  EarthDataConnector,
  ConnectorCapabilities,
  ConnectorFetchOptions,
  ConnectorFetchResult,
  VariableId,
  StationMetadata,
} from './types';
import { haversineDistance } from './types';
import { processTimeSeries } from '../statistics';

const BASE_URL = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter';

// ── Coastal tide gauge stations ───────────────────────────────────────

interface TideStation {
  id: string;
  name: string;
  lat: number;
  lon: number;
}

/**
 * Maximum acceptable great-circle distance (km) between the query point and
 * a tide gauge for its monthly mean sea level to be considered a valid local
 * measurement. Beyond this threshold the connector returns an explicit
 * "no geographically appropriate station" status instead of silently
 * substituting a distant US station.
 *
 * 500 km is roughly the limit at which coastal sea-level anomalies remain
 * meaningfully correlated; beyond that, regional oceanography dominates.
 */
export const MAX_STATION_DISTANCE_KM = 500;

const TIDE_STATIONS: TideStation[] = [
  { id: '8518750', name: 'The Battery, New York, NY',   lat: 40.7006, lon: -74.0142 },
  { id: '9414290', name: 'San Francisco, CA',           lat: 37.8063, lon: -122.4659 },
  { id: '8443970', name: 'Boston, MA',                  lat: 42.3539, lon: -71.0503 },
  { id: '8658120', name: 'Wilmington, NC',              lat: 34.2275, lon: -77.9536 },
  { id: '8545240', name: 'Philadelphia, PA',            lat: 39.9334, lon: -75.1418 },
  { id: '8574680', name: 'Baltimore, MD',               lat: 39.2667, lon: -76.5794 },
  { id: '8726520', name: 'St. Petersburg, FL',          lat: 27.7606, lon: -82.6269 },
  { id: '9410660', name: 'Los Angeles, CA',             lat: 33.7200, lon: -118.2720 },
  { id: '9447130', name: 'Seattle, WA',                 lat: 47.6026, lon: -122.3393 },
  { id: '1612340', name: 'Honolulu, HI',                lat: 21.3067, lon: -157.8670 },
];

/**
 * Validate a coordinate pair as a real geographic location.
 * Rejects NaN, out-of-range latitude/longitude.
 */
export function isValidCoordinate(lat: number, lon: number): boolean {
  return (
    Number.isFinite(lat) &&
    Number.isFinite(lon) &&
    lat >= -90 && lat <= 90 &&
    lon >= -180 && lon <= 180
  );
}

interface NearestStationResult {
  station: TideStation;
  distanceKm: number;
}

/**
 * Find the nearest NOAA tide station to the given coordinates, or null when
 * no station lies within {@link MAX_STATION_DISTANCE_KM}. Never silently
 * returns a distant US station for a query outside that radius.
 */
export function findNearestStation(lat: number, lon: number): NearestStationResult | null {
  if (!isValidCoordinate(lat, lon)) return null;

  let nearest: TideStation | null = null;
  let minDist = Infinity;

  for (const station of TIDE_STATIONS) {
    const dist = haversineDistance(lat, lon, station.lat, station.lon);
    if (dist < minDist) {
      minDist = dist;
      nearest = station;
    }
  }

  if (!nearest || minDist > MAX_STATION_DISTANCE_KM) return null;
  return { station: nearest, distanceKm: minDist };
}

// ── Connector implementation ──────────────────────────────────────────

const noaaCapabilities: ConnectorCapabilities = {
  timeSeries: true,
  mapTiles: false,
  catalogSearch: false,
  pointSubset: true,
  regionSubset: false,
};

const SUPPORTED_VARIABLES: VariableId[] = ['sea_level'];

/** Build the unavailable result returned when no nearby tide gauge exists. */
function unavailableResult(lat: number, lon: number, startYear: number, endYear: number): ConnectorFetchResult {
  return {
    points: [],
    provenance: 'in_situ_gauge',
    sourceLabel: 'NOAA Tides',
    unavailable: true,
    requestedRange: `${startYear}-${endYear}`,
    statusMessage:
      `No NOAA tide gauge within ${MAX_STATION_DISTANCE_KM} km of ` +
      `(${lat.toFixed(3)}°, ${lon.toFixed(3)}°). Monthly mean sea level ` +
      `is not available as a local in-situ measurement for this location. ` +
      `Try a coastal region near a US station, or select a different variable.`,
  };
}

export const noaaConnector: EarthDataConnector = {
  id: 'noaa-tides',
  name: 'NOAA Tides & Currents',
  shortName: 'NOAA Tides',
  description:
    'NOAA CO-OPS Tides & Currents API — monthly mean sea level observations from U.S. coastal tide gauge stations. Datum: MSL, units: meters.',
  baseUrl: BASE_URL,
  capabilities: noaaCapabilities,
  supportedVariables: SUPPORTED_VARIABLES,
  provenanceType: 'in_situ_gauge',
  requiresAuth: false,
  requiresProxy: false,

  supportsVariable(variableId: VariableId): boolean {
    return variableId === 'sea_level';
  },

  async fetchTimeSeries(options: ConnectorFetchOptions): Promise<ConnectorFetchResult> {
    const { lat, lon, variableId, startYear = 2000, endYear = 2024 } = options;

    if (variableId !== 'sea_level') {
      return {
        points: [],
        provenance: 'in_situ_gauge',
        sourceLabel: 'NOAA Tides',
        statusMessage: `Variable "${variableId}" is not supported by NOAA Tides connector.`,
      };
    }

    if (!isValidCoordinate(lat, lon)) {
      return {
        points: [],
        provenance: 'in_situ_gauge',
        sourceLabel: 'NOAA Tides',
        unavailable: true,
        requestedRange: `${startYear}-${endYear}`,
        statusMessage:
          `Invalid query coordinates (${lat}, ${lon}). Provide a latitude in ` +
          `[-90, 90] and longitude in [-180, 180].`,
      };
    }

    const nearest = findNearestStation(lat, lon);
    if (!nearest) {
      return unavailableResult(lat, lon, startYear, endYear);
    }
    const { station, distanceKm: distKm } = nearest;

    const startDate = `${startYear}0101`;
    const endDate = `${endYear}1231`;

    const url =
      `${BASE_URL}` +
      `?begin_date=${startDate}` +
      `&end_date=${endDate}` +
      `&station=${station.id}` +
      `&product=monthly_mean` +
      `&datum=MSL` +
      `&units=metric` +
      `&time_zone=gmt` +
      `&format=json`;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 7000);

      const response = await fetch(url, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`NOAA API returned status ${response.status}`);
      }

      const json = await response.json();

      if (!json.data || !Array.isArray(json.data)) {
        throw new Error('Invalid NOAA response: missing data array');
      }

      const rawPoints: { date: string; value: number }[] = [];

      for (const entry of json.data) {
        const t: string = entry.t || ''; // format: "YYYY-MM-DD HH:MM" or "YYYY-MM"
        const v: string = entry.v || '';
        const val = parseFloat(v);

        if (isNaN(val)) continue;

        let dateStr: string;
        if (t.length >= 7) {
          dateStr = t.substring(0, 7);
        } else {
          continue;
        }

        // Exclude annual summary keys (month 13) — not a monthly observation
        const month = parseInt(dateStr.substring(5, 7), 10);
        if (month < 1 || month > 12) continue;

        rawPoints.push({
          date: dateStr,
          value: +val.toFixed(4),
        });
      }

      if (rawPoints.length === 0) {
        return {
          points: [],
          provenance: 'in_situ_gauge',
          sourceLabel: 'NOAA Tides',
          unavailable: true,
          requestedRange: `${startYear}-${endYear}`,
          station: buildStationMetadata(station, distKm, rawPoints, startYear, endYear),
          statusMessage:
            `NOAA tide gauge ${station.name} (${station.id}) returned no valid ` +
            `monthly observations in ${startYear}-${endYear}.`,
        };
      }

      if (rawPoints.length < 12) {
        return {
          points: [],
          provenance: 'in_situ_gauge',
          sourceLabel: 'NOAA Tides',
          unavailable: true,
          requestedRange: `${startYear}-${endYear}`,
          station: buildStationMetadata(station, distKm, rawPoints, startYear, endYear),
          statusMessage:
            `Insufficient NOAA data from ${station.name} (${station.id}): ` +
            `only ${rawPoints.length} monthly points. At least 12 are required.`,
        };
      }

      rawPoints.sort((a, b) => a.date.localeCompare(b.date));

      const actualStart = rawPoints[0].date;
      const actualEnd = rawPoints[rawPoints.length - 1].date;
      const latestValidObservation = actualEnd;

      return {
        points: processTimeSeries(rawPoints),
        provenance: 'in_situ_gauge',
        sourceLabel: `NOAA Tides (${station.name})`,
        statusMessage:
          `Retrieved ${rawPoints.length} months of sea level data from ${station.name} ` +
          `(${station.id}, ${distKm.toFixed(0)} km from query point).`,
        station: buildStationMetadata(station, distKm, rawPoints, startYear, endYear),
        requestedRange: `${startYear}-${endYear}`,
        actualRange: `${actualStart.slice(0, 4)}-${actualEnd.slice(0, 4)}`,
        latestValidObservation,
      };
    } catch (err) {
      console.warn(`NOAA Tides connector error:`, err);
      return {
        points: [],
        provenance: 'in_situ_gauge',
        sourceLabel: 'NOAA Tides',
        requestedRange: `${startYear}-${endYear}`,
        statusMessage: `NOAA Tides request failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  },
};

/**
 * Build the scientific metadata object describing the selected tide gauge
 * and the observations actually returned.
 */
function buildStationMetadata(
  station: TideStation,
  distKm: number,
  rawPoints: { date: string; value: number }[],
  startYear: number,
  endYear: number,
): StationMetadata {
  const dates = rawPoints.map(p => p.date).sort();
  const obsStart = dates.length ? dates[0].slice(0, 4) : String(startYear);
  const obsEnd = dates.length ? dates[dates.length - 1].slice(0, 4) : String(endYear);
  return {
    stationId: station.id,
    stationName: station.name,
    stationCoordinates: [station.lat, station.lon],
    distanceKm: +distKm.toFixed(1),
    measurementType: 'Tide gauge',
    observationPeriod: `${obsStart}-${obsEnd}`,
    units: 'm',
    sourceAgency: 'NOAA CO-OPS',
  };
}