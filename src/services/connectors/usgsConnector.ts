/**
 * USGS Water Services Connector — National Water Information System (NWIS)
 * Implements EarthDataConnector for US river streamflow discharge via NWIS monthly statistics (RDB format).
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

const BASE_URL = 'https://waterservices.usgs.gov/nwis';

/** Cubic feet per second to cubic meters per second conversion factor */
const CFS_TO_CMS = 0.0283168466;

/**
 * Maximum acceptable great-circle distance (km) between the query point and
 * a streamflow gauge for its monthly mean discharge to be considered a valid
 * local measurement. Beyond this threshold the connector returns an explicit
 * "no geographically appropriate station" status instead of silently
 * substituting a distant US station.
 *
 * 500 km is roughly the limit at which river discharge remains meaningfully
 * correlated within the same basin; beyond that, different hydrologic regimes
 * dominate.
 */
export const MAX_STATION_DISTANCE_KM = 500;

/** Curated USGS NWIS Streamflow Gauge Stations */
const GAUGE_SITES = [
  { id: '01646500', name: 'Potomac River near Washington, DC', lat: 38.9498, lon: -77.1276 },
  { id: '07010000', name: 'Mississippi River at St. Louis, MO', lat: 38.6245, lon: -90.1848 },
  { id: '09380000', name: 'Colorado River at Lees Ferry, AZ', lat: 36.865, lon: -111.588 },
  { id: '14105700', name: 'Columbia River at The Dalles, OR', lat: 45.6075, lon: -121.172 },
  { id: '01357500', name: 'Mohawk River at Cohoes, NY', lat: 42.7853, lon: -73.7083 },
  { id: '08066500', name: 'Trinity River at Romayor, TX', lat: 30.425, lon: -94.851 },
  { id: '02336000', name: 'Chattahoochee River at Atlanta, GA', lat: 33.859, lon: -84.454 },
  { id: '12113000', name: 'Green River near Auburn, WA', lat: 47.3125, lon: -122.203 },
  { id: '06893000', name: 'Missouri River at Kansas City, MO', lat: 39.112, lon: -94.588 },
  { id: '02236000', name: 'St. Johns River near De Land, FL', lat: 29.008, lon: -81.383 },
];

interface GaugeSite {
  id: string;
  name: string;
  lat: number;
  lon: number;
}

interface NearestSiteResult {
  site: GaugeSite;
  distanceKm: number;
}

/** Validate a coordinate pair as a real geographic location. */
export function isValidCoordinate(lat: number, lon: number): boolean {
  return (
    Number.isFinite(lat) &&
    Number.isFinite(lon) &&
    lat >= -90 && lat <= 90 &&
    lon >= -180 && lon <= 180
  );
}

/**
 * Find the nearest USGS streamflow gauge to the given coordinates, or null when
 * no station lies within {@link MAX_STATION_DISTANCE_KM}. Never silently
 * returns a distant US station for a query outside that radius.
 */
export function findNearestSite(lat: number, lon: number): NearestSiteResult | null {
  if (!isValidCoordinate(lat, lon)) return null;

  let nearest: GaugeSite | null = null;
  let minDist = Infinity;

  for (const s of GAUGE_SITES) {
    const dist = haversineDistance(lat, lon, s.lat, s.lon);
    if (dist < minDist) {
      minDist = dist;
      nearest = s;
    }
  }

  if (!nearest || minDist > MAX_STATION_DISTANCE_KM) return null;
  return { site: nearest, distanceKm: minDist };
}

const usgsCapabilities: ConnectorCapabilities = {
  timeSeries: true,
  mapTiles: false,
  catalogSearch: false,
  pointSubset: true,
  regionSubset: false,
};

const SUPPORTED_VARIABLES: VariableId[] = ['streamflow'];

/** Build the unavailable result returned when no nearby streamflow gauge exists. */
function unavailableResult(lat: number, lon: number, startYear: number, endYear: number): ConnectorFetchResult {
  return {
    points: [],
    provenance: 'in_situ_gauge',
    sourceLabel: 'USGS Water',
    unavailable: true,
    requestedRange: `${startYear}-${endYear}`,
    statusMessage:
      `No USGS streamflow gauge within ${MAX_STATION_DISTANCE_KM} km of ` +
      `(${lat.toFixed(3)}°, ${lon.toFixed(3)}°). Monthly mean streamflow ` +
      `is not available as a local in-situ measurement for this location. ` +
      `Try a riverine region near a US gauge, or select a different variable.`,
  };
}

/** Parse USGS tab-delimited RDB monthly statistics text into YYYY-MM -> mean m^3/s */
function parseUsgsRdbMonthly(rdbText: string): { date: string; value: number }[] {
  const lines = rdbText.split(/\r?\n/);
  let headers: string[] | null = null;
  let skippedTypeRow = false;

  const yearIdxFallback = 5;
  const monthIdxFallback = 6;
  const meanIdxFallback = 7;

  let yearIdx = yearIdxFallback;
  let monthIdx = monthIdxFallback;
  let meanIdx = meanIdxFallback;

  const monthlyBuckets = new Map<string, number[]>();

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    const cols = rawLine.split('\t');

    if (!headers) {
      headers = cols.map(c => c.trim());
      const yi = headers.indexOf('year_nu');
      const mi = headers.indexOf('month_nu');
      const vi = headers.indexOf('mean_va');
      if (yi !== -1) yearIdx = yi;
      if (mi !== -1) monthIdx = mi;
      if (vi !== -1) meanIdx = vi;
      continue;
    }

    if (!skippedTypeRow) {
      skippedTypeRow = true;
      // RDB always places column format definitions like "5s\t15s..." right under header
      if (cols.every(c => /^\d+[snd]$/.test(c.trim()))) {
        continue;
      }
    }

    const year = parseInt(cols[yearIdx]?.trim() || '', 10);
    const month = parseInt(cols[monthIdx]?.trim() || '', 10);
    const valCfs = parseFloat(cols[meanIdx]?.trim() || '');

    if (isNaN(year) || isNaN(month) || month < 1 || month > 12 || isNaN(valCfs) || valCfs < 0) {
      continue;
    }

    const dateStr = `${year}-${month < 10 ? `0${month}` : month}`;
    const valueCms = valCfs * CFS_TO_CMS;
    const bucket = monthlyBuckets.get(dateStr) || [];
    bucket.push(valueCms);
    monthlyBuckets.set(dateStr, bucket);
  }

  const allPoints: { date: string; value: number }[] = [];
  for (const [date, vals] of monthlyBuckets.entries()) {
    const avg = vals.reduce((s, v) => s + v, 0) / vals.length;
    allPoints.push({ date, value: +avg.toFixed(3) });
  }
  allPoints.sort((a, b) => a.date.localeCompare(b.date));

  // Focus on the modern 1995-present era when available so charts align cleanly
  const modernPoints = allPoints.filter(p => parseInt(p.date.slice(0, 4), 10) >= 1995);
  return modernPoints.length >= 24 ? modernPoints : allPoints;
}

export const usgsConnector: EarthDataConnector = {
  id: 'usgs-water',
  name: 'USGS Water Services (NWIS)',
  shortName: 'USGS Water',
  description:
    'USGS National Water Information System (NWIS) — monthly mean streamflow from major U.S. river gauge stations. Parameter code 00060 (discharge in ft³/s, converted to m³/s).',
  baseUrl: BASE_URL,
  capabilities: usgsCapabilities,
  supportedVariables: SUPPORTED_VARIABLES,
  provenanceType: 'in_situ_gauge',
  requiresAuth: false,
  requiresProxy: false,

  supportsVariable(variableId: VariableId): boolean {
    return variableId === 'streamflow';
  },

  async fetchTimeSeries(options: ConnectorFetchOptions): Promise<ConnectorFetchResult> {
    const { lat, lon, variableId, startYear = 1995, endYear = new Date().getFullYear() } = options;

    if (variableId !== 'streamflow') {
      return {
        points: [],
        provenance: 'in_situ_gauge',
        sourceLabel: 'USGS Water',
        statusMessage: `Variable "${variableId}" is not supported by USGS Water connector.`,
      };
    }

    if (!isValidCoordinate(lat, lon)) {
      return {
        points: [],
        provenance: 'in_situ_gauge',
        sourceLabel: 'USGS Water',
        unavailable: true,
        requestedRange: `${startYear}-${endYear}`,
        statusMessage:
          `Invalid query coordinates (${lat}, ${lon}). Provide a latitude in ` +
          `[-90, 90] and longitude in [-180, 180].`,
      };
    }

    const nearest = findNearestSite(lat, lon);
    if (!nearest) {
      return unavailableResult(lat, lon, startYear, endYear);
    }
    const { site, distanceKm: distKm } = nearest;

    // Note: USGS /nwis/stat/ supports format=rdb (tab-separated) and rejects format=json with 400
    const url =
      `${BASE_URL}/stat/` +
      `?format=rdb` +
      `&sites=${site.id}` +
      `&statReportType=monthly` +
      `&statTypeCd=mean` +
      `&parameterCd=00060`;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 9000);

      const response = await fetch(url, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`USGS API returned status ${response.status}`);
      }

      const rdbText = await response.text();
      const rawPoints = parseUsgsRdbMonthly(rdbText);

      if (rawPoints.length === 0) {
        return {
          points: [],
          provenance: 'in_situ_gauge',
          sourceLabel: 'USGS Water',
          unavailable: true,
          requestedRange: `${startYear}-${endYear}`,
          station: buildStationMetadata(site, distKm, rawPoints, startYear, endYear),
          statusMessage:
            `USGS streamflow gauge ${site.name} (${site.id}) returned no valid ` +
            `monthly observations in ${startYear}-${endYear}.`,
        };
      }

      if (rawPoints.length < 12) {
        return {
          points: [],
          provenance: 'in_situ_gauge',
          sourceLabel: 'USGS Water',
          unavailable: true,
          requestedRange: `${startYear}-${endYear}`,
          station: buildStationMetadata(site, distKm, rawPoints, startYear, endYear),
          statusMessage:
            `Insufficient USGS data from ${site.name} (${site.id}): ` +
            `only ${rawPoints.length} monthly points. At least 12 are required.`,
        };
      }

      const actualStart = rawPoints[0].date;
      const actualEnd = rawPoints[rawPoints.length - 1].date;
      const latestValidObservation = actualEnd;

      return {
        points: processTimeSeries(rawPoints),
        provenance: 'in_situ_gauge',
        sourceLabel: `USGS (${site.name})`,
        statusMessage:
          `Retrieved ${rawPoints.length} months of streamflow data from ${site.name} ` +
          `(${site.id}, ${distKm.toFixed(0)} km from query point).`,
        station: buildStationMetadata(site, distKm, rawPoints, startYear, endYear),
        requestedRange: `${startYear}-${endYear}`,
        actualRange: `${actualStart.slice(0, 4)}-${actualEnd.slice(0, 4)}`,
        latestValidObservation,
      };
    } catch (err) {
      console.warn(`USGS Water connector error:`, err);
      return {
        points: [],
        provenance: 'in_situ_gauge',
        sourceLabel: 'USGS Water',
        requestedRange: `${startYear}-${endYear}`,
        statusMessage: `USGS request failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  },
};

/**
 * Build the scientific metadata object describing the selected streamflow gauge
 * and the observations actually returned.
 */
function buildStationMetadata(
  site: GaugeSite,
  distKm: number,
  rawPoints: { date: string; value: number }[],
  startYear: number,
  endYear: number,
): StationMetadata {
  const dates = rawPoints.map(p => p.date).sort();
  const obsStart = dates.length ? dates[0].slice(0, 4) : String(startYear);
  const obsEnd = dates.length ? dates[dates.length - 1].slice(0, 4) : String(endYear);
  return {
    stationId: site.id,
    stationName: site.name,
    stationCoordinates: [site.lat, site.lon],
    distanceKm: +distKm.toFixed(1),
    measurementType: 'Streamflow gauge',
    observationPeriod: `${obsStart}-${obsEnd}`,
    units: 'm³/s',
    sourceAgency: 'USGS NWIS',
  };
}