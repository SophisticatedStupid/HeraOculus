/**
 * Connector Registry — Central registry for all EarthDataConnectors
 *
 * Selection is geographically aware: candidates are filtered by variable
 * support AND by whether the connector actually covers the query point.
 * Regional providers are never selected outside their service area, and the
 * outcome (selected connector, rejected candidates, fallback reason) is fully
 * transparent and deterministic.
 */
import type {
  EarthDataConnector,
  VariableId,
} from './types';
import { haversineDistance } from './types';

/** Verdict for a single connector considered during selection. */
export interface ConnectorCandidateVerdict {
  connector: EarthDataConnector;
  /** True when the connector supports the requested variable. */
  supportsVariable: boolean;
  /** True when the connector covers the query point (global always true; regional via coversPoint). */
  coversLocation: boolean;
  /** Numeric rank score (higher = preferred). Only meaningful for covered supporters. */
  score: number;
  /** Human-readable explanation of acceptance or rejection. */
  reason: string;
}

/** Outcome of getBestConnector. `connector` is undefined when no suitable
 * connector exists — the deterministic fallback. Callers must not present a
 * failed/regional-outside-area connector as a successful live source. */
export interface ConnectorSelection {
  /** Selected connector, or undefined when no connector is suitable. */
  connector: EarthDataConnector | undefined;
  /** True when this is a deliberate "no suitable connector" fallback. */
  fallback: boolean;
  /** Human-readable explanation of the selection (or rejection of all candidates). */
  reason: string;
  /** Every registered connector considered, with accept/reject reasons. */
  candidates: ConnectorCandidateVerdict[];
}

/** Provenance preference order (higher score = preferred). */
const PROVENANCE_SCORE: Record<string, number> = {
  satellite_observation: 3,
  in_situ_gauge: 2,
  reanalysis_model: 1,
  user_upload: 0,
  synthetic_demo: -1,
};

function scoreConnector(c: EarthDataConnector): number {
  return (
    (PROVENANCE_SCORE[c.provenanceType] ?? 0) * 100 +
    (c.capabilities.timeSeries ? 10 : 0) +
    (c.capabilities.pointSubset ? 5 : 0) +
    (c.capabilities.regionSubset ? 2 : 0)
  );
}

/** Validate a coordinate pair as a real geographic location. */
function isValidCoordinate(lat: number, lon: number): boolean {
  return (
    Number.isFinite(lat) &&
    Number.isFinite(lon) &&
    lat >= -90 && lat <= 90 &&
    lon >= -180 && lon <= 180
  );
}

/** Test whether a connector covers the given point. */
function coversPoint(c: EarthDataConnector, lat: number, lon: number): boolean {
  if (c.capabilities.coverage.kind === 'global') return true;
  // Regional: defer to the connector's own implementation when provided;
  // conservatively false when the connector does not declare one.
  return typeof c.coversPoint === 'function' ? c.coversPoint!(lat, lon) : false;
}

function describeCoverage(c: EarthDataConnector): string {
  const cov = c.capabilities.coverage;
  if (cov.kind === 'regional' && cov.maxStationDistanceKm != null) {
    return `${cov.description} (nearest-station limit: ${cov.maxStationDistanceKm} km)`;
  }
  return cov.description;
}

class ConnectorRegistry {
  private connectors: Map<string, EarthDataConnector> = new Map();

  /** Register a connector */
  register(connector: EarthDataConnector): void {
    this.connectors.set(connector.id, connector);
  }

  /** Get a connector by ID */
  get(id: string): EarthDataConnector | undefined {
    return this.connectors.get(id);
  }

  /** Get all registered connectors */
  getAll(): EarthDataConnector[] {
    return Array.from(this.connectors.values());
  }

  /** Find all connectors that support a given variable */
  findForVariable(variableId: VariableId): EarthDataConnector[] {
    return this.getAll().filter(c => c.supportsVariable(variableId));
  }

  /**
   * Get the best connector for a variable at a given location.
   *
   * Ranking dimensions: variable support (gate), spatial coverage (gate),
   * temporal/data availability (provenance + timeSeries + pointSubset score),
   * with a deterministic ID tie-break.
   *
   * Regional providers are rejected outright when the point lies outside their
   * service area — never down-ranked into selection. When no connector is
   * suitable, returns a deterministic fallback with `connector: undefined` and
   * an explanatory reason; no failed connector is presented as live.
   */
  getBestConnector(
    variableId: VariableId,
    lat: number,
    lon: number
  ): ConnectorSelection {
    const all = this.getAll();

    // Coordinate validity is a precondition for any geographic selection.
    if (!isValidCoordinate(lat, lon)) {
      const candidates = all.map(c => buildVerdict(c, variableId, lat, lon, false));
      return {
        connector: undefined,
        fallback: true,
        reason: `Invalid coordinates (${lat}, ${lon}): latitude must be in [-90, 90] and longitude in [-180, 180]. No connector can serve this location.`,
        candidates,
      };
    }

    // Build a verdict for every registered connector (transparency).
    const candidates = all.map(c => buildVerdict(c, variableId, lat, lon, true));

    const coveredSupporters = candidates.filter(cd => cd.supportsVariable && cd.coversLocation);

    if (coveredSupporters.length === 0) {
      const noSupport = all.every(c => !c.supportsVariable(variableId));
      let reason: string;
      if (noSupport) {
        reason = `No registered connector supports variable "${variableId}".`;
      } else {
        const rejected = candidates
          .filter(cd => cd.supportsVariable && !cd.coversLocation)
          .map(cd => `${cd.connector.id} (${describeCoverage(cd.connector)})`)
          .join('; ');
        reason = `Variable "${variableId}" is supported, but no connector covers (${lat.toFixed(3)}°, ${lon.toFixed(3)}°). Rejected by coverage: ${rejected}.`;
      }
      return {
        connector: undefined,
        fallback: true,
        reason,
        candidates,
      };
    }

    // Deterministic ranking: score desc, then connector id asc.
    coveredSupporters.sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return a.connector.id.localeCompare(b.connector.id);
    });

    const best = coveredSupporters[0];
    const cov = best.connector.capabilities.coverage;
    const capabilityNotes: string[] = [];
    if (best.connector.capabilities.timeSeries) capabilityNotes.push('time series');
    if (best.connector.capabilities.pointSubset) capabilityNotes.push('point subset');
    if (best.connector.capabilities.regionSubset) capabilityNotes.push('region subset');
    const capabilityText = capabilityNotes.length > 0 ? ` (${capabilityNotes.join(', ')})` : '';

    const reason = `Selected ${best.connector.name} (${best.connector.id}): ${best.connector.provenanceType}; ${describeCoverage(best.connector)}; covers this location; supports "${variableId}"${capabilityText}. Score ${best.score}.`;

    return {
      connector: best.connector,
      fallback: false,
      reason,
      candidates,
    };
  }
}

function buildVerdict(
  c: EarthDataConnector,
  variableId: VariableId,
  lat: number,
  lon: number,
  includeCoverage: boolean
): ConnectorCandidateVerdict {
  const supports = c.supportsVariable(variableId);
  const covered = includeCoverage ? coversPoint(c, lat, lon) : false;
  const score = scoreConnector(c);

  let reason: string;
  if (!supports) {
    reason = `Rejected ${c.name} (${c.id}): does not support variable "${variableId}".`;
  } else if (!covered) {
    reason = `Rejected ${c.name} (${c.id}): ${describeCoverage(c)} — this point lies outside its service area.`;
  } else {
    reason = `Accepted ${c.name} (${c.id}): ${c.provenanceType}; covers this location; supports "${variableId}". Score ${score}.`;
  }

  return { connector: c, supportsVariable: supports, coversLocation: covered, score, reason };
}

export const connectorRegistry = new ConnectorRegistry();
