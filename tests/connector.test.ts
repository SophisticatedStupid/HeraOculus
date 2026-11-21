/**
 * Connector selection and error handling tests.
 *
 * Converted from tests/connector_selection.verify.ts. Validates that
 * connector selection is geographically defensible, deterministic, and
 * transparent:
 * - Regional providers (USGS streamflow, NOAA tide gauges) are never selected
 *   outside the United States / beyond 500 km of a station.
 * - Global providers (NASA POWER, ORNL MODIS) are selected at any valid
 *   coordinate they support.
 * - No suitable connector yields a deterministic fallback (connector:
 *   undefined) with an explanatory reason — never a masquerading live source.
 * - Selection is deterministic across repeated calls.
 */
import {
  connectorRegistry,
  type ConnectorSelection,
  type VariableId,
} from '../src/services/connectors';
import { powerConnector } from '../src/services/connectors/powerConnector';
import { modisConnector } from '../src/services/connectors/modisConnector';
import { usgsConnector } from '../src/services/connectors/usgsConnector';
import { noaaConnector } from '../src/services/connectors/noaaConnector';
import { stacConnector } from '../src/services/connectors/stacConnector';
import { describe, expect, it } from 'vitest';

const RESULTS: Array<{
  label: string;
  lat: number;
  lon: number;
  variableId: VariableId;
  expected?: string;
}> = [
  { label: 'Arctic — surface air temperature', lat: 82.0, lon: -50.0, variableId: 'temp_air', expected: 'nasa-power' },
  { label: 'Tropical — surface air temperature', lat: 5.0, lon: 75.0, variableId: 'temp_air', expected: 'nasa-power' },
  { label: 'Equatorial ocean — surface air temperature', lat: 0.0, lon: -120.0, variableId: 'temp_air', expected: 'nasa-power' },
  { label: 'Southern Hemisphere — surface air temperature', lat: -33.0, lon: -70.0, variableId: 'temp_air', expected: 'nasa-power' },
  { label: 'South Asia — NDVI', lat: 22.0, lon: 78.0, variableId: 'ndvi', expected: 'ornl-modis' },
  { label: 'Equatorial ocean — NDVI', lat: 0.0, lon: -120.0, variableId: 'ndvi', expected: 'ornl-modis' },
  { label: 'North America near US gauge — streamflow', lat: 38.9, lon: -77.1, variableId: 'streamflow', expected: 'usgs-water' },
  { label: 'North America coastal — sea level', lat: 40.7, lon: -74.0, variableId: 'sea_level', expected: 'noaa-tides' },
];

function selectedId(sel: ConnectorSelection): string | undefined {
  return sel.connector?.id;
}

function findCandidate(
  sel: ConnectorSelection,
  connectorId: string
): ConnectorSelection['candidates'][number] | undefined {
  return sel.candidates.find(c => c.connector.id === connectorId);
}

describe('connector registry', () => {
  it('registers all seven connectors', () => {
    const ids = connectorRegistry.getAll().map(c => c.id).sort();
    expect(ids).toHaveLength(7);
    expect(ids).toContain('nasa-power');
    expect(ids).toContain('ornl-modis');
    expect(ids).toContain('usgs-water');
    expect(ids).toContain('noaa-tides');
    expect(ids).toContain('nasa-cmr');
    expect(ids).toContain('nasa-gibs');
    expect(ids).toContain('stac-catalog');
  });

  it('distinguishes global vs regional coverage', () => {
    const byId = new Map(connectorRegistry.getAll().map(c => [c.id, c]));
    expect(byId.get('nasa-power')!.capabilities.coverage.kind).toBe('global');
    expect(byId.get('ornl-modis')!.capabilities.coverage.kind).toBe('global');
    expect(byId.get('usgs-water')!.capabilities.coverage.kind).toBe('regional');
    expect(byId.get('noaa-tides')!.capabilities.coverage.kind).toBe('regional');
    expect(byId.get('usgs-water')!.capabilities.coverage.maxStationDistanceKm).toBe(500);
    expect(byId.get('noaa-tides')!.capabilities.coverage.maxStationDistanceKm).toBe(500);
    expect(byId.get('nasa-cmr')!.capabilities.coverage.kind).toBe('global');
    expect(byId.get('nasa-gibs')!.capabilities.coverage.kind).toBe('global');
    expect(byId.get('stac-catalog')!.capabilities.coverage.kind).toBe('global');
  });
});

describe('regional coversPoint', () => {
  it('respects station distance limits', () => {
    expect(usgsConnector.coversPoint?.(38.9, -77.1)).toBe(true);
    expect(usgsConnector.coversPoint?.(82.0, -50.0)).toBe(false);
    expect(usgsConnector.coversPoint?.(-33.0, -70.0)).toBe(false);
    expect(usgsConnector.coversPoint?.(22.0, 78.0)).toBe(false);

    expect(noaaConnector.coversPoint?.(40.7, -74.0)).toBe(true);
    expect(noaaConnector.coversPoint?.(0.0, -120.0)).toBe(false);
    expect(noaaConnector.coversPoint?.(-33.0, -70.0)).toBe(false);
  });
});

describe('global provider selection', () => {
  it.each(RESULTS.filter(r => r.variableId === 'temp_air' || r.variableId === 'ndvi'))(
    'selects $expected for $label',
    ({ label, lat, lon, variableId, expected }) => {
      const sel = connectorRegistry.getBestConnector(variableId, lat, lon);
      expect(sel.fallback).toBe(false);
      expect(selectedId(sel)).toBe(expected);
      const chosen = sel.candidates.find(c => c.connector.id === expected);
      expect(chosen?.supportsVariable).toBe(true);
      expect(chosen?.coversLocation).toBe(true);
    }
  );
});

describe('regional rejection outside coverage', () => {
  it('returns a deterministic fallback for streamflow at the Arctic', () => {
    const sel = connectorRegistry.getBestConnector('streamflow', 82.0, -50.0);
    expect(sel.fallback).toBe(true);
    expect(selectedId(sel)).toBeUndefined();
    const usgs = findCandidate(sel, 'usgs-water');
    expect(usgs?.supportsVariable).toBe(true);
    expect(usgs?.coversLocation).toBe(false);
    expect(usgs?.reason).toContain('outside its service area');
    expect(sel.reason).not.toContain('isLive');
    expect(sel.reason).not.toContain('live source');
  });

  it('returns a deterministic fallback for sea level at the equatorial ocean', () => {
    const sel = connectorRegistry.getBestConnector('sea_level', 0.0, -120.0);
    expect(sel.fallback).toBe(true);
    expect(selectedId(sel)).toBeUndefined();
    const noaa = findCandidate(sel, 'noaa-tides');
    expect(noaa?.supportsVariable).toBe(true);
    expect(noaa?.coversLocation).toBe(false);
  });

  it('returns a fallback for streamflow in the Southern Hemisphere', () => {
    const sel = connectorRegistry.getBestConnector('streamflow', -33.0, -70.0);
    expect(sel.fallback).toBe(true);
    expect(selectedId(sel)).toBeUndefined();
  });

  it('returns a fallback for sea level in South Asia', () => {
    const sel = connectorRegistry.getBestConnector('sea_level', 22.0, 78.0);
    expect(sel.fallback).toBe(true);
    expect(selectedId(sel)).toBeUndefined();
  });
});

describe('regional selection inside coverage', () => {
  it('selects USGS for US streamflow near a gauge', () => {
    const sel = connectorRegistry.getBestConnector('streamflow', 38.9, -77.1);
    expect(sel.fallback).toBe(false);
    expect(selectedId(sel)).toBe('usgs-water');
    expect(usgsConnector.coversPoint?.(38.9, -77.1)).toBe(true);
  });

  it('selects NOAA for US coastal sea level', () => {
    const sel = connectorRegistry.getBestConnector('sea_level', 40.7, -74.0);
    expect(sel.fallback).toBe(false);
    expect(selectedId(sel)).toBe('noaa-tides');
    expect(noaaConnector.coversPoint?.(40.7, -74.0)).toBe(true);
  });
});

describe('no suitable connector', () => {
  it('falls back when the only supporter does not cover the location', () => {
    const sel = connectorRegistry.getBestConnector('streamflow', 82.0, -50.0);
    expect(sel.fallback).toBe(true);
    expect(selectedId(sel)).toBeUndefined();
    expect(sel.reason).toContain('no connector covers');
  });
});

describe('determinism', () => {
  it('produces identical results across repeated calls', () => {
    const cases: Array<[VariableId, number, number]> = [
      ['temp_air', 82.0, -50.0],
      ['ndvi', 22.0, 78.0],
      ['streamflow', 38.9, -77.1],
      ['sea_level', 0.0, -120.0],
    ];
    for (const [v, la, lo] of cases) {
      const a = connectorRegistry.getBestConnector(v, la, lo);
      const b = connectorRegistry.getBestConnector(v, la, lo);
      expect(selectedId(a)).toBe(selectedId(b));
      expect(a.reason).toBe(b.reason);
      expect(a.fallback).toBe(b.fallback);
      expect(a.candidates).toHaveLength(b.candidates.length);
    }
  });
});

describe('invalid coordinates', () => {
  it('produces a fallback, never a live source', () => {
    const bad = connectorRegistry.getBestConnector('temp_air', 91.0, 0.0);
    expect(bad.fallback).toBe(true);
    expect(selectedId(bad)).toBeUndefined();
    expect(bad.reason).toContain('Invalid coordinates');

    const badLon = connectorRegistry.getBestConnector('temp_air', 0.0, 200.0);
    expect(badLon.fallback).toBe(true);
    expect(selectedId(badLon)).toBeUndefined();
  });
});

describe('no masquerading live sources', () => {
  it('never selects a connector when the result is a fallback', () => {
    for (const r of RESULTS) {
      const sel = connectorRegistry.getBestConnector(r.variableId, r.lat, r.lon);
      if (sel.fallback) {
        expect(selectedId(sel)).toBeUndefined();
      }
      for (const c of sel.candidates) {
        if (c.supportsVariable && !c.coversLocation) {
          expect(c.reason).toContain('outside its service area');
        }
      }
    }
  });
});

describe('STAC connector — provenance labelling & capabilities', () => {
  it('is labelled satellite_observation with global catalog-only coverage', () => {
    expect(stacConnector.provenanceType).toBe('satellite_observation');
    expect(stacConnector.capabilities.catalogSearch).toBe(true);
    expect(stacConnector.capabilities.timeSeries).toBe(false);
    expect(stacConnector.capabilities.mapTiles).toBe(false);
    expect(stacConnector.capabilities.regionSubset).toBe(true);
    expect(stacConnector.capabilities.coverage.kind).toBe('global');
    expect(stacConnector.supportedVariables).toHaveLength(0);
    expect(stacConnector.id).toBe('stac-catalog');
  });

  it('fetchTimeSeries returns an explicit empty catalog-only result', async () => {
    const res = await stacConnector.fetchTimeSeries({
      lat: 37.0,
      lon: -122.0,
      variableId: 'ndvi',
    });
    expect(res.points).toEqual([]);
    expect(res.provenance).toBe('satellite_observation');
    expect(res.sourceLabel).toBe('STAC Catalog');
    expect(res.statusMessage).toContain('catalog search service');
  });

  it('is registered and discoverable in the registry', () => {
    const ids = connectorRegistry.getAll().map(c => c.id);
    expect(ids).toContain('stac-catalog');
    const byId = new Map(connectorRegistry.getAll().map(c => [c.id, c]));
    expect(byId.get('stac-catalog')).toBeDefined();
  });
});
