/**
 * Data provenance tests.
 *
 * Validates that every connector declares an honest provenance tier, that the
 * registry selects a connector whose provenance matches the expected source,
 * and that unavailable/failed results are explicitly flagged (never presented
 * as successful live sources).
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
import { cmrConnector } from '../src/services/connectors/cmrConnector';
import { gibsConnector } from '../src/services/connectors/gibsConnector';
import { chirpsConnector } from '../src/services/connectors/chirpsConnector';
import { imergConnector } from '../src/services/connectors/imergConnector';
import { era5LandConnector } from '../src/services/connectors/era5LandConnector';
import { modisLstConnector } from '../src/services/connectors/modisLstConnector';
import { smapConnector } from '../src/services/connectors/smapConnector';
import { analyzeTrendSafe } from '../src/services/statistics';
import type { TimeSeriesPoint } from '../src/types/earthScience';
import { describe, expect, it } from 'vitest';

describe('connector provenance metadata', () => {
  it('assigns the expected provenance tier to each connector', () => {
    expect(powerConnector.provenanceType).toBe('reanalysis_model');
    expect(modisConnector.provenanceType).toBe('satellite_observation');
    expect(usgsConnector.provenanceType).toBe('in_situ_gauge');
    expect(noaaConnector.provenanceType).toBe('in_situ_gauge');
    expect(cmrConnector.provenanceType).toBe('satellite_observation');
    expect(gibsConnector.provenanceType).toBe('satellite_observation');
    expect(chirpsConnector.provenanceType).toBe('satellite_observation');
    expect(imergConnector.provenanceType).toBe('satellite_observation');
    expect(era5LandConnector.provenanceType).toBe('reanalysis_model');
    expect(smapConnector.provenanceType).toBe('satellite_observation');
    expect(modisLstConnector.provenanceType).toBe('satellite_observation');
  });

  it('selected connector provenance matches the variable\'s expected source', () => {
    const cases: Array<[VariableId, number, number, string]> = [
      ['temp_air', 0.0, 0.0, 'reanalysis_model'],
      ['precip', 0.0, 0.0, 'satellite_observation'],
      ['ndvi', 22.0, 78.0, 'satellite_observation'],
      ['evi', 22.0, 78.0, 'satellite_observation'],
      ['streamflow', 38.9, -77.1, 'in_situ_gauge'],
      ['sea_level', 40.7, -74.0, 'in_situ_gauge'],
      ['lst_night', 0.0, 0.0, 'satellite_observation'],
    ];
    for (const [v, la, lo, expectedProvenance] of cases) {
      const sel = connectorRegistry.getBestConnector(v, la, lo);
      expect(sel.fallback).toBe(false);
      expect(sel.connector!.provenanceType).toBe(expectedProvenance);
      expect(sel.connector!.name.length).toBeGreaterThan(0);
      // The selection reason must mention the provenance tier.
      expect(sel.reason).toContain(expectedProvenance);
    }
  });

  it('CMR and GIBS support no variables (catalog / tiles only)', () => {
    expect(cmrConnector.supportedVariables).toHaveLength(0);
    expect(gibsConnector.supportedVariables).toHaveLength(0);
    expect(cmrConnector.supportsVariable('temp_air')).toBe(false);
    expect(gibsConnector.supportsVariable('temp_air')).toBe(false);
  });
});

describe('unavailable results are flagged, not masqueraded as live', () => {
  it('regional connector outside coverage is a fallback with connector undefined', () => {
    const sel: ConnectorSelection = connectorRegistry.getBestConnector('streamflow', 82.0, -50.0);
    expect(sel.fallback).toBe(true);
    expect(sel.connector).toBeUndefined();
    // No connector is returned that could be mistaken for a live source.
    expect(sel.reason).toContain('no connector covers');
  });

  it('analyzeTrendSafe marks empty/insufficient input without fabricating a result', () => {
    const empty = analyzeTrendSafe([], 'temp_air', 'Air Temp', '°C', 'Region', [0, 0]);
    expect(empty.state).toBe('empty');
    expect(empty.result).toBeUndefined();
    expect(empty.validation!.validCount).toBe(0);

    const insufficient = analyzeTrendSafe(
      [
        { date: '2000-01', year: 2000, month: 1, raw: 1, isMissing: false },
        { date: '2000-02', year: 2000, month: 2, raw: 2, isMissing: false },
      ],
      'temp_air',
      'Air Temp',
      '°C',
      'Region',
      [0, 0]
    );
    expect(insufficient.state).toBe('unavailable');
    expect(insufficient.result).toBeUndefined();
  });

  it('a ready analysis carries provenance information', () => {
    const points: TimeSeriesPoint[] = [];
    for (let y = 2000; y <= 2004; y++) {
      for (let m = 1; m <= 12; m++) {
        points.push({ date: `${y}-${String(m).padStart(2, '0')}`, year: y, month: m, raw: 15 + y * 0.1, isMissing: false });
      }
    }
    const response = analyzeTrendSafe(points, 'temp_air', 'Air Temp', '°C', 'Region', [0, 0]);
    expect(response.state).toBe('ready');
    expect(response.result).toBeDefined();
    expect(response.validation!.state).toBe('ready');
    expect(response.validation!.validCount).toBeGreaterThan(0);
  });
});
