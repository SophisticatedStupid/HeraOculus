/**
 * Explorer workflow integration tests.
 *
 * Chains the pure pieces of the live-data workflow without network calls:
 *   region + variable → connector selection → validation → trend analysis.
 *
 * Validates that the Explorer gracefully handles:
 * - globally covered variables at any location,
 * - regionally covered variables inside their service area,
 * - regionally covered variables outside their service area (deterministic fallback,
 *   no masquerading live source),
 * - empty / insufficient data (analyzeTrendSafe non-ready states).
 */
import {
  connectorRegistry,
  type ConnectorSelection,
  type VariableId,
} from '../src/services/connectors';
import { analyzeTrendSafe, validateTimeSeriesForAnalysis } from '../src/services/statistics';
import type { TimeSeriesPoint } from '../src/types/earthScience';
import { describe, expect, it } from 'vitest';

function monthlyPoints(
  startYear: number,
  count: number,
  options: { trend?: number; base?: number } = {}
): TimeSeriesPoint[] {
  const trend = options.trend ?? 0;
  const base = options.base ?? 15;
  const points: TimeSeriesPoint[] = [];
  for (let i = 0; i < count; i++) {
    const month = (i % 12) + 1;
    const year = startYear + Math.floor(i / 12);
    points.push({
      date: `${year}-${String(month).padStart(2, '0')}`,
      year,
      month,
      raw: base + trend * (i / 12),
      isMissing: false,
    });
  }
  return points;
}

function runWorkflow(variableId: VariableId, lat: number, lon: number): {
  selection: ConnectorSelection;
  analysis: ReturnType<typeof analyzeTrendSafe>;
} {
  const selection = connectorRegistry.getBestConnector(variableId, lat, lon);
  const points = monthlyPoints(2000, 60, { trend: 0.5 });
  const analysis = analyzeTrendSafe(points, variableId, 'Variable', '°C', 'Region', [lat, lon]);
  return { selection, analysis };
}

describe('Explorer live-data workflow', () => {
  it('selects a global connector and produces a ready analysis for any location', () => {
    const locations: Array<[number, number]> = [
      [82.0, -50.0], // Arctic
      [5.0, 75.0], // Tropical
      [0.0, -120.0], // Equatorial ocean
      [-33.0, -70.0], // Southern Hemisphere
    ];
    for (const [lat, lon] of locations) {
      const { selection, analysis } = runWorkflow('temp_air', lat, lon);
      expect(selection.fallback).toBe(false);
      expect(selection.connector!.id).toBe('nasa-power');
      expect(analysis.state).toBe('ready');
      expect(analysis.result!.variableId).toBe('temp_air');
    }
  });

  it('selects a regional connector inside its service area and analyzes', () => {
    const { selection, analysis } = runWorkflow('streamflow', 38.9, -77.1);
    expect(selection.fallback).toBe(false);
    expect(selection.connector!.id).toBe('usgs-water');
    expect(analysis.state).toBe('ready');
  });

  it('selects NOAA for US coastal sea level and analyzes', () => {
    const { selection, analysis } = runWorkflow('sea_level', 40.7, -74.0);
    expect(selection.fallback).toBe(false);
    expect(selection.connector!.id).toBe('noaa-tides');
    expect(analysis.state).toBe('ready');
  });

  it('falls back (no live connector) for a regional variable outside coverage', () => {
    const selection = connectorRegistry.getBestConnector('streamflow', 82.0, -50.0);
    expect(selection.fallback).toBe(true);
    expect(selection.connector).toBeUndefined();
    // The workflow must not present a regional connector as a live source.
    expect(selection.reason).toContain('no connector covers');
  });

  it('falls back for sea level outside NOAA coverage', () => {
    const selection = connectorRegistry.getBestConnector('sea_level', 0.0, -120.0);
    expect(selection.fallback).toBe(true);
    expect(selection.connector).toBeUndefined();
  });

  it('reports empty state when the workflow has no data', () => {
    const analysis = analyzeTrendSafe([], 'temp_air', 'Variable', '°C', 'Region', [0, 0]);
    expect(analysis.state).toBe('empty');
    expect(analysis.result).toBeUndefined();
  });

  it('reports unavailable state for insufficient data', () => {
    const analysis = analyzeTrendSafe(
      [
        { date: '2000-01', year: 2000, month: 1, raw: 1, isMissing: false },
        { date: '2000-02', year: 2000, month: 2, raw: 2, isMissing: false },
      ],
      'temp_air',
      'Variable',
      '°C',
      'Region',
      [0, 0]
    );
    expect(analysis.state).toBe('unavailable');
    expect(analysis.result).toBeUndefined();
  });

  it('validation and analysis agree on readiness', () => {
    const points = monthlyPoints(2000, 60);
    const validation = validateTimeSeriesForAnalysis(points, 'temp_air');
    expect(validation.state).toBe('ready');
    const analysis = analyzeTrendSafe(points, 'temp_air', 'Variable', '°C', 'Region', [0, 0]);
    expect(analysis.state).toBe('ready');
  });
});
