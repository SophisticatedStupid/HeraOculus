/**
 * CSV ingestion and empty-data handling tests.
 *
 * Validates parseCustomCsv, detectCsvPrimaryVariable, and extractCsvCoordinates
 * against valid, empty, and malformed CSV inputs.
 */
import {
  detectCsvPrimaryVariable,
  extractCsvCoordinates,
  parseCustomCsv,
} from '../src/services/nasaData';
import type { VariableId } from '../src/types/earthScience';
import { describe, expect, it } from 'vitest';

const validCsv = `date,temp_air
2000-01,1.0
2000-02,2.0
2000-03,3.0
2000-04,4.0
2000-05,5.0
2000-06,6.0`;

const multiVarCsv = `date,temp_air,ndvi
2000-01,1.0,0.5
2000-02,2.0,0.6
2000-03,3.0,0.55
2000-04,4.0,0.58
2000-05,5.0,0.52`;

const coordsCsv = `date,lat,lon,temp_air
2000-01,40.7,-74.0,1.0
2000-02,41.0,-75.0,2.0`;

describe('detectCsvPrimaryVariable', () => {
  it('detects a single clearly named variable column', () => {
    expect(detectCsvPrimaryVariable(validCsv)).toBe('temp_air');
  });

  it('returns null for multi-variable CSVs (ambiguous)', () => {
    expect(detectCsvPrimaryVariable(multiVarCsv)).toBeNull();
  });

  it('returns null for CSV with no recognizable variable column', () => {
    expect(detectCsvPrimaryVariable('date,foo,bar\n2000-01,1,2\n')).toBeNull();
  });

  it('returns null for empty / header-only input', () => {
    expect(detectCsvPrimaryVariable('')).toBeNull();
    expect(detectCsvPrimaryVariable('date,temp_air\n')).toBeNull();
  });
});

describe('extractCsvCoordinates', () => {
  it('extracts lat/lon when lat/lon columns are present', () => {
    expect(extractCsvCoordinates(coordsCsv)).toEqual([40.7, -74.0]);
  });

  it('accepts latitude/longitude and lng/long aliases', () => {
    expect(extractCsvCoordinates(`date,latitude,longitude,temp_air\n2000-01,-33.0,151.0,1.0\n`)).toEqual([
      -33.0,
      151.0,
    ]);
    expect(extractCsvCoordinates(`date,lat,lng,temp_air\n2000-01,0.0,0.0,1.0\n`)).toEqual([0.0, 0.0]);
  });

  it('returns null when no lat/lon columns exist', () => {
    expect(extractCsvCoordinates(validCsv)).toBeNull();
  });

  it('returns null for out-of-range or non-numeric coordinates', () => {
    expect(extractCsvCoordinates(`date,lat,lon,temp_air\n2000-01,91.0,0.0,1.0\n`)).toBeNull();
    expect(extractCsvCoordinates(`date,lat,lon,temp_air\n2000-01,abc,def,1.0\n`)).toBeNull();
  });

  it('returns null for empty input', () => {
    expect(extractCsvCoordinates('')).toBeNull();
  });
});

describe('parseCustomCsv', () => {
  it('parses a valid single-variable CSV into date/value points', () => {
    const points = parseCustomCsv(validCsv, 'temp_air');
    expect(points).toHaveLength(6);
    expect(points[0]).toEqual({ date: '2000-01', value: 1.0 });
    expect(points[1]).toEqual({ date: '2000-02', value: 2.0 });
    expect(points[5]).toEqual({ date: '2000-06', value: 6.0 });
  });

  it('parses a valid multi-variable CSV when targetVariableId is given', () => {
    const points = parseCustomCsv(multiVarCsv, 'ndvi');
    expect(points).toHaveLength(5);
    expect(points[0]).toEqual({ date: '2000-01', value: 0.5 });
    expect(points[1]).toEqual({ date: '2000-02', value: 0.6 });
  });

  it('throws on header-only CSV (no data rows)', () => {
    expect(() => parseCustomCsv('date,temp_air\n', 'temp_air')).toThrow(
      'CSV file must have a header and at least 2 data rows.'
    );
  });

  it('throws on empty input', () => {
    expect(() => parseCustomCsv('', 'temp_air')).toThrow(
      'CSV file must have a header and at least 2 data rows.'
    );
  });
});
