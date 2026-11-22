import {
  describe,
  expect,
  it,
} from 'vitest';
import {
  generateJsScript,
  generatePythonScript,
  GeeScriptError,
  GeeScriptOptions,
  sanitizeExportName,
} from '../../src/services/gee/scriptGenerator';
import {
  GEE_DATASET_MAP,
  listGeeDatasets,
  resolveGeeDataset,
  hasGeeSupport,
  type GeeDatasetConfig,
} from '../../src/services/gee/datasetMap';
import type { VariableId } from '../../../src/types/earthScience';

const sylhet: GeeScriptOptions['geometry'] = {
  kind: 'point',
  lat: 24.8948,
  lon: 91.8687,
};

const baseOptions: GeeScriptOptions = {
  geometry: sylhet,
  startDate: '2024-01-01',
  endDate: '2024-12-31',
  variableId: 'ndvi' as VariableId,
  datasetIndex: 0,
  temporalAggregation: 'none',
  spatialReducer: 'mean',
  exportTarget: 'chart',
};

function genJs(options: Partial<GeeScriptOptions> = {}): string {
  return generateJsScript({ ...baseOptions, ...options });
}

function genPy(options: Partial<GeeScriptOptions> = {}): string {
  return generatePythonScript({ ...baseOptions, ...options });
}

/* -------------------------------------------------------------------------- */
/* datasetMap field completeness                                              */
/* -------------------------------------------------------------------------- */

describe('datasetMap completeness', () => {
  it('covers every VariableId (empty array = no GEE equivalent)', () => {
    const allIds: VariableId[] = [
      'temp_air', 'precip', 'ndvi', 'soil_moist', 'evapo', 'aod', 'lst_day',
      'lst_night', 'ozone', 'pressure', 'humidity', 'specific_humidity',
      'wind_speed', 'radiation_sw', 'radiation_lw', 'albedo', 'snow_depth',
      'snow_precip', 'ozone_total', 'soil_moist_top', 'soil_moist_root',
      'soil_moist_prof', 'sea_level', 'sst', 'streamflow', 'evi',
    ];
    for (const id of allIds) {
      expect(listGeeDatasets(id)).toBeInstanceOf(Array);
    }
  });

  it('every configured entry has all required fields', () => {
    const required: (keyof GeeDatasetConfig)[] = [
      'variableId', 'label', 'eeCollectionId', 'band', 'scale', 'offset',
      'units', 'nativeScaleMeters', 'reducer', 'notes',
    ];
    for (const configs of Object.values(GEE_DATASET_MAP)) {
      for (const cfg of configs) {
        for (const field of required) {
          expect(cfg[field]).toBeDefined();
        }
        expect(typeof cfg.scale).toBe('number');
        expect(typeof cfg.offset).toBe('number');
        expect(typeof cfg.nativeScaleMeters).toBe('number');
        expect(cfg.nativeScaleMeters).toBeGreaterThan(0);
        expect(['mean', 'median', 'first', 'max', 'sum']).toContain(
          cfg.reducer
        );
      }
    }
  });

  it('resolves known datasets and rejects unknown variables', () => {
    expect(resolveGeeDataset('ndvi' as VariableId, 0)?.eeCollectionId).toBe(
      'MODIS/061/MOD13Q1'
    );
    expect(resolveGeeDataset('precip' as VariableId, 0)?.eeCollectionId).toBe(
      'UCSB-CHG/CHIRPS/DAILY'
    );
    expect(resolveGeeDataset('lst_night' as VariableId, 0)?.eeCollectionId).toBe(
      'MODIS/061/MOD11A1'
    );
    expect(resolveGeeDataset('temp_air' as VariableId, 0)?.eeCollectionId).toBe(
      'ECMWF/ERA5_LAND/DAILY_AGGR'
    );
    expect(resolveGeeDataset('soil_moist' as VariableId)).toBeUndefined();
    expect(hasGeeSupport('soil_moist' as VariableId)).toBe(false);
    expect(hasGeeSupport('ndvi' as VariableId)).toBe(true);
  });
});

/* -------------------------------------------------------------------------- */
/* JS script snapshots                                                        */
/* -------------------------------------------------------------------------- */

describe('generateJsScript', () => {
  it('MODIS NDVI chart snapshot', () => {
    expect(genJs({ variableId: 'ndvi' as VariableId, datasetIndex: 0 })).toMatchSnapshot();
  });

  it('MODIS EVI snapshot', () => {
    expect(genJs({ variableId: 'evi' as VariableId, datasetIndex: 0 })).toMatchSnapshot();
  });

  it('CHIRPS precip snapshot', () => {
    expect(genJs({ variableId: 'precip' as VariableId, datasetIndex: 0 })).toMatchSnapshot();
  });

  it('IMERG precip snapshot', () => {
    expect(genJs({ variableId: 'precip' as VariableId, datasetIndex: 1 })).toMatchSnapshot();
  });

  it('MODIS LST night snapshot', () => {
    expect(genJs({ variableId: 'lst_night' as VariableId, datasetIndex: 0 })).toMatchSnapshot();
  });

  it('ERA5-Land temp_air snapshot', () => {
    expect(genJs({ variableId: 'temp_air' as VariableId, datasetIndex: 0 })).toMatchSnapshot();
  });

  it('Sentinel-2 NDVI (computed expression) snapshot', () => {
    expect(genJs({ variableId: 'ndvi' as VariableId, datasetIndex: 1 })).toMatchSnapshot();
  });

  it('monthly aggregation + CSV to Drive snapshot', () => {
    expect(
      genJs({
        variableId: 'precip' as VariableId,
        datasetIndex: 0,
        temporalAggregation: 'monthly',
        exportTarget: 'csv-drive',
      })
    ).toMatchSnapshot();
  });

  it('annual aggregation + GeoTIFF to Drive snapshot', () => {
    expect(
      genJs({
        variableId: 'lst_day' as VariableId,
        temporalAggregation: 'annual',
        exportTarget: 'geotiff-drive',
      })
    ).toMatchSnapshot();
  });

  it('area-mean bbox snapshot', () => {
    expect(
      genJs({
        variableId: 'ndvi' as VariableId,
        geometry: {
          kind: 'bbox',
          west: 91.8,
          south: 24.8,
          east: 91.9,
          north: 25.0,
        },
      })
    ).toMatchSnapshot();
  });
});

/* -------------------------------------------------------------------------- */
/* Python script snapshots                                                    */
/* -------------------------------------------------------------------------- */

describe('generatePythonScript', () => {
  it('MODIS NDVI chart snapshot', () => {
    expect(genPy({ variableId: 'ndvi' as VariableId, datasetIndex: 0 })).toMatchSnapshot();
  });

  it('IMERG precip snapshot', () => {
    expect(genPy({ variableId: 'precip' as VariableId, datasetIndex: 1 })).toMatchSnapshot();
  });

  it('ERA5-Land temp_air snapshot', () => {
    expect(genPy({ variableId: 'temp_air' as VariableId, datasetIndex: 0 })).toMatchSnapshot();
  });

  it('CSV to Drive snapshot', () => {
    expect(
      genPy({
        variableId: 'precip' as VariableId,
        datasetIndex: 0,
        exportTarget: 'csv-drive',
      })
    ).toMatchSnapshot();
  });
});

/* -------------------------------------------------------------------------- */
/* Validation                                                                 */
/* -------------------------------------------------------------------------- */

describe('validation', () => {
  it('rejects reversed dates', () => {
    expect(() =>
      genJs({ startDate: '2024-12-01', endDate: '2024-01-01' })
    ).toThrow(GeeScriptError);
  });

  it('rejects invalid date format', () => {
    expect(() => genJs({ startDate: 'not-a-date' })).toThrow(GeeScriptError);
  });

  it('rejects out-of-range coordinates', () => {
    expect(() =>
      genJs({ geometry: { kind: 'point', lat: 91, lon: 0 } as any })
    ).toThrow(GeeScriptError);
    expect(() =>
      genJs({ geometry: { kind: 'point', lat: 0, lon: 181 } as any })
    ).toThrow(GeeScriptError);
  });

  it('rejects unsupported variable', () => {
    expect(() =>
      genJs({ variableId: 'soil_moist' as VariableId })
    ).toThrow(GeeScriptError);
  });

  it('rejects empty polygon ring', () => {
    expect(() =>
      genJs({
        geometry: {
          kind: 'polygon',
          rings: [[[91.8, 24.8], [91.9, 24.8], [91.9, 25.0]]],
        } as any,
      })
    ).toThrow(GeeScriptError);
  });

  it('sanitizes hostile export names', () => {
    expect(sanitizeExportName('../../etc/passwd')).toBe('etc_passwd');
    expect(sanitizeExportName("foo'; DROP TABLE--")).toBe('foo_DROP_TABLE--');
    expect(sanitizeExportName('<script>alert(1)</script>')).toBe(
      'script_alert_1_script'
    );
    expect(sanitizeExportName('')).toBe('heraoculus_export');
  });

  it('generated JS embeds dataset id, band, scale, and geometry', () => {
    const js = genJs({ variableId: 'ndvi' as VariableId, datasetIndex: 0 });
    expect(js).toContain("MODIS/061/MOD13Q1");
    expect(js).toContain("'NDVI'");
    expect(js).toContain('0.0001');
    expect(js).toContain('ee.Point([91.8687, 24.8948])');
    expect(js).toContain('filterDate(\'2024-01-01\', \'2024-12-31\')');
  });

  it('generated Python embeds dataset id, band, scale, and ee.Initialize hint', () => {
    const py = genPy({ variableId: 'lst_night' as VariableId, datasetIndex: 0 });
    expect(py).toContain("MODIS/061/MOD11A1");
    expect(py).toContain('"LST_Night_1km"');
    expect(py).toContain('0.02');
    expect(py).toContain("ee.Initialize(project='YOUR_GEE_PROJECT_ID')");
    expect(py).toContain('ee.Point([91.8687, 24.8948])');
  });

  it('does not interpolate hostile region names into code', () => {
    // Coordinates are numbers; hostile strings should be rejected, not embedded.
    expect(() =>
      genJs({ geometry: { kind: 'point', lat: NaN, lon: 0 } as any })
    ).toThrow(GeeScriptError);
    const safe = genJs();
    expect(safe).not.toContain('<script>');
    expect(safe).not.toContain('DROP TABLE');
  });
});

/* -------------------------------------------------------------------------- */
/* Temporal aggregation structure                                             */
/* -------------------------------------------------------------------------- */

describe('temporal aggregation structure', () => {
  function jsAggBlock(js: string): string {
    const start = js.indexOf('// 3. Apply temporal aggregation.');
    const end = js.indexOf('// 4. Export / chart.');
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    return js.slice(start, end);
  }

  function pyAggBlock(py: string): string {
    const start = py.indexOf('# 4. Apply temporal aggregation.');
    const end = py.indexOf('# 5. Export / chart');
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    return py.slice(start, end);
  }

  it('JS annual aggregation declares `years` before using it', () => {
    const js = genJs({ temporalAggregation: 'annual', exportTarget: 'chart' });
    const block = jsAggBlock(js);
    expect(block).toContain("var years = ee.List.sequence(2024, 2024);");
    expect(block).toContain('years.map(function(y) {');
    expect(block).toContain("calendarRange(y, y, 'year')");
    expect(block).toContain("ee.Date.fromYMD(y, 1, 1)");
    // `years` must be declared before it is referenced (no undefined reference).
    const decl = block.indexOf('var years = ee.List.sequence');
    const use = block.indexOf('years.map');
    expect(decl).toBeGreaterThanOrEqual(0);
    expect(use).toBeGreaterThan(decl);
  });

  it('JS monthly aggregation declares `years` and `months`', () => {
    const js = genJs({ temporalAggregation: 'monthly', exportTarget: 'chart' });
    const block = jsAggBlock(js);
    expect(block).toContain("var years = ee.List.sequence(2024, 2024);");
    expect(block).toContain("var months = ee.List.sequence(1, 12);");
    expect(block).toContain("calendarRange(y, y, 'year')");
    expect(block).toContain("calendarRange(m, m, 'month')");
    expect(block).toContain("ee.Date.fromYMD(y, m, 1)");
    expect(block).toContain('.flatten()');
  });

  it('different start and end years produce a year sequence', () => {
    const js = genJs({
      temporalAggregation: 'annual',
      startDate: '2020-03-01',
      endDate: '2024-11-30',
      exportTarget: 'chart',
    });
    const block = jsAggBlock(js);
    expect(block).toContain("var years = ee.List.sequence(2020, 2024);");
    expect(js).toContain("filterDate('2020-03-01', '2024-11-30')");
  });

  it('leap-year boundary date normalizes and is covered by annual aggregation', () => {
    // 2024 is a leap year; 2023 is not.
    expect(() =>
      genJs({ startDate: '2024-02-29', endDate: '2024-02-29' })
    ).not.toThrow();
    expect(() =>
      genJs({ startDate: '2023-02-29', endDate: '2023-02-29' })
    ).toThrow(GeeScriptError);
    const js = genJs({
      temporalAggregation: 'annual',
      startDate: '2024-01-01',
      endDate: '2024-12-31',
    });
    expect(js).toContain("var years = ee.List.sequence(2024, 2024);");
  });

  it('single-day range (start == end) is valid', () => {
    const js = genJs({
      temporalAggregation: 'annual',
      startDate: '2024-06-15',
      endDate: '2024-06-15',
    });
    expect(js).toContain("filterDate('2024-06-15', '2024-06-15')");
    expect(js).toContain("var years = ee.List.sequence(2024, 2024);");
  });

  it('reversed date range (truly empty) is rejected', () => {
    expect(() =>
      genJs({ startDate: '2024-12-31', endDate: '2024-01-01' })
    ).toThrow(GeeScriptError);
  });

  it('generated script references the configured band', () => {
    const js = genJs({
      variableId: 'ndvi' as VariableId,
      datasetIndex: 0,
      temporalAggregation: 'annual',
    });
    expect(js).toContain("'NDVI'");
    const py = genPy({
      variableId: 'ndvi' as VariableId,
      datasetIndex: 0,
      temporalAggregation: 'annual',
    });
    expect(py).toContain('"NDVI"');
  });

  it('unknown variable (missing band/dataset) is rejected', () => {
    expect(() =>
      genJs({ variableId: 'soil_moist' as VariableId, temporalAggregation: 'annual' })
    ).toThrow(GeeScriptError);
  });

  it('Python annual aggregation declares `years`', () => {
    const py = genPy({ temporalAggregation: 'annual', exportTarget: 'chart' });
    const block = pyAggBlock(py);
    expect(block).toContain('years = ee.List.sequence(2024, 2024)');
    expect(block).toContain('def year_img(y):');
    expect(block).toContain("calendarRange(y, y, 'year')");
    expect(block).toContain("ee.Date.fromYMD(y, 1, 1)");
  });

  it('Python monthly aggregation declares `years` and `months`', () => {
    const py = genPy({ temporalAggregation: 'monthly', exportTarget: 'chart' });
    const block = pyAggBlock(py);
    expect(block).toContain('years = ee.List.sequence(2024, 2024)');
    expect(block).toContain('months = ee.List.sequence(1, 12)');
    expect(block).toContain("calendarRange(m, m, 'month')");
  });

  it('generated JS script has the expected top-level structure', () => {
    const js = genJs({ temporalAggregation: 'annual' });
    expect(js).toContain(
      '// HeraOculus → Google Earth Engine (Code Editor) JavaScript'
    );
    expect(js).toMatch(/var geometry = /);
    expect(js).toMatch(/var collection = ee\.ImageCollection\(/);
    expect(js).toMatch(/aggColl = ee\.ImageCollection\.fromImages\(/);
    expect(js).toMatch(/var chart = /);
    expect(js).toContain('// End of HeraOculus GEE script.');
    const collIdx = js.indexOf('var collection =');
    const aggIdx = js.indexOf('aggColl =');
    const chartIdx = js.indexOf('var chart =');
    expect(collIdx).toBeGreaterThanOrEqual(0);
    expect(aggIdx).toBeGreaterThan(collIdx);
    expect(chartIdx).toBeGreaterThan(aggIdx);
  });
});
