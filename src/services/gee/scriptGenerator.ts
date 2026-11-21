/**
 * HeraOculus → Google Earth Engine script generator.
 *
 * Pure client-side code generation. No GEE credentials, no backend, no network
 * calls. Produces ready-to-run JavaScript (Code Editor) and Python
 * (earthengine-api) scripts from the user's current selection.
 *
 * Every interpolated value is validated/escaping is applied so user-supplied
 * region labels, coordinates and dates cannot break or inject the generated
 * code.
 */

import {
  GeeDatasetConfig,
  GeeExportTarget,
  GeeGeometry,
  GeeReducer,
  GeeScriptOptions,
  GeeTemporalAgg,
  resolveGeeDataset,
} from './datasetMap';
import type { VariableId } from '../../types/earthScience';

/* -------------------------------------------------------------------------- */
/* Validation helpers                                                         */
/* -------------------------------------------------------------------------- */

export class GeeScriptError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GeeScriptError';
  }
}

/** Sanitize a user-facing label into a safe GEE export description/id. */
export function sanitizeExportName(input: string): string {
  return input
    .replace(/[^A-Za-z0-9_-]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 60) || 'heraoculus_export';
}

/** Format a coordinate for safe interpolation into GEE code. */
function formatCoord(n: number): number {
  if (!Number.isFinite(n)) throw new GeeScriptError('Coordinate is not a finite number.');
  return Number(n.toFixed(6));
}

/** Validate and normalize a YYYY-MM-DD date string. */
function normalizeDate(input: string): string {
  if (typeof input !== 'string') throw new GeeScriptError('Date must be a string.');
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(input.trim());
  if (!m) throw new GeeScriptError(`Invalid date '${input}'; expected YYYY-MM-DD.`);
  const y = Number(m[1]), mo = Number(m[2]), d = Number(m[3]);
  const date = new Date(Date.UTC(y, mo - 1, d));
  if (
    date.getUTCFullYear() !== y ||
    date.getUTCMonth() + 1 !== mo ||
    date.getUTCDate() !== d
  ) {
    throw new GeeScriptError(`Invalid calendar date '${input}'.`);
  }
  return input.trim();
}

/** Resolve + validate the full option set. Returns the dataset config. */
export function resolveOptions(options: GeeScriptOptions): {
  config: GeeDatasetConfig;
  normalized: GeeScriptOptions;
} {
  const config = resolveGeeDataset(options.variableId, options.datasetIndex);
  if (!config) {
    throw new GeeScriptError(
      `No GEE equivalent configured for variable '${options.variableId}'. ` +
      'This variable cannot be exported to a GEE script.'
    );
  }
  const startDate = normalizeDate(options.startDate);
  const endDate = normalizeDate(options.endDate);
  if (endDate < startDate) {
    throw new GeeScriptError(
      `end_date (${endDate}) is before start_date (${startDate}).`
    );
  }

  let geometry: GeeGeometry;
  if (options.geometry.kind === 'point') {
    const { lat, lon } = options.geometry;
    if (!Number.isFinite(lat) || lat < -90 || lat > 90) {
      throw new GeeScriptError(`Latitude ${lat} out of range [-90, 90].`);
    }
    if (!Number.isFinite(lon) || lon < -180 || lon > 180) {
      throw new GeeScriptError(`Longitude ${lon} out of range [-180, 180].`);
    }
    geometry = { kind: 'point', lat: formatCoord(lat), lon: formatCoord(lon) };
  } else if (options.geometry.kind === 'bbox') {
    const { west, south, east, north } = options.geometry;
    for (const v of [west, south, east, north]) {
      if (!Number.isFinite(v)) throw new GeeScriptError('BBox coordinate is not finite.');
    }
    if (west < -180 || east > 180 || south < -90 || north > 90) {
      throw new GeeScriptError('BBox out of geographic range.');
    }
    if (west >= east || south >= north) {
      throw new GeeScriptError('BBox min must be less than max.');
    }
    geometry = {
      kind: 'bbox',
      west: formatCoord(west), south: formatCoord(south),
      east: formatCoord(east), north: formatCoord(north),
    };
  } else {
    // polygon
    if (!Array.isArray(options.geometry.rings) || options.geometry.rings.length === 0) {
      throw new GeeScriptError('Polygon must have at least one ring.');
    }
    const rings = options.geometry.rings.map((ring) => {
      if (!Array.isArray(ring) || ring.length < 4) {
        throw new GeeScriptError('Polygon ring needs at least 4 points.');
      }
      return ring.map((pt) => {
        if (!Array.isArray(pt) || pt.length !== 2) {
          throw new GeeScriptError('Polygon point must be [lon, lat].');
        }
        const [lon, lat] = pt as [number, number];
        if (!Number.isFinite(lat) || lat < -90 || lat > 90) {
          throw new GeeScriptError(`Polygon latitude ${lat} out of range [-90, 90].`);
        }
        if (!Number.isFinite(lon) || lon < -180 || lon > 180) {
          throw new GeeScriptError(`Polygon longitude ${lon} out of range [-180, 180].`);
        }
        return [formatCoord(lon), formatCoord(lat)] as [number, number];
      });
    });
    geometry = { kind: 'polygon', rings };
  }

  const temporalAggregation =
    options.temporalAggregation ?? 'none';
  const spatialReducer =
    options.spatialReducer ?? config.reducer;
  const exportTarget = options.exportTarget ?? 'chart';

  return {
    config,
    normalized: {
      ...options,
      geometry,
      startDate,
      endDate,
      datasetIndex: options.datasetIndex ?? 0,
      temporalAggregation,
      spatialReducer,
      exportTarget,
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Geometry expressions                                                       */
/* -------------------------------------------------------------------------- */

function jsGeometry(geom: GeeGeometry): string {
  if (geom.kind === 'point') {
    return `ee.Point([${geom.lon}, ${geom.lat}])`;
  }
  if (geom.kind === 'bbox') {
    return `ee.Rectangle([${geom.west}, ${geom.south}, ${geom.east}, ${geom.north}])`;
  }
  const rings = geom.rings
    .map((ring) =>
      `[${ring.map(([lon, lat]) => `${lon}, ${lat}`).join(', ')}]`
    )
    .join(', ');
  return `ee.Polygon([${rings}])`;
}

function pyGeometry(geom: GeeGeometry): string {
  // Python API uses the same [lon, lat] ordering.
  return jsGeometry(geom);
}

/* -------------------------------------------------------------------------- */
/* ImageCollection build (scale, offset, mask, temporal aggregation)          */
/* -------------------------------------------------------------------------- */

function buildMapFunction(config: GeeDatasetConfig, lang: 'js' | 'py'): string {
  const scale = config.scale;
  const offset = config.offset;
  const band = config.band;
  const bands = config.expressionBands
    ? Object.values(config.expressionBands)
    : [band];

  const selectJs = bands.map((b) => `'${b}'`).join(', ');
  const selectPy = bands.map((b) => `"${b}"`).join(', ');

  let scaleBlockJs: string;
  let scaleBlockPy: string;

  if (config.expression) {
    // Computed index (e.g. S2 NDVI from B8/B4). Scale cancels; bands used raw.
    const args = Object.entries(config.expressionBands ?? {})
      .map(([v, b]) => `${v}: img.select('${b}')`)
      .join(', ');
    scaleBlockJs = `
    var scaled = img.expression(${JSON.stringify(config.expression)}, {${args}});`;
    scaleBlockPy = `
    band_img = {${Object.entries(config.expressionBands ?? {})
      .map(([v, b]) => `'${v}': img.select("${b}")`)
      .join(', ')}}
    scaled = img.expression(${JSON.stringify(config.expression)}, band_img)`;
  } else {
    scaleBlockJs = `
    var scaled = img.select('${band}').multiply(${scale}).add(${offset});`;
    scaleBlockPy = `
    scaled = img.select("${band}").multiply(${scale}).add(${offset})`;
  }

  let maskBlockJs = '';
  let maskBlockPy = '';
  if (config.cloudMask) {
    maskBlockJs = `
    var qa = img.select('QA60');
    var cloudBitMask = 1 << 10;
    var cloudMask = qa.bitwiseAnd(cloudBitMask).eq(0);
    scaled = scaled.updateMask(cloudMask);`;
    maskBlockPy = `
    qa = img.select('QA60')
    cloud_bit_mask = 1 << 10
    cloud_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0)
    scaled = scaled.updateMask(cloud_mask)`;
  }

  let validMinBlockJs = '';
  let validMinBlockPy = '';
  if (config.validMin !== undefined) {
    validMinBlockJs = `
    scaled = scaled.updateMask(img.gte(${config.validMin}));`;
    validMinBlockPy = `
    scaled = scaled.updateMask(img.gte(${config.validMin}))`;
  }

  if (lang === 'js') {
    return `
  .map(function(img) {
    ${scaleBlockJs}
    ${maskBlockJs}
    ${validMinBlockJs}
    return scaled.rename('value').copyProperties(img, ['system:time_start']);
  })`;
  }
  return `
def scale_and_mask(img):
    ${scaleBlockPy}
    ${maskBlockPy}
    ${validMinBlockPy}
    return scaled.rename('value').copyProperties(img, ['system:time_start'])`;
}

function buildTemporalAggregation(
  collExpr: string,
  temporalAggregation: GeeTemporalAgg,
  startDate: string,
  endDate: string,
  lang: 'js' | 'py'
): string {
  if (temporalAggregation === 'none') return collExpr;
  const [sy, sm] = startDate.split('-').map(Number);
  const [ey, em] = endDate.split('-').map(Number);

  if (lang === 'js') {
    if (temporalAggregation === 'monthly') {
      return `
  // Aggregate to monthly means.
  var years = ee.List.sequence(${sy}, ${ey});
  var months = ee.List.sequence(1, 12);
  aggColl = ee.ImageCollection.fromImages(
    years.map(function(y) {
      return months.map(function(m) {
        var c = ${collExpr}
          .filter(ee.Filter.calendarRange(y, y, 'year'))
          .filter(ee.Filter.calendarRange(m, m, 'month'));
        return c.mean().set('system:time_start', ee.Date.fromYMD(y, m, 1));
      });
    }).flatten()
  );`;
    }
    // annual
    return `
  // Aggregate to annual means.
  aggColl = ee.ImageCollection.fromImages(
    years.map(function(y) {
      var c = ${collExpr}
        .filter(ee.Filter.calendarRange(y, y, 'year'));
      return c.mean().set('system:time_start', ee.Date.fromYMD(y, 1, 1));
    })
  );`;
  }

    if (lang === 'py') {
    if (temporalAggregation === 'monthly') {
      return `
# Aggregate to monthly means.
years = ee.List.sequence(${sy}, ${ey})
months = ee.List.sequence(1, 12)
def month_img(y, m):
    c = (${collExpr}
         .filter(ee.Filter.calendarRange(y, y, 'year'))
         .filter(ee.Filter.calendarRange(m, m, 'month')))
    return c.mean().set('system:time_start', ee.Date.fromYMD(y, m, 1))
aggColl = ee.ImageCollection.fromImages(
    years.map(lambda y: months.map(lambda m: month_img(y, m))).flatten()
)`;
    }
    return `
# Aggregate to annual means.
years = ee.List.sequence(${sy}, ${ey})
def year_img(y):
    c = ${collExpr}.filter(ee.Filter.calendarRange(y, y, 'year'))
    return c.mean().set('system:time_start', ee.Date.fromYMD(y, 1, 1))
aggColl = ee.ImageCollection.fromImages(years.map(year_img))`;
  }
  return collExpr;
}

/** Band names to select: primary band(s) plus QA60 when cloud masking is on. */
export function bandListFor(config: GeeDatasetConfig): string[] {
  const bands = config.expressionBands
    ? Object.values(config.expressionBands)
    : [config.band];
  if (config.cloudMask && !bands.includes('QA60')) bands.push('QA60');
  return bands;
}

/* -------------------------------------------------------------------------- */
/* Script generation                                                          */
/* -------------------------------------------------------------------------- */

function header(
  lang: 'js' | 'py',
  config: GeeDatasetConfig,
  opts: GeeScriptOptions,
  geomText: string
): string[] {
  const lines: string[] = [];
  lines.push(lang === 'js'
    ? '// HeraOculus → Google Earth Engine (Code Editor) JavaScript'
    : '# HeraOculus → Google Earth Engine (Python / earthengine-api)');
  lines.push(lang === 'js'
    ? '// Generated automatically. Run this in your own GEE Code Editor.'
    : '# Generated automatically. Run this in your own Python environment.');
  lines.push(lang === 'js'
    ? '// Geo-Iris/HeraOculus does not execute, store, or transmit this code.'
    : '# HeraOculus does not execute, store, or transmit this code.');
  lines.push(lang === 'js'
    ? `// Dataset: ${config.label}`
    : `# Dataset: ${config.label}`);
  lines.push(lang === 'js'
    ? `// eeCollectionId: ${config.eeCollectionId} | band: ${config.band} | scale: ${config.scale} | units: ${config.units}`
    : `# eeCollectionId: ${config.eeCollectionId} | band: ${config.band} | scale: ${config.scale} | units: ${config.units}`);
  lines.push(lang === 'js'
    ? `// Native resolution: ${config.nativeScaleMeters} m | reducer: ${config.reducer}`
    : `# Native resolution: ${config.nativeScaleMeters} m | reducer: ${opts.spatialReducer}`);
  lines.push(lang === 'js'
    ? `// Region: ${JSON.stringify(opts.geometry)}`
    : `# Region: ${JSON.stringify(opts.geometry)}`);
  lines.push(lang === 'js'
    ? `// Date range: ${opts.startDate} to ${opts.endDate}`
    : `# Date range: ${opts.startDate} to ${opts.endDate}`);
  lines.push(lang === 'js'
    ? `// Variable: ${opts.variableId} | export target: ${opts.exportTarget}`
    : `# Variable: ${opts.variableId} | export target: ${opts.exportTarget}`);
  lines.push(lang === 'js'
    ? '// Geometry uses [lon, lat] ordering as required by Earth Engine.'
    : '# Geometry uses [lon, lat] ordering as required by Earth Engine.');
  lines.push('');
  if (config.notes) {
    const noteLines = config.notes.split('. ').filter(Boolean);
    noteLines.forEach((n) => {
      lines.push(lang === 'js' ? `// ${n.trim()}.` : `# ${n.trim()}.`);
    });
    lines.push('');
  }
  if (config.coverageNote) {
    lines.push(
      lang === 'js'
        ? `// Coverage: ${config.coverageNote}`
        : `# Coverage: ${config.coverageNote}`
    );
    lines.push('');
  }
  if (config.deprecated) {
    lines.push(
      lang === 'js'
        ? '// WARNING: this dataset is deprecated in the GEE catalog.'
        : '# WARNING: this dataset is deprecated in the GEE catalog.'
    );
    lines.push('');
  }
  return lines;
}

export function generateJsScript(options: GeeScriptOptions): string {
  const { config, normalized: opts } = resolveOptions(options);
  const geom = jsGeometry(opts.geometry);
  const lines: string[] = [];

  lines.push(...header('js', config, opts, geom));
  lines.push('// 1. Define the geometry from the HeraOculus selection.');
  lines.push(`var geometry = ${geom};`);
  lines.push('');

  lines.push('// 2. Load the collection, filter by date, select band(s), apply scale & masks.');
  const bands = bandListFor(config);
  const bandList = bands.map((b) => `'${b}'`).join(', ');
  lines.push(
    `var collection = ee.ImageCollection('${config.eeCollectionId}')`
  );
  lines.push(`    .filterDate('${opts.startDate}', '${opts.endDate}')`);
  lines.push(`    .select([${bandList}])`);
  lines.push(buildMapFunction(config, 'js'));
  lines.push('');

  lines.push('// 3. Apply temporal aggregation.');
  if (opts.temporalAggregation === 'none') {
    lines.push('var aggColl = collection;');
  } else {
    lines.push(buildTemporalAggregation(
      'collection',
      opts.temporalAggregation,
      opts.startDate,
      opts.endDate,
      'js'
    ).replace(/^  /gm, ''));
  }
  lines.push('');

  const reducerName = opts.spatialReducer;
  const reducerCall =
    reducerName === 'median' ? 'ee.Reducer.median()'
    : reducerName === 'first' ? 'ee.Reducer.first()'
    : reducerName === 'max' ? 'ee.Reducer.max()'
    : reducerName === 'sum' ? 'ee.Reducer.sum()'
    : 'ee.Reducer.mean()';

  const scale = config.nativeScaleMeters;

  lines.push('// 4. Export / chart.');
  if (opts.exportTarget === 'chart') {
    lines.push(
      `var chart = ui.Chart.image.seriesByRegion({`
    );
    lines.push(`  imageCollection: aggColl,`);
    lines.push(`  regions: geometry,`);
    lines.push(`  reducer: ${reducerCall},`);
    lines.push(`  scale: ${scale},`);
    lines.push(`  xProperty: 'system:time_start'`);
    lines.push(`});`);
    lines.push(`print(chart);`);
    lines.push(
      `// Y-axis units: ${config.units}. Values already have scale ${config.scale} applied.`
    );
  } else if (opts.exportTarget === 'csv-drive') {
    const desc = sanitizeExportName(`HeraOculus_${config.variableId}_${opts.startDate}_${opts.endDate}`);
    lines.push(`var features = aggColl.map(function(img) {`);
    lines.push(`  var val = img.reduceRegion({`);
    lines.push(`    reducer: ${reducerCall},`);
    lines.push(`    geometry: geometry,`);
    lines.push(`    scale: ${scale},`);
    lines.push(`    maxPixels: 1e9,`);
    lines.push(`    bestEffort: true`);
    lines.push(`  });`);
    lines.push(`  return ee.Feature(null, {`);
    lines.push(`    date: img.date().format('YYYY-MM-dd'),`);
    lines.push(`    value: val.get('value')`);
    lines.push(`  });`);
    lines.push(`});`);
    lines.push(`var fc = ee.FeatureCollection(features);`);
    lines.push(`Export.table.toDrive({`);
    lines.push(`  collection: fc,`);
    lines.push(`  description: '${desc}',`);
    lines.push(`  fileFormat: 'CSV'`);
    lines.push(`});`);
  } else {
    // geotiff-drive
    const desc = sanitizeExportName(`HeraOculus_${config.variableId}_${opts.startDate}_${opts.endDate}`);
    lines.push(`var image = aggColl.mean().clip(geometry);`);
    lines.push(`Export.image.toDrive({`);
    lines.push(`  image: image,`);
    lines.push(`  description: '${desc}',`);
    lines.push(`  scale: ${scale},`);
    lines.push(`  region: geometry,`);
    lines.push(`  maxPixels: 1e9`);
    lines.push(`});`);
  }
  lines.push('');
  lines.push('// End of HeraOculus GEE script.');

  return lines.join('\n');
}

export function generatePythonScript(options: GeeScriptOptions): string {
  const { config, normalized: opts } = resolveOptions(options);
  const geom = pyGeometry(opts.geometry);
  const lines: string[] = [];

  lines.push(...header('py', config, opts, geom));
  lines.push('# 1. Authenticate in your own environment before running:');
  lines.push('#   ee.Authenticate()');
  lines.push("#   ee.Initialize(project='YOUR_GEE_PROJECT_ID')");
  lines.push('# 2. Define the geometry from the HeraOculus selection.');
  lines.push(`geometry = ${geom}`);
  lines.push('');

  lines.push('# 3. Load the collection, filter by date, select band(s), apply scale & masks.');
  const bands = bandListFor(config);
  const bandList = bands.map((b) => `"${b}"`).join(', ');
  // Define the scale/mask helper first, then build the collection with parens so
  // Python allows implicit line continuation across the chained calls.
  lines.push(buildMapFunction(config, 'py'));
  lines.push('');
  lines.push(
    `collection = (ee.ImageCollection('${config.eeCollectionId}')`
  );
  lines.push(`    .filterDate('${opts.startDate}', '${opts.endDate}')`);
  lines.push(`    .select([${bandList}])`);
  lines.push(`    .map(scale_and_mask)`);
  lines.push(')');
  lines.push('');

  lines.push('# 4. Apply temporal aggregation.');
  if (opts.temporalAggregation === 'none') {
    lines.push('aggColl = collection');
  } else {
    lines.push(
      buildTemporalAggregation(
        'collection',
        opts.temporalAggregation,
        opts.startDate,
        opts.endDate,
        'py'
      )
    );
  }
  lines.push('');

  const reducerName = opts.spatialReducer;
  const reducerCall =
    reducerName === 'median' ? 'ee.Reducer.median()'
    : reducerName === 'first' ? 'ee.Reducer.first()'
    : reducerName === 'max' ? 'ee.Reducer.max()'
    : reducerName === 'sum' ? 'ee.Reducer.sum()'
    : 'ee.Reducer.mean()';
  const scale = config.nativeScaleMeters;

  lines.push('# 5. Export / chart (earthengine-api has no ui.Chart; export CSV/TIFF).');
  if (opts.exportTarget === 'chart') {
    // Python: reduce each image, build FeatureCollection, export CSV for charting elsewhere.
    const desc = sanitizeExportName(`HeraOculus_${config.variableId}_${opts.startDate}_${opts.endDate}`);
    lines.push('def reduce_image(img):');
    lines.push(`    val = img.reduceRegion(`);
    lines.push(`        reducer=${reducerCall},`);
    lines.push(`        geometry=geometry,`);
    lines.push(`        scale=${scale},`);
    lines.push(`        maxPixels=1e9,`);
    lines.push(`        bestEffort=True`);
    lines.push(`    )`);
    lines.push(`    return ee.Feature(None, {`);
    lines.push(`        'date': img.date().format('YYYY-MM-dd'),`);
    lines.push(`        'value': val.get('value')`);
    lines.push(`    })`);
    lines.push(`features = aggColl.map(reduce_image)`);
    lines.push(`fc = ee.FeatureCollection(features)`);
    lines.push(`Export.table.toDrive(`);
    lines.push(`    collection=fc,`);
    lines.push(`    description='${desc}',`);
    lines.push(`    fileFormat='CSV'`);
    lines.push(`)`);
    lines.push(
      `# Note: open the exported CSV to chart; units are ${config.units} (scale ${config.scale} applied).`
    );
  } else if (opts.exportTarget === 'csv-drive') {
    const desc = sanitizeExportName(`HeraOculus_${config.variableId}_${opts.startDate}_${opts.endDate}`);
    lines.push('def reduce_image(img):');
    lines.push(`    val = img.reduceRegion(`);
    lines.push(`        reducer=${reducerCall},`);
    lines.push(`        geometry=geometry,`);
    lines.push(`        scale=${scale},`);
    lines.push(`        maxPixels=1e9,`);
    lines.push(`        bestEffort=True`);
    lines.push(`    )`);
    lines.push(`    return ee.Feature(None, {`);
    lines.push(`        'date': img.date().format('YYYY-MM-dd'),`);
    lines.push(`        'value': val.get('value')`);
    lines.push(`    })`);
    lines.push(`features = aggColl.map(reduce_image)`);
    lines.push(`fc = ee.FeatureCollection(features)`);
    lines.push(`Export.table.toDrive(`);
    lines.push(`    collection=fc,`);
    lines.push(`    description='${desc}',`);
    lines.push(`    fileFormat='CSV'`);
    lines.push(`)`);
  } else {
    const desc = sanitizeExportName(`HeraOculus_${config.variableId}_${opts.startDate}_${opts.endDate}`);
    lines.push(`image = aggColl.mean().clip(geometry)`);
    lines.push(`Export.image.toDrive(`);
    lines.push(`    image=image,`);
    lines.push(`    description='${desc}',`);
    lines.push(`    scale=${scale},`);
    lines.push(`    region=geometry,`);
    lines.push(`    maxPixels=1e9`);
    lines.push(`)`);
  }
  lines.push('');
  lines.push('# End of HeraOculus GEE script.');

  return lines.join('\n');
}
