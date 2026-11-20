/**
 * Regression tests for empty/invalid/insufficient time series handling
 * Validates that no empty dataset can crash the app, no unavailable feed is
 * presented as valid science, and valid workflows remain intact.
 */

import {
  validateTimeSeriesForAnalysis,
  validateTimeSeriesForTheilSen,
  validateTimeSeriesForOLS,
  validateTimeSeriesForANOVA,
  validateTimeSeriesForSeasonalMK,
  getMinRequiredObservations,
  analyzeTrendSafe,
  processTimeSeries,
} from '../src/services/statistics';

function assert(cond: boolean, message: string): void {
  if (!cond) {
    throw new Error(`ASSERTION FAILED: ${message}`);
  }
}

function assertEqual<T>(actual: T, expected: T, message: string): void {
  if (actual !== expected) {
    throw new Error(`ASSERTION FAILED: ${message} — expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

// Helper to create minimal valid points
function createPoints(count: number, options: { 
  raw?: number | null; 
  anomaly?: number;
  dateStart?: string;
  monthStep?: number;
} = {}): { date: string; year: number; month: number; raw: number | null; anomaly?: number; isMissing?: boolean }[] {
  const points = [];
  const [startYear, startMonth] = options.dateStart?.split('-').map(Number) ?? [2000, 1];
  const step = options.monthStep ?? 1;
  for (let i = 0; i < count; i++) {
    const month = ((startMonth - 1 + i * step) % 12) + 1;
    const year = startYear + Math.floor((startMonth - 1 + i * step) / 12);
    points.push({
      date: `${year}-${String(month).padStart(2, '0')}`,
      year,
      month,
      raw: options.raw ?? 10 + i * 0.1,
      anomaly: options.anomaly,
      isMissing: options.raw === null,
    });
  }
  return points;
}

// Helper to create points with some nulls/NaN/Infinity
function createMixedPoints(validCount: number, nullCount: number, nanCount: number, infCount: number) {
  const points = [];
  for (let i = 0; i < validCount; i++) {
    points.push({ date: `2000-${String(i + 1).padStart(2, '0')}`, year: 2000, month: i + 1, raw: 10 + i });
  }
  for (let i = 0; i < nullCount; i++) {
    points.push({ date: `2000-${String(validCount + i + 1).padStart(2, '0')}`, year: 2000, month: validCount + i + 1, raw: null });
  }
  for (let i = 0; i < nanCount; i++) {
    points.push({ date: `2000-${String(validCount + nullCount + i + 1).padStart(2, '0')}`, year: 2000, month: validCount + nullCount + i + 1, raw: NaN });
  }
  for (let i = 0; i < infCount; i++) {
    points.push({ date: `2000-${String(validCount + nullCount + nanCount + i + 1).padStart(2, '0')}`, year: 2000, month: validCount + nullCount + nanCount + i + 1, raw: Infinity });
  }
  return points;
}

console.log('=== EMPTY TIME-SERIES REGRESSION TESTS ===\n');

// ============================================================
// 1. getMinRequiredObservations per-method thresholds
// ============================================================
{
  console.log('1. getMinRequiredObservations per-method thresholds');
  
  assertEqual(getMinRequiredObservations('mann-kendall', true), 4, 'mann-kendall anomaly');
  assertEqual(getMinRequiredObservations('theil-sen'), 2, 'theil-sen');
  assertEqual(getMinRequiredObservations('ols-regression'), 3, 'ols-regression');
  assertEqual(getMinRequiredObservations('seasonal-mk'), 12, 'seasonal-mk');
  assertEqual(getMinRequiredObservations('anova'), 3, 'anova');
  assertEqual(getMinRequiredObservations('unknown-method'), 3, 'unknown defaults to 3');
  
  console.log('   PASS\n');
}

// ============================================================
// 2. validateTimeSeriesForAnalysis — empty array
// ============================================================
{
  console.log('2. validateTimeSeriesForAnalysis — empty array');
  
  const result = validateTimeSeriesForAnalysis([], 'temp_air');
  
  assertEqual(result.state, 'empty', 'state should be empty');
  assertEqual(result.totalPoints, 0, 'totalPoints 0');
  assertEqual(result.validCount, 0, 'validCount 0');
  assertEqual(result.minRequired, 4, 'minRequired 4 for trend-analysis');
  assert(result.messages.length === 0, 'no messages for empty array');
  
  console.log('   PASS\n');
}

// ============================================================
// 3. validateTimeSeriesForAnalysis — all null
// ============================================================
{
  console.log('3. validateTimeSeriesForAnalysis — all null');
  
  const points = createMixedPoints(0, 5, 0, 0);
  const result = validateTimeSeriesForAnalysis(points, 'temp_air');
  
  assertEqual(result.state, 'unavailable', 'state should be unavailable (0 valid < 4 min)');
  assertEqual(result.totalPoints, 5, 'totalPoints 5');
  assertEqual(result.validCount, 0, 'validCount 0');
  assert(result.messages.some(m => m.includes('5 null observations')), 'should report nulls');
  
  console.log('   PASS\n');
}

// ============================================================
// 4. validateTimeSeriesForAnalysis — one valid observation
// ============================================================
{
  console.log('4. validateTimeSeriesForAnalysis — one valid observation');
  
  const points = createMixedPoints(1, 0, 0, 0);
  const result = validateTimeSeriesForAnalysis(points, 'temp_air');
  
  assertEqual(result.state, 'unavailable', 'state should be unavailable (1 valid < 4 min)');
  assertEqual(result.validCount, 1, 'validCount 1');
  
  console.log('   PASS\n');
}

// ============================================================
// 5. validateTimeSeriesForAnalysis — two valid observations
// ============================================================
{
  console.log('5. validateTimeSeriesForAnalysis — two valid observations');
  
  const points = createMixedPoints(2, 0, 0, 0);
  const result = validateTimeSeriesForAnalysis(points, 'temp_air');
  
  assertEqual(result.state, 'unavailable', 'state should be unavailable (2 valid < 4 min)');
  assertEqual(result.validCount, 2, 'validCount 2');
  
  console.log('   PASS\n');
}

// ============================================================
// 6. validateTimeSeriesForAnalysis — NaN and Infinity
// ============================================================
{
  console.log('6. validateTimeSeriesForAnalysis — NaN and Infinity');
  
  const points = createMixedPoints(3, 0, 2, 1);
  const result = validateTimeSeriesForAnalysis(points, 'temp_air');
  
  assertEqual(result.state, 'unavailable', 'state should be unavailable (3 valid < 4 min)');
  assertEqual(result.validCount, 3, 'validCount 3');
  assert(result.messages.some(m => m.includes('NaN')), 'should report NaN');
  assert(result.messages.some(m => m.includes('Infinity')), 'should report Infinity');
  
  console.log('   PASS\n');
}

// ============================================================
// 7. validateTimeSeriesForAnalysis — exactly min required (4)
// ============================================================
{
  console.log('7. validateTimeSeriesForAnalysis — exactly min required (4)');
  
  const points = createMixedPoints(4, 0, 0, 0);
  const result = validateTimeSeriesForAnalysis(points, 'temp_air');
  
  assertEqual(result.state, 'ready', 'state should be ready (4 valid >= 4 min)');
  assertEqual(result.validCount, 4, 'validCount 4');
  
  console.log('   PASS\n');
}

// ============================================================
// 8. validateTimeSeriesForAnalysis — duplicate timestamps
// ============================================================
{
  console.log('8. validateTimeSeriesForAnalysis — duplicate timestamps');
  
  const points = createPoints(4, { raw: 10 });
  points[1].date = points[0].date; // duplicate
  const result = validateTimeSeriesForAnalysis(points, 'temp_air');
  
  assertEqual(result.state, 'error', 'state should be error (duplicate timestamps)');
  assert(result.messages.some(m => m.includes('Duplicate')), 'should report duplicate');
  
  console.log('   PASS\n');
}

// ============================================================
// 9. validateTimeSeriesForAnalysis — non-monotonic timestamps
// ============================================================
{
  console.log('9. validateTimeSeriesForAnalysis — non-monotonic timestamps');
  
  const points = createPoints(4, { raw: 10 });
  points[2].date = '2000-01'; // goes back in time
  const result = validateTimeSeriesForAnalysis(points, 'temp_air');
  
  assertEqual(result.state, 'error', 'state should be error (non-monotonic)');
  assert(result.messages.some(m => m.includes('Non-monotonic')), 'should report non-monotonic');
  
  console.log('   PASS\n');
}

// ============================================================
// 10. validateTimeSeriesForAnalysis — invalid date format
// ============================================================
{
  console.log('10. validateTimeSeriesForAnalysis — invalid date format');
  
  const points = createPoints(4, { raw: 10 });
  points[0].date = '2000/01'; // wrong format
  const result = validateTimeSeriesForAnalysis(points, 'temp_air');
  
  assertEqual(result.state, 'error', 'state should be error (invalid date)');
  assert(result.messages.some(m => m.includes('Invalid date')), 'should report invalid date');
  
  console.log('   PASS\n');
}

// ============================================================
// 11. validateTimeSeriesForTheilSen — requires 2 valid
// ============================================================
{
  console.log('11. validateTimeSeriesForTheilSen — requires 2 valid');
  
  // 1 valid = unavailable
  let points = createMixedPoints(1, 0, 0, 0);
  let result = validateTimeSeriesForTheilSen(points, 'temp_air');
  assertEqual(result.state, 'unavailable', '1 valid < 2 min');
  
  // 2 valid = ready
  points = createMixedPoints(2, 0, 0, 0);
  result = validateTimeSeriesForTheilSen(points, 'temp_air');
  assertEqual(result.state, 'ready', '2 valid >= 2 min');
  
  console.log('   PASS\n');
}

// ============================================================
// 12. validateTimeSeriesForOLS — requires 3 valid
// ============================================================
{
  console.log('12. validateTimeSeriesForOLS — requires 3 valid');
  
  let points = createMixedPoints(2, 0, 0, 0);
  let result = validateTimeSeriesForOLS(points, 'temp_air');
  assertEqual(result.state, 'unavailable', '2 valid < 3 min');
  
  points = createMixedPoints(3, 0, 0, 0);
  result = validateTimeSeriesForOLS(points, 'temp_air');
  assertEqual(result.state, 'ready', '3 valid >= 3 min');
  
  console.log('   PASS\n');
}

// ============================================================
// 13. validateTimeSeriesForANOVA — per-group min
// ============================================================
{
  console.log('13. validateTimeSeriesForANOVA — per-group min');
  
  // 2 groups, need 4 total
  let points = createMixedPoints(3, 0, 0, 0);
  let result = validateTimeSeriesForANOVA(points, 2);
  assertEqual(result.state, 'unavailable', '3 valid < 4 min (2 groups * 2)');
  
  points = createMixedPoints(4, 0, 0, 0);
  result = validateTimeSeriesForANOVA(points, 2);
  assertEqual(result.state, 'ready', '4 valid >= 4 min');
  
  // 3 groups, need 6 total
  points = createMixedPoints(5, 0, 0, 0);
  result = validateTimeSeriesForANOVA(points, 3);
  assertEqual(result.state, 'unavailable', '5 valid < 6 min (3 groups * 2)');
  
  console.log('   PASS\n');
}

// ============================================================
// 14. validateTimeSeriesForSeasonalMK — requires 24 (12*2)
// ============================================================
{
  console.log('14. validateTimeSeriesForSeasonalMK — requires 24 (12*2)');
  
  let points = createMixedPoints(23, 0, 0, 0);
  let result = validateTimeSeriesForSeasonalMK(points);
  assertEqual(result.state, 'unavailable', '23 valid < 24 min');
  
  points = createMixedPoints(24, 0, 0, 0);
  result = validateTimeSeriesForSeasonalMK(points);
  assertEqual(result.state, 'ready', '24 valid >= 24 min');
  
  console.log('   PASS\n');
}

// ============================================================
// 15. analyzeTrendSafe — empty array returns empty state
// ============================================================
{
  console.log('15. analyzeTrendSafe — empty array');
  
  const response = analyzeTrendSafe(
    [],
    'temp_air',
    'Surface Air Temperature',
    '°C',
    'Test Region',
    [0, 0]
  );
  
  assertEqual(response.state, 'empty', 'state should be empty');
  assertEqual(response.result, undefined, 'result should be undefined');
  assert(response.explanation !== undefined && response.explanation.includes('No time series data'), 'explanation should mention no data');
  assert(response.validation !== undefined, 'validation should be present');
  assertEqual(response.validation?.state, 'empty', 'validation.state empty');
  
  console.log('   PASS\n');
}

// ============================================================
// 16. analyzeTrendSafe — insufficient data returns unavailable
// ============================================================
{
  console.log('16. analyzeTrendSafe — insufficient data (3 points)');
  
  const points = createMixedPoints(3, 0, 0, 0);
  const response = analyzeTrendSafe(
    points,
    'temp_air',
    'Surface Air Temperature',
    '°C',
    'Test Region',
    [0, 0]
  );
  
  assertEqual(response.state, 'unavailable', 'state should be unavailable');
  assertEqual(response.result, undefined, 'result should be undefined');
  assert(response.explanation !== undefined, 'explanation should be present');
  assert(response.validation !== undefined, 'validation should be present');
  assertEqual(response.validation?.validCount, 3, 'validation.validCount 3');
  assertEqual(response.validation?.minRequired, 4, 'validation.minRequired 4');
  
  console.log('   PASS\n');
}

// ============================================================
// 17. analyzeTrendSafe — sufficient data returns ready with result
// ============================================================
{
  console.log('17. analyzeTrendSafe — sufficient data (6 points)');
  
  const points = createMixedPoints(6, 0, 0, 0);
  const response = analyzeTrendSafe(
    points,
    'temp_air',
    'Surface Air Temperature',
    '°C',
    'Test Region',
    [0, 0]
  );
  
  assertEqual(response.state, 'ready', 'state should be ready');
  assert(response.result !== undefined, 'result should be present');
  assertEqual(response.result!.variableId, 'temp_air', 'result.variableId');
  assertEqual(response.result!.sampleCount, 6, 'result.sampleCount 6');
  assert(response.explanation === undefined || response.explanation === '', 'no explanation for ready state');
  
  console.log('   PASS\n');
}

// ============================================================
// 18. analyzeTrendSafe — NaN/Infinity filtered out
// ============================================================
{
  console.log('18. analyzeTrendSafe — NaN/Infinity filtered');
  
  // 4 valid + 2 NaN + 1 Inf = 4 valid (meets min)
  const points = createMixedPoints(4, 0, 2, 1);
  const response = analyzeTrendSafe(
    points,
    'temp_air',
    'Surface Air Temperature',
    '°C',
    'Test Region',
    [0, 0]
  );
  
  assertEqual(response.state, 'ready', 'should be ready (4 valid after filtering)');
  assertEqual(response.validation?.validCount, 4, 'validation.validCount 4');
  assertEqual(response.validation?.totalPoints, 7, 'validation.totalPoints 7');
  
  console.log('   PASS\n');
}

// ============================================================
// 19. processTimeSeries preserves null slots (polar night)
// ============================================================
{
  console.log('19. processTimeSeries preserves null slots');
  
  const rawPoints = [
    { date: '2000-01', value: 10 },
    { date: '2000-02', value: -999 }, // NASA fill value
    { date: '2000-03', value: 12 },
  ];
  
  const processed = processTimeSeries(rawPoints);
  
  assertEqual(processed.length, 3, '3 points preserved');
  assertEqual(processed[0].raw, 10, 'first point raw 10');
  assertEqual(processed[1].raw, null, 'second point raw null (polar night preserved)');
  assertEqual(processed[1].isMissing, true, 'second point isMissing true');
  assertEqual(processed[2].raw, 12, 'third point raw 12');
  
  console.log('   PASS\n');
}

// ============================================================
// 20. analyzeTrendSafe — uses anomaly when requested (default)
// ============================================================
{
  console.log('20. analyzeTrendSafe — uses anomaly for trend');
  
  // Create points with seasonal cycle and trend
  const points = createPoints(60, { raw: 15, dateStart: '2000-01' });
  // Add seasonal signal to raw
  for (let i = 0; i < points.length; i++) {
    const seasonal = 5 * Math.sin((2 * Math.PI * (points[i].month - 1)) / 12);
    const trend = 0.02 * i; // small upward trend
    points[i].raw = points[i].raw! + seasonal + trend;
  }
  
  const response = analyzeTrendSafe(
    points,
    'temp_air',
    'Surface Air Temperature',
    '°C',
    'Test Region',
    [0, 0],
    true // use anomaly
  );
  
  assertEqual(response.state, 'ready', 'state ready');
  // With anomaly, seasonal cycle removed, trend should be detected
  assert(response.result !== undefined, 'result present');
  
  console.log('   PASS\n');
}

console.log('=== ALL EMPTY TIME-SERIES REGRESSION TESTS PASSED ===');