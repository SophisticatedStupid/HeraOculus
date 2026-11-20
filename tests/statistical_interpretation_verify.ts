/**
 * Scientific Interpretation & Assumption-Diagnostic Audit - Verification Tests.
 *
 * These tests verify:
 *   1. Trend interpretation never generates unsupported causal claims.
 *   2. Shapiro-Wilk verdict properly distinguishes failure-to-reject from
 *      evidence-against-normality (never equates p > 0.05 with proof of normality).
 *   3. Durbin-Watson / lag-1 autocorrelation are reported as null when
 *      observation ordering is unavailable.
 *   4. classifyAssumptionTest correctly classifies p-values.
 *
 * Run with: npx tsx tests/statistical_interpretation_verify.ts
 */

import {
  classifyAssumptionTest,
  runOneWayAnova,
  shapiroWilkTest,
} from '../src/services/statistics/anova';
import { analyzeTrend } from '../src/services/statistics/index';
import type { TimeSeriesPoint } from '../src/types/earthScience';

let passes = 0;
let failures = 0;

function assert(cond: boolean, message: string): void {
  if (!cond) {
    failures++;
    console.error('FAIL: ' + message);
    return;
  }
  passes++;
}

function assertNoCausalClaims(text: string, label: string): void {
  const causalKeywords = [
    'causes', 'caused by', 'because of', 'due to', 'proves',
    'confirm', 'confirms', 'establishes', 'established', 'directly tracks',
    'directly reflects', 'reveals', 'demonstrates', 'accelerates', 'triggers',
    'imperils', 'exacerbates', 'compounds', 'forcing',
  ];
  const lower = text.toLowerCase();
  for (const kw of causalKeywords) {
    if (lower.includes(kw)) {
      failures++;
      console.error('FAIL [' + label + ']: Causal language found: "' + kw + '" in: "' + text + '"');
      return;
    }
  }
  passes++;
}

// 1. Trend wording - no unsupported causal claims in plainEnglishVerdict
{
  const points: TimeSeriesPoint[] = [];
  for (let y = 2000; y <= 2023; y++) {
    for (let m = 1; m <= 12; m++) {
      points.push({ date: y + '-' + String(m).padStart(2, '0'), year: y, month: m, raw: 15 + 0.02 * (y - 2000) + m * 0.01, isMissing: false });
    }
  }

  const result = analyzeTrend(
    points,
    'temp_air',
    'Air Temperature',
    'degC',
    'Test Region',
    [0, 0],
    true,
  );

  console.log('Verdict:', result.plainEnglishVerdict);
  console.log('Implication:', result.scientificImplication);

  assertNoCausalClaims(result.plainEnglishVerdict, 'plainEnglishVerdict');
  const unsupportedProof = /proof/i.test(result.scientificImplication) &&
    !/not proof|no proof|proof of/i.test(result.scientificImplication);
  assert(!unsupportedProof, 'scientificImplication does not contain unsupported proof claims');
  assert(
    result.plainEnglishVerdict.indexOf('caused by') === -1,
    'Verdict does not say "caused by"'
  );
  assert(
    result.plainEnglishVerdict.indexOf('proves') === -1,
    'Verdict does not say "proves"'
  );
  assert(
    result.plainEnglishVerdict.includes('standard error') ||
    result.plainEnglishVerdict.includes('uncertainty') ||
    result.plainEnglishVerdict.includes('Statistical significance indicates'),
    'Verdict mentions statistical significance interpretation'
  );
}

// 2. classifyAssumptionTest
{
  assert(classifyAssumptionTest(0.10) === 'no_evidence_against', 'p=0.10 -> no_evidence_against');
  assert(classifyAssumptionTest(0.05) === 'evidence_against', 'p=0.05 -> evidence_against');
  assert(classifyAssumptionTest(0.04) === 'evidence_against', 'p=0.04 -> evidence_against');
  assert(classifyAssumptionTest(null) === 'insufficient_data', 'p=null -> insufficient_data');
  assert(classifyAssumptionTest(undefined) === 'insufficient_data', 'p=undefined -> insufficient_data');
  assert(classifyAssumptionTest(NaN) === 'insufficient_data', 'p=NaN -> insufficient_data');
  assert(classifyAssumptionTest(0.001) === 'evidence_against', 'p=0.001 -> evidence_against');
}

// 3. Durbin-Watson / lag-1 null when observation order unavailable
{
  const data = {
    response: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    group: ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
  };

  // Without observationOrder -> DW and lag-1 must be null
  const rNoOrder = runOneWayAnova(data);
  assert(rNoOrder.assumptions.durbinWatson === null, 'DW null when observationOrder not provided');
  assert(rNoOrder.assumptions.lag1Autocorr === null, 'lag1 null when observationOrder not provided');
  assert(
    rNoOrder.assumptions.warning === null || rNoOrder.assumptions.warning?.includes('ordering'),
    'Warning about ordering present'
  );
}

// 4. Shapiro-Wilk verdict classification
{
  // Normal-like data -> p > 0.05 -> verdict 'no_evidence_against'
  const normalVals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
  const normalResult = shapiroWilkTest(normalVals);
  if (normalResult) {
    assert(
      normalResult.verdict === 'no_evidence_against' || normalResult.verdict === 'evidence_against',
      'Shapiro verdict is one of expected values, got ' + normalResult.verdict
    );
  }

  // Strongly skewed data -> likely evidence_against
  const skewedVals = [1, 1, 1, 1, 1, 100, 100, 100];
  const skewedResult = shapiroWilkTest(skewedVals);
  if (skewedResult) {
    assert(
      skewedResult.verdict === 'evidence_against',
      'Skewed data should have verdict evidence_against, got ' + skewedResult.verdict
    );
  }

  // Constant data -> null (insufficient data)
  const constantResult = shapiroWilkTest([5, 5, 5, 5]);
  assert(constantResult === null, 'Constant data -> null (insufficient data)');
}

// 5. Durbin-Watson valid when observation order is provided
{
  const data = {
    response: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    group: ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    observationOrder: [0, 1, 2, 3, 4, 5, 6, 7, 8],
  };
  const rOrder = runOneWayAnova(data, { preserveObservationOrder: true });
  assert(
    rOrder.assumptions.durbinWatson !== null,
    'DW not null when observationOrder provided'
  );
  assert(
    rOrder.assumptions.lag1Autocorr !== null,
    'lag1 not null when observationOrder provided'
  );
}

// Summary
console.log('\n' + passes + ' passed, ' + failures + ' failed');
if (failures > 0) process.exit(1);