/**
 * Statistical algorithm tests.
 *
 * Converted from the standalone verification scripts:
 *   tests/phase1_stats.verify.ts
 *   tests/empty_timeseries.verify.ts
 *   tests/statistical_interpretation_verify.ts
 *   tests/anova.verify.ts
 *
 * These run under vitest and assert the statistics engine behaves correctly
 * for missing data, autocorrelation corrections, factorial ANOVA, nonlinear
 * fits, GEV, one-way ANOVA, Shapiro-Wilk, and empty/insufficient input.
 */
import {
  processTimeSeries,
  analyzeTrend,
  analyzeTrendSafe,
  seasonalMannKendall,
  hamedRaoCorrection,
  tfpwMannKendall,
  twoWayAnova,
  nonlinearRegression,
  fitGEV,
  runOneWayAnova,
  classifyAssumptionTest,
  shapiroWilkTest,
  validateTimeSeriesForAnalysis,
  validateTimeSeriesForTheilSen,
  validateTimeSeriesForOLS,
  validateTimeSeriesForANOVA,
  validateTimeSeriesForSeasonalMK,
  getMinRequiredObservations,
  bootstrapCI,
  senSlopeExact,
  computeDescriptiveStats,
} from '../src/services/statistics';
import type { TimeSeriesPoint } from '../src/types/earthScience';
import { describe, expect, it } from 'vitest';

function makePoints(
  count: number,
  options: { base?: number; trend?: number; startYear?: number; monthlySeasonal?: boolean } = {}
): TimeSeriesPoint[] {
  const base = options.base ?? 10;
  const trend = options.trend ?? 0;
  const startYear = options.startYear ?? 2000;
  const points: TimeSeriesPoint[] = [];
  for (let i = 0; i < count; i++) {
    const month = (i % 12) + 1;
    const year = startYear + Math.floor(i / 12);
    const seasonal = options.monthlySeasonal ? 5 * Math.sin((2 * Math.PI * (month - 1)) / 12) : 0;
    points.push({
      date: `${year}-${String(month).padStart(2, '0')}`,
      year,
      month,
      raw: base + trend * (i / 12) + seasonal,
      isMissing: false,
    });
  }
  return points;
}

describe('processTimeSeries — missing slot preservation', () => {
  it('preserves all time slots including null polar-night slots', () => {
    const raw: { date: string; value: number | null }[] = [];
    for (let i = 0; i < 60; i++) {
      const yr = 2018 + Math.floor(i / 12);
      const mo = (i % 12) + 1;
      raw.push({
        date: `${yr}-${String(mo).padStart(2, '0')}`,
        value: i === 12 || i === 36 ? null : Number((10 + 0.12 * (i / 12)).toFixed(3)),
      });
    }
    const processed = processTimeSeries(raw);
    expect(processed).toHaveLength(60);
    expect(processed[12].raw).toBeNull();
    expect(processed[12].isMissing).toBe(true);
    expect(processed[12].movingAvg).toBeUndefined();
  });

  it('empty input returns empty array', () => {
    expect(processTimeSeries([])).toEqual([]);
  });
});

describe('analyzeTrend / analyzeTrendSafe — missing data and significance', () => {
  it('detects a significant trend despite two missing slots', () => {
    const raw: { date: string; value: number | null }[] = [];
    for (let i = 0; i < 60; i++) {
      const yr = 2018 + Math.floor(i / 12);
      const mo = (i % 12) + 1;
      raw.push({
        date: `${yr}-${String(mo).padStart(2, '0')}`,
        value: i === 12 || i === 36 ? null : Number((10 + 0.12 * (i / 12)).toFixed(3)),
      });
    }
    const processed = processTimeSeries(raw);
    const summary = analyzeTrend(
      processed,
      'temp_air',
      'Surface Temp',
      '°C',
      'Svalbard Test',
      [78, 15],
      true
    );
    expect(summary.missingCount).toBe(2);
    expect(summary.sampleCount).toBe(58);
    expect(Math.abs(summary.slopePerDecade - 1.2)).toBeLessThan(0.25);
    expect(summary.mkPValue).toBeLessThan(0.001);
    expect(summary.tPValue).toBeLessThan(0.001);
  });

  it('empty input throws from analyzeTrend', () => {
    expect(() =>
      analyzeTrend([], 'temp_air', 'Air Temp', '°C', 'Region', [0, 0], true)
    ).toThrow('Cannot analyze empty time series');
  });

  it('empty input returns empty state from analyzeTrendSafe', () => {
    const response = analyzeTrendSafe([], 'temp_air', 'Air Temp', '°C', 'Region', [0, 0]);
    expect(response.state).toBe('empty');
    expect(response.result).toBeUndefined();
    expect(response.validation?.state).toBe('empty');
  });

  it('insufficient input returns unavailable from analyzeTrendSafe', () => {
    const points = makePoints(2);
    const response = analyzeTrendSafe(points, 'temp_air', 'Air Temp', '°C', 'Region', [0, 0]);
    expect(response.state).toBe('unavailable');
    expect(response.result).toBeUndefined();
  });

  it('sufficient input returns ready from analyzeTrendSafe', () => {
    const points = makePoints(60, { trend: 0.5 });
    const response = analyzeTrendSafe(points, 'temp_air', 'Air Temp', '°C', 'Region', [0, 0]);
    expect(response.state).toBe('ready');
    expect(response.result).toBeDefined();
    expect(response.result!.variableId).toBe('temp_air');
  });
});

describe('validation thresholds', () => {
  it('getMinRequiredObservations per method', () => {
    expect(getMinRequiredObservations('mann-kendall', true)).toBe(4);
    expect(getMinRequiredObservations('theil-sen')).toBe(2);
    expect(getMinRequiredObservations('ols-regression')).toBe(3);
    expect(getMinRequiredObservations('seasonal-mk')).toBe(12);
    expect(getMinRequiredObservations('anova')).toBe(3);
  });

  it('validateTimeSeriesForAnalysis — empty is empty', () => {
    const r = validateTimeSeriesForAnalysis([], 'temp_air');
    expect(r.state).toBe('empty');
    expect(r.validCount).toBe(0);
  });

  it('validateTimeSeriesForAnalysis — all null is unavailable', () => {
    const points = Array.from({ length: 5 }, (_, i) => ({
      date: `2000-${String(i + 1).padStart(2, '0')}`,
      year: 2000,
      month: i + 1,
      raw: null,
      isMissing: true,
    }));
    const r = validateTimeSeriesForAnalysis(points, 'temp_air');
    expect(r.state).toBe('unavailable');
    expect(r.validCount).toBe(0);
  });

  it('validateTimeSeriesForAnalysis — exactly 4 valid is ready', () => {
    const points = makePoints(4);
    const r = validateTimeSeriesForAnalysis(points, 'temp_air');
    expect(r.state).toBe('ready');
    expect(r.validCount).toBe(4);
  });

  it('theil-sen requires 3; ols requires 3; seasonal-mk requires 24; anova per-group', () => {
    // validateTimeSeriesForTheilSen/OLS delegate to validateTimeSeriesForAnalysis
    // with method 'trend-analysis', whose minimum is 3.
    expect(validateTimeSeriesForTheilSen(makePoints(2)).state).toBe('unavailable');
    expect(validateTimeSeriesForTheilSen(makePoints(3)).state).toBe('ready');
    expect(validateTimeSeriesForOLS(makePoints(2)).state).toBe('unavailable');
    expect(validateTimeSeriesForOLS(makePoints(3)).state).toBe('ready');
    expect(validateTimeSeriesForSeasonalMK(makePoints(23)).state).toBe('unavailable');
    expect(validateTimeSeriesForSeasonalMK(makePoints(24)).state).toBe('ready');
    expect(validateTimeSeriesForANOVA(makePoints(3), 2).state).toBe('unavailable');
    expect(validateTimeSeriesForANOVA(makePoints(4), 2).state).toBe('ready');
  });
});

describe('seasonal Mann-Kendall corrections', () => {
  it('Hamed-Rao inflates variance under positive AR(1)', () => {
    const redNoise: number[] = [];
    let state = 0;
    let rngState = 12345;
    const rand = () => {
      rngState = (rngState * 1664525 + 1013904223) >>> 0;
      return rngState / 4294967296 - 0.5;
    };
    for (let i = 0; i < 180; i++) {
      state = 0.85 * state + rand() * 2.0;
      redNoise.push(4.0 * Math.cos((2 * Math.PI * (i % 12)) / 12) + state);
    }
    const mkStd = seasonalMannKendall(redNoise, 12, 'standard');
    const mkHr = hamedRaoCorrection(redNoise, 12);
    expect(mkHr.varS).toBeGreaterThanOrEqual(mkStd.varS);
  });

  it('TFPW detects a true trend in autocorrelated seasonal data', () => {
    const redNoise: number[] = [];
    let state = 0;
    let rngState = 12345;
    const rand = () => {
      rngState = (rngState * 1664525 + 1013904223) >>> 0;
      return rngState / 4294967296 - 0.5;
    };
    for (let i = 0; i < 180; i++) {
      state = 0.85 * state + rand() * 2.0;
      redNoise.push(4.0 * Math.cos((2 * Math.PI * (i % 12)) / 12) + state);
    }
    const trending = redNoise.map((v, i) => v + 0.08 * i);
    const mkTfpw = tfpwMannKendall(trending, 12);
    expect(mkTfpw.z).toBeGreaterThan(4);
    expect(mkTfpw.pValue).toBeLessThan(0.001);
  });
});

describe('factorial ANOVA', () => {
  it('detects significant main effects and interaction', () => {
    const obs = [
      { levelA: 'North', levelB: 'Summer', value: 10 },
      { levelA: 'North', levelB: 'Summer', value: 11 },
      { levelA: 'North', levelB: 'Summer', value: 9 },
      { levelA: 'North', levelB: 'Winter', value: 2 },
      { levelA: 'North', levelB: 'Winter', value: 3 },
      { levelA: 'North', levelB: 'Winter', value: 1 },
      { levelA: 'South', levelB: 'Summer', value: 20 },
      { levelA: 'South', levelB: 'Summer', value: 21 },
      { levelA: 'South', levelB: 'Summer', value: 19 },
      { levelA: 'South', levelB: 'Winter', value: 19 },
      { levelA: 'South', levelB: 'Winter', value: 18 },
      { levelA: 'South', levelB: 'Winter', value: 20 },
    ];
    const anova2 = twoWayAnova(obs, 'Latitude', 'Season');
    expect(anova2.factorA.pValue).toBeLessThan(0.001);
    expect(anova2.factorB.pValue).toBeLessThan(0.001);
    expect(anova2.interaction.pValue).toBeLessThan(0.001);
  });
});

describe('nonlinear regression and GEV', () => {
  it('quadratic fit recovers near-perfect R²', () => {
    const x = Array.from({ length: 40 }, (_, i) => i + 1);
    const yQuad = x.map(xi => 3 - 0.5 * xi + 0.08 * xi * xi);
    const resQuad = nonlinearRegression(x, yQuad, 'quadratic');
    expect(resQuad.rSquared).toBeGreaterThan(0.999);
  });

  it('exponential and logistic fits exceed R² 0.99', () => {
    const x = Array.from({ length: 40 }, (_, i) => i + 1);
    const yExp = x.map((_, idx) => 1.5 + 2.0 * Math.exp(1.8 * (idx / 39)));
    const resExp = nonlinearRegression(x, yExp, 'exponential');
    expect(resExp.rSquared).toBeGreaterThan(0.99);

    const yLogi = x.map((_, idx) => 5.0 + 12.0 / (1 + Math.exp(-8.0 * (idx / 39 - 0.5))));
    const resLogi = nonlinearRegression(x, yLogi, 'logistic');
    expect(resLogi.rSquared).toBeGreaterThan(0.99);
  });

  it('GEV block-maxima fit returns monotonic return levels', () => {
    const vals: number[] = [];
    let s = 9876;
    const rnd = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return Math.max(1e-4, Math.min(0.9999, s / 4294967296));
    };
    for (let i = 0; i < 240; i++) {
      vals.push(25 + (-10) * Math.log(-Math.log(rnd())));
    }
    const gev = fitGEV(vals, 12);
    expect(gev.blockMaxima).toHaveLength(20);
    expect(gev.scaleSigma).toBeGreaterThan(0);
    expect(Number.isFinite(gev.shapeXi)).toBe(true);
    for (let i = 1; i < gev.returnLevels.length; i++) {
      expect(gev.returnLevels[i].level).toBeGreaterThanOrEqual(gev.returnLevels[i - 1].level);
    }
  });
});

describe('descriptive stats and bootstrap', () => {
  it('computeDescriptiveStats returns basic moments', () => {
    const stats = computeDescriptiveStats([1, 2, 3, 4, 5]);
    expect(stats.count).toBe(5);
    expect(stats.mean).toBe(3);
    expect(stats.min).toBe(1);
    expect(stats.max).toBe(5);
  });

  it('bootstrapCI returns lower/upper bounds', () => {
    const values = Array.from({ length: 50 }, (_, i) => i + 1);
    const ci = bootstrapCI(values, arr => arr.reduce((a, b) => a + b, 0) / arr.length, 1000, 0.05);
    expect(Number.isFinite(ci.estimate)).toBe(true);
    expect(Number.isFinite(ci.ciLower)).toBe(true);
    expect(Number.isFinite(ci.ciUpper)).toBe(true);
    expect(ci.ciLower).toBeLessThanOrEqual(ci.ciUpper);
    expect(ci.ciLower).toBeLessThanOrEqual(ci.estimate);
    expect(ci.estimate).toBeLessThanOrEqual(ci.ciUpper);
  });
});

describe('one-way ANOVA and assumption classification', () => {
  it('runOneWayAnova computes F and p', () => {
    const data = {
      response: [1, 2, 3, 4, 5, 6, 7, 8, 9],
      group: ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    };
    const r = runOneWayAnova(data);
    expect(r.error).toBeNull();
    expect(r.k).toBe(3);
    expect(r.N).toBe(9);
    expect(Number.isFinite(r.table[0]?.F ?? -1)).toBe(true);
    expect(Number.isFinite(r.table[0]?.p ?? -1)).toBe(true);
  });

  it('Durbin-Watson is null without observation order', () => {
    const data = {
      response: [1, 2, 3, 4, 5, 6, 7, 8, 9],
      group: ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    };
    const r = runOneWayAnova(data);
    expect(r.assumptions.durbinWatson).toBeNull();
    expect(r.assumptions.lag1Autocorr).toBeNull();
  });

  it('Durbin-Watson is non-null with observation order', () => {
    const data = {
      response: [1, 2, 3, 4, 5, 6, 7, 8, 9],
      group: ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
      observationOrder: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    };
    const r = runOneWayAnova(data, { preserveObservationOrder: true });
    expect(r.assumptions.durbinWatson).not.toBeNull();
    expect(r.assumptions.lag1Autocorr).not.toBeNull();
  });

  it('classifyAssumptionTest verdicts', () => {
    expect(classifyAssumptionTest(0.10)).toBe('no_evidence_against');
    expect(classifyAssumptionTest(0.05)).toBe('evidence_against');
    expect(classifyAssumptionTest(0.04)).toBe('evidence_against');
    expect(classifyAssumptionTest(null)).toBe('insufficient_data');
    expect(classifyAssumptionTest(undefined)).toBe('insufficient_data');
    expect(classifyAssumptionTest(NaN)).toBe('insufficient_data');
  });
});

describe('Shapiro-Wilk and causal-claim audit', () => {
  it('plain English verdict contains no unsupported causal claims', () => {
    const points: TimeSeriesPoint[] = [];
    for (let y = 2000; y <= 2023; y++) {
      for (let m = 1; m <= 12; m++) {
        points.push({
          date: `${y}-${String(m).padStart(2, '0')}`,
          year: y,
          month: m,
          raw: 15 + 0.02 * (y - 2000) + m * 0.01,
          isMissing: false,
        });
      }
    }
    const result = analyzeTrend(points, 'temp_air', 'Air Temperature', 'degC', 'Test Region', [0, 0], true);
    const causalKeywords = [
      'causes', 'caused by', 'because of', 'due to', 'proves', 'confirm',
      'confirms', 'establishes', 'established', 'directly tracks',
      'directly reflects', 'reveals', 'demonstrates', 'accelerates',
      'triggers', 'imperils', 'exacerbates', 'compounds', 'forcing',
    ];
    const lower = result.plainEnglishVerdict.toLowerCase();
    for (const kw of causalKeywords) {
      expect(lower).not.toContain(kw);
    }
    expect(result.plainEnglishVerdict.indexOf('caused by')).toBe(-1);
    expect(result.plainEnglishVerdict.indexOf('proves')).toBe(-1);
  });

  it('normal-like data yields an expected Shapiro verdict', () => {
    const normalVals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const normalResult = shapiroWilkTest(normalVals);
    if (normalResult) {
      expect(['no_evidence_against', 'evidence_against']).toContain(normalResult.verdict);
    }
  });

  it('constant data yields null from Shapiro-Wilk', () => {
    expect(shapiroWilkTest([5, 5, 5, 5])).toBeNull();
  });
});

describe('senSlopeExact', () => {
  it('returns a finite slope for a monotonic series', () => {
    const values = [1, 2, 3, 4, 5];
    const slope = senSlopeExact(values);
    expect(Number.isFinite(slope)).toBe(true);
    expect(slope).toBeGreaterThan(0);
  });
});
