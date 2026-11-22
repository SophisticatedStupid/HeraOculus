/**
 * Regression tests for the seasonal Mann-Kendall / Hamed-Rao implementation.
 *
 * These tests target the temporal-indexing defect in the Hamed-Rao
 * autocorrelation calculation: observations must be keyed by their original
 * array index (year + season), NOT by year alone, and the autocorrelation may
 * only pair observations of the SAME season separated by the correct temporal
 * lag (e.g. January with January one year later).  January must never be
 * correlated with February merely because they fall in adjacent years.
 *
 * Vitest runs these under Node; no live network is required.
 */
import {
  describe,
  expect,
  it,
} from 'vitest';
import {
  hamedRaoCorrection,
  seasonalMannKendall,
  senSlopeExact,
} from '../src/services/statistics/timeseries';

const SEASON_LENGTH = 12;

/** Deterministic LCG in [-0.5, 0.5). */
function lcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
}

/** Build a complete monthly series (values[i]: year=floor(i/12), season=i%12). */
function buildMonthly(
  nYears: number,
  value: (year: number, month: number) => number
): number[] {
  const vals: number[] = [];
  for (let y = 0; y < nYears; y++) {
    for (let m = 0; m < SEASON_LENGTH; m++) {
      vals.push(value(y, m));
    }
  }
  return vals;
}

/** Average ranks with ties (independent re-implementation for cross-check). */
function averageRanks(values: number[]): number[] {
  const indexed = values.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  const ranks = new Array(values.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length && indexed[j].v === indexed[i].v) j++;
    const avgRank = (i + j - 1) / 2 + 1;
    for (let k = i; k < j; k++) ranks[indexed[k].i] = avgRank;
    i = j;
  }
  return ranks;
}

/** Independent recomputation of the Hamed-Rao per-lag rho using same-season
 *  pairing by scanning (no year-only map).  Used to verify that the
 *  index-map-based implementation pairs exactly the intended observations. */
function independentHamedRaoRho(
  vals: number[],
  seasonLength: number
): number[] {
  const n = vals.length;
  const validObs = vals
    .map((v, i) => ({
      year: Math.floor(i / seasonLength),
      season: i % seasonLength,
      idx: i,
      value: v,
    }))
    .filter((o) => !isNaN(o.value) && isFinite(o.value));
  const mLen = validObs.length;
  if (mLen < 6) return [];
  const years = validObs.map((o) => o.year);
  const valuesArr = validObs.map((o) => o.value);
  const beta = senSlopeExact(valuesArr, years);
  const detrended = validObs.map((o) => o.value - beta * o.year);
  const ranks = averageRanks(detrended);
  const meanRank = ranks.reduce((a, b) => a + b, 0) / mLen;
  const centered = ranks.map((r) => r - meanRank);
  const denom = centered.reduce((s, r) => s + r * r, 0);
  const out: number[] = [];
  if (denom <= 1e-12) return out;
  const maxLag = Math.min(12, Math.floor(mLen / 4));
  for (let lag = 1; lag <= maxLag; lag++) {
    let num = 0;
    let pairCount = 0;
    const step = lag * seasonLength;
    for (let i = 0; i < mLen; i++) {
      const partner = validObs.findIndex(
        (o) => o.idx === validObs[i].idx + step
      );
      if (partner >= 0) {
        num += centered[i] * centered[partner];
        pairCount++;
      }
    }
    out.push(pairCount > 0 ? num / denom : 0);
  }
  return out;
}

describe('seasonal Mann-Kendall / Hamed-Rao temporal indexing', () => {
  it('independent seasonal observations are not flagged as a trend', () => {
    const vals = buildMonthly(20, () => lcg(7)());
    const hr = hamedRaoCorrection(vals, SEASON_LENGTH);
    expect(hr.pValue).toBeGreaterThan(0.05);
    // Independent data => no autocorrelation correction applied.
    expect(hr.autocorrelationCorrection).toBeUndefined();
  });

  it('complete monthly series with a monotonic trend is significant', () => {
    const vals = buildMonthly(20, (y) => y + 0.05 * y);
    const hr = hamedRaoCorrection(vals, SEASON_LENGTH);
    expect(Math.abs(hr.z)).toBeGreaterThan(2);
    expect(hr.pValue).toBeLessThan(0.05);
    expect(hr.s).not.toBe(0);
  });

  it('strong seasonal structure without a trend is not significant', () => {
    const vals = buildMonthly(20, (y, m) =>
      10 + 5 * Math.sin((2 * Math.PI * m) / SEASON_LENGTH) + 0.5 * lcg(11)()
    );
    const hr = hamedRaoCorrection(vals, SEASON_LENGTH);
    expect(hr.pValue).toBeGreaterThan(0.05);
  });

  it('autocorrelated seasonal observations inflate the variance', () => {
    const vals: number[] = [];
    let state = 0;
    const rand = lcg(12345);
    for (let i = 0; i < 180; i++) {
      state = 0.85 * state + rand() * 2.0;
      vals.push(4 * Math.cos((2 * Math.PI * (i % 12)) / 12) + state);
    }
    const std = seasonalMannKendall(vals, SEASON_LENGTH, 'standard');
    const hr = hamedRaoCorrection(vals, SEASON_LENGTH);
    expect(hr.varS).toBeGreaterThanOrEqual(std.varS);
  });

  it('missing months are skipped without creating spurious pairs', () => {
    const vals = buildMonthly(20, (y, m) => y + 0.1 * m + lcg(5)());
    for (let y = 0; y < 20; y++) {
      vals[y * 12 + 0] = NaN; // January of every year
      if (y % 2 === 0) vals[y * 12 + 6] = NaN; // July of even years
    }
    const hr = hamedRaoCorrection(vals, SEASON_LENGTH);
    expect(Number.isFinite(hr.varS)).toBe(true);
    expect(Number.isFinite(hr.z)).toBe(true);
    expect(hr.pValue).toBeGreaterThanOrEqual(0);
    expect(hr.pValue).toBeLessThanOrEqual(1);
    expect(hr.s).toBeDefined();
  });

  it('missing entire years are handled without crashing', () => {
    const vals = buildMonthly(20, (y, m) => y + 0.1 * m + lcg(3)());
    // Blank out years 5 and 12 entirely.
    for (const y of [5, 12]) {
      for (let m = 0; m < SEASON_LENGTH; m++) vals[y * 12 + m] = NaN;
    }
    const hr = hamedRaoCorrection(vals, SEASON_LENGTH);
    expect(Number.isFinite(hr.varS)).toBe(true);
    expect(Number.isFinite(hr.z)).toBe(true);
    expect(hr.pValue).toBeGreaterThanOrEqual(0);
    expect(hr.pValue).toBeLessThanOrEqual(1);
  });

  it('same-season pairing captures within-season autocorrelation', () => {
    const nYears = 20;
    const rand = lcg(999);
    const vals: number[] = [];
    let janState = 0;
    for (let y = 0; y < nYears; y++) {
      for (let m = 0; m < SEASON_LENGTH; m++) {
        if (m === 0) {
          janState = 0.9 * janState + rand() * 0.5; // strongly autocorrelated
          vals.push(janState);
        } else if (m === 1) {
          vals.push(rand()); // independent noise
        } else {
          vals.push(0.0); // constant
        }
      }
    }
    const std = seasonalMannKendall(vals, SEASON_LENGTH, 'standard');
    const hr = hamedRaoCorrection(vals, SEASON_LENGTH);
    // Same-season pairing must detect January's lag-1 autocorrelation and
    // inflate the variance relative to the standard test.
    expect(hr.varS).toBeGreaterThan(std.varS);
    expect(hr.autocorrelationCorrection).toBeGreaterThan(1.0);
    // rho at lag 1 (same season, one year apart) must be clearly positive,
    // reflecting January's year-over-year autocorrelation.
    expect(hr.hamedRaoRhoDiagnostic?.[0]).toBeGreaterThan(0.2);
  });

  it('cross-season observations are NOT paired (shuffle destroys signal)', () => {
    const nYears = 20;
    const rand = lcg(999);
    const original: number[] = [];
    let janState = 0;
    for (let y = 0; y < nYears; y++) {
      for (let m = 0; m < SEASON_LENGTH; m++) {
        if (m === 0) {
          janState = 0.9 * janState + rand() * 0.5;
          original.push(janState);
        } else if (m === 1) {
          original.push(rand());
        } else {
          original.push(0.0);
        }
      }
    }
    const base = hamedRaoCorrection(original, SEASON_LENGTH);
    // Shuffle January and February within each year: this destroys January's
    // same-season year-over-year autocorrelation while preserving the yearly
    // structure, so cross-season pairing must not rescue it.
    const shuffled = [...original];
    for (let y = 0; y < nYears; y++) {
      const janIdx = y * 12 + 0;
      const febIdx = y * 12 + 1;
      [shuffled[janIdx], shuffled[febIdx]] = [
        shuffled[febIdx],
        shuffled[janIdx],
      ];
    }
    const shuffledHr = hamedRaoCorrection(shuffled, SEASON_LENGTH);
    expect(shuffledHr.autocorrelationCorrection).toBeLessThanOrEqual(
      base.autocorrelationCorrection
    );
    expect(shuffledHr.varS).toBeLessThanOrEqual(base.varS);
  });

  it('Hamed-Rao rho matches an independent same-season pairing computation', () => {
    const vals = buildMonthly(15, (y, m) =>
      y + 0.3 * m + 0.8 * Math.sin((2 * Math.PI * m) / 12) + lcg(2024)()
    );
    const hr = hamedRaoCorrection(vals, SEASON_LENGTH);
    const ref = independentHamedRaoRho(vals, SEASON_LENGTH);
    expect(hr.hamedRaoRhoDiagnostic).toBeDefined();
    expect(hr.hamedRaoRhoDiagnostic!.length).toBe(ref.length);
    for (let i = 0; i < ref.length; i++) {
      expect(hr.hamedRaoRhoDiagnostic![i]).toBeCloseTo(ref[i], 6);
    }
  });
});
