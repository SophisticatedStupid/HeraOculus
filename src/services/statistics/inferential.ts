import { tCDF, fCDF, tQuantile } from './distributions';
import { Matrix, SVD, pseudoInverse } from 'ml-matrix';
import type { TwoWayAnovaResult } from '../../types/earthScience';

export type { TwoWayAnovaResult };

export interface TTestResult {
  t: number;
  df: number;
  pValue: number;
  meanDiff: number;
  ciLower: number;
  ciUpper: number;
  meanA: number;
  meanB?: number;
}

export interface AnovaResult {
  f: number;
  dfBetween: number;
  dfWithin: number;
  pValue: number;
  groupMeans: number[];
  grandMean: number;
  ssBetween: number;
  ssWithin: number;
  msBetween: number;
  msWithin: number;
}

export interface BootstrapResult {
  estimate: number;
  ciLower: number;
  ciUpper: number;
  standardError: number;
  iterations: number;
}

/** One-sample Student's t-test */
export function oneSampleTTest(values: number[], mu0 = 0, alpha = 0.05): TTestResult {
  const clean = values.filter(v => !isNaN(v) && isFinite(v));
  const n = clean.length;
  if (n < 2) {
    return { t: 0, df: 1, pValue: 1, meanDiff: 0, ciLower: 0, ciUpper: 0, meanA: 0 };
  }

  const mean = clean.reduce((s, v) => s + v, 0) / n;
  const variance = clean.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1);
  const se = Math.sqrt(variance / n);
  const df = n - 1;
  const t = se > 0 ? (mean - mu0) / se : 0;
  const pValue = 2 * (1 - tCDF(Math.abs(t), df));

  const tCrit = tQuantile(1 - alpha / 2, df);
  const ciLower = mean - tCrit * se;
  const ciUpper = mean + tCrit * se;

  return {
    t,
    df,
    pValue: Math.max(0, Math.min(1, pValue)),
    meanDiff: mean - mu0,
    ciLower,
    ciUpper,
    meanA: mean,
  };
}

/** Two-sample Student's t-test (independent Welch or paired) */
export function twoSampleTTest(
  a: number[],
  b: number[],
  paired = false,
  alpha = 0.05
): TTestResult {
  if (paired) {
    const minLen = Math.min(a.length, b.length);
    const diffs: number[] = [];
    for (let i = 0; i < minLen; i++) {
      if (!isNaN(a[i]) && !isNaN(b[i])) {
        diffs.push(a[i] - b[i]);
      }
    }
    const res = oneSampleTTest(diffs, 0, alpha);
    const meanA = a.reduce((s, v) => s + v, 0) / a.length;
    const meanB = b.reduce((s, v) => s + v, 0) / b.length;
    return { ...res, meanA, meanB };
  }

  const cleanA = a.filter(v => !isNaN(v) && isFinite(v));
  const cleanB = b.filter(v => !isNaN(v) && isFinite(v));
  const n1 = cleanA.length;
  const n2 = cleanB.length;
  if (n1 < 2 || n2 < 2) {
    return { t: 0, df: 1, pValue: 1, meanDiff: 0, ciLower: 0, ciUpper: 0, meanA: 0, meanB: 0 };
  }

  const m1 = cleanA.reduce((s, v) => s + v, 0) / n1;
  const m2 = cleanB.reduce((s, v) => s + v, 0) / n2;
  const v1 = cleanA.reduce((s, v) => s + (v - m1) ** 2, 0) / (n1 - 1);
  const v2 = cleanB.reduce((s, v) => s + (v - m2) ** 2, 0) / (n2 - 1);

  const se1 = v1 / n1;
  const se2 = v2 / n2;
  const seDiff = Math.sqrt(se1 + se2);

  const df = (se1 + se2) ** 2 / ((se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1));
  const t = seDiff > 0 ? (m1 - m2) / seDiff : 0;
  const pValue = 2 * (1 - tCDF(Math.abs(t), df));

  const tCrit = tQuantile(1 - alpha / 2, Math.max(1, Math.round(df)));
  const ciLower = (m1 - m2) - tCrit * seDiff;
  const ciUpper = (m1 - m2) + tCrit * seDiff;

  return {
    t,
    df,
    pValue: Math.max(0, Math.min(1, pValue)),
    meanDiff: m1 - m2,
    ciLower,
    ciUpper,
    meanA: m1,
    meanB: m2,
  };
}

/** One-way Analysis of Variance (ANOVA) */
export function oneWayAnova(groups: number[][]): AnovaResult {
  const cleanGroups = groups.map(g => g.filter(v => !isNaN(v) && isFinite(v))).filter(g => g.length > 0);
  const k = cleanGroups.length;
  if (k < 2) {
    return {
      f: 0,
      dfBetween: 1,
      dfWithin: 1,
      pValue: 1,
      groupMeans: [],
      grandMean: 0,
      ssBetween: 0,
      ssWithin: 0,
      msBetween: 0,
      msWithin: 0,
    };
  }

  const groupMeans = cleanGroups.map(g => g.reduce((s, v) => s + v, 0) / g.length);
  const totalN = cleanGroups.reduce((s, g) => s + g.length, 0);
  const grandMean = cleanGroups.reduce((s, g) => s + g.reduce((ss, v) => ss + v, 0), 0) / totalN;

  let ssBetween = 0;
  let ssWithin = 0;

  for (let i = 0; i < k; i++) {
    const ni = cleanGroups[i].length;
    const mi = groupMeans[i];
    ssBetween += ni * (mi - grandMean) ** 2;

    for (const val of cleanGroups[i]) {
      ssWithin += (val - mi) ** 2;
    }
  }

  const dfBetween = k - 1;
  const dfWithin = totalN - k;

  const msBetween = dfBetween > 0 ? ssBetween / dfBetween : 0;
  const msWithin = dfWithin > 0 ? ssWithin / dfWithin : 0;

  const f = msWithin > 0 ? msBetween / msWithin : 0;
  const pValue = 1 - fCDF(f, dfBetween, dfWithin);

  return {
    f,
    dfBetween,
    dfWithin,
    pValue: Math.max(0, Math.min(1, pValue)),
    groupMeans,
    grandMean,
    ssBetween,
    ssWithin,
    msBetween,
    msWithin,
  };
}

/** Percentile Bootstrap Confidence Interval */
export function bootstrapCI(
  values: number[],
  statFn?: (arr: number[]) => number,
  B = 2000,
  alpha = 0.05
): BootstrapResult {
  const clean = values.filter(v => !isNaN(v) && isFinite(v));
  const n = clean.length;
  if (n === 0) {
    return { estimate: 0, ciLower: 0, ciUpper: 0, standardError: 0, iterations: B };
  }

  const func = statFn || ((arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length);
  const originalEstimate = func(clean);

  let seed = 12345;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
    return seed / 0x7fffffff;
  };

  const bootEstimates: number[] = [];
  for (let b = 0; b < B; b++) {
    const sample: number[] = [];
    for (let i = 0; i < n; i++) {
      sample.push(clean[Math.floor(random() * n)]);
    }
    bootEstimates.push(func(sample));
  }

  bootEstimates.sort((a, b) => a - b);
  const lowerIdx = Math.max(0, Math.floor((alpha / 2) * B));
  const upperIdx = Math.min(B - 1, Math.floor((1 - alpha / 2) * B));

  const ciLower = bootEstimates[lowerIdx];
  const ciUpper = bootEstimates[upperIdx];

  const bMean = bootEstimates.reduce((s, v) => s + v, 0) / B;
  const bVar = bootEstimates.reduce((s, v) => s + (v - bMean) ** 2, 0) / (B - 1);
  const standardError = Math.sqrt(bVar);

  return {
    estimate: originalEstimate,
    ciLower,
    ciUpper,
    standardError,
    iterations: B,
  };
}

/**
 * Two-Way Factorial ANOVA (Fixed Effects) via General Linear Model.
 *
 * Supports balanced and unbalanced designs, including empty factor cells, by
 * fitting the cell-means saturated model with reference-cell dummy coding and
 * computing sums of squares by model comparison (extra sum of squares).
 *
 * Sums-of-squares conventions:
 *   Type I (sequential):    A, then B adjusted for A, then AB adjusted for A+B.
 *   Type II (hierarchical): A adjusted for B, B adjusted for A, AB adjusted for A+B.
 *   Type III (marginal):    A adjusted for B+AB, B adjusted for A+AB, AB adjusted for A+B.
 *
 * For balanced designs all three types coincide. For unbalanced designs they
 * generally differ; the caller must choose the convention matching their
 * hypothesis. Type III is the default (marginal, coding-invariant tests).
 *
 * An effect is reported as non-estimable (f = null, pValue = null) when its
 * columns add no rank beyond the other terms (e.g. empty cells make a main
 * effect or interaction not independently estimable) or when there is no
 * residual degrees of freedom / zero residual variance. A zero F statistic is
 * never returned in place of a meaningful test.
 */
export interface TwoWayAnovaOptions {
  type?: 'I' | 'II' | 'III';
}

export function twoWayAnova(
  observations: { value: number; levelA: string; levelB: string }[],
  factorALabel = 'Factor A',
  factorBLabel = 'Factor B',
  options: TwoWayAnovaOptions = {}
): TwoWayAnovaResult {
  const type = options.type ?? 'III';
  const warnings: string[] = [];
  const clean = observations.filter(o => isFinite(o.value) && !isNaN(o.value));
  const n = clean.length;

  if (n === 0) {
    return emptyResult(factorALabel, factorBLabel, type, 'No valid observations.', warnings);
  }

  const levelsA = Array.from(new Set(clean.map(o => o.levelA))).sort();
  const levelsB = Array.from(new Set(clean.map(o => o.levelB))).sort();
  if (levelsA.length < 2 || levelsB.length < 2) {
    return emptyResult(
      factorALabel,
      factorBLabel,
      type,
      'Two-Way ANOVA requires at least 2 distinct levels in both Factor A and Factor B.',
      warnings
    );
  }

  const a = levelsA.length;
  const b = levelsB.length;
  const grandMean = clean.reduce((s, o) => s + o.value, 0) / n;

  // Cell means (non-empty cells only).
  const cells = new Map<string, number[]>();
  for (const o of clean) {
    const k = `${o.levelA}__||__${o.levelB}`;
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k)!.push(o.value);
  }
  const cellMeans: TwoWayAnovaResult['cellMeans'] = [];
  for (const [key, vals] of cells) {
    const [la, lb] = key.split('__||__');
    cellMeans.push({ levelA: la, levelB: lb, mean: vals.reduce((s, v) => s + v, 0) / vals.length, n: vals.length });
  }

  // ── Effect (sum-to-zero) coded design matrix ──────────────────────────
  // Effect coding is required so that drop-one-term model comparison yields
  // coding-invariant Type II and Type III sums of squares. With reference
  // (dummy) coding, dropping a main effect while retaining the interaction
  // imposes spurious constraints on cell means and is not Type III.
  // Columns: intercept(0) | A effect [1, a) | B effect [a, a+b) | AB effect [a+b, a*b)
  // Total = 1 + (a-1) + (b-1) + (a-1)(b-1) = a*b.
  const nCol = a * b;
  const aIndex = new Map<string, number>();
  const bIndex = new Map<string, number>();
  levelsA.forEach((l, i) => aIndex.set(l, i));
  levelsB.forEach((l, i) => bIndex.set(l, i));

  // Effect-coded value for level index `lv` in factor with `nLevels` levels,
  // contrast column `j` (0..nLevels-2): +1 if lv==j, -1 if lv==nLevels-1, else 0.
  const effect = (lv: number, j: number, nLevels: number): number => {
    if (lv === j) return 1;
    if (lv === nLevels - 1) return -1;
    return 0;
  };

  const X: number[][] = [];
  const y: number[] = [];
  for (const o of clean) {
    const ia = aIndex.get(o.levelA)!;
    const ib = bIndex.get(o.levelB)!;
    const row: number[] = [1]; // intercept
    for (let j = 0; j < a - 1; j++) row.push(effect(ia, j, a));
    for (let j = 0; j < b - 1; j++) row.push(effect(ib, j, b));
    for (let j = 0; j < a - 1; j++) {
      const aj = effect(ia, j, a);
      for (let k = 0; k < b - 1; k++) row.push(aj * effect(ib, k, b));
    }
    X.push(row);
    y.push(o.value);
  }

  const colsA: number[] = Array.from({ length: a - 1 }, (_, i) => i + 1); // 1..a-1
  const colsB: number[] = Array.from({ length: b - 1 }, (_, i) => a + i); // a..a+b-2
  const colsAB: number[] = [];
  for (let j = 0; j < a - 1; j++) {
    for (let k = 0; k < b - 1; k++) colsAB.push(a + b - 1 + j * (b - 1) + k);
  }

  const colsIntercept: number[] = [0];
  const colsAll: number[] = Array.from({ length: nCol }, (_, i) => i);

  // Model column sets
  const M0 = colsIntercept; // intercept only
  const MA = [...colsIntercept, ...colsA];
  const MB = [...colsIntercept, ...colsB];
  const MAB = [...colsIntercept, ...colsA, ...colsB];
  const MA_AB = [...colsIntercept, ...colsA, ...colsAB]; // A + AB (no B main)  -> Type III B
  const MB_AB = [...colsIntercept, ...colsB, ...colsAB]; // B + AB (no A main)  -> Type III A
  const MFull = colsAll;

  // Cache fits by column set
  const fitCache = new Map<string, { sse: number; rank: number }>();
  const fit = (cols: number[]) => {
    const key = cols.join(',');
    if (fitCache.has(key)) return fitCache.get(key)!;
    const Xsub = X.map(row => cols.map(c => row[c]));
    const res = fitSSE(Xsub, y);
    fitCache.set(key, res);
    return res;
  };

  const f0 = fit(M0);
  const fA = fit(MA);
  const fB = fit(MB);
  const fAB = fit(MAB);
  const fMA_AB = fit(MA_AB);
  const fMB_AB = fit(MB_AB);
  const fFull = fit(MFull);

  const rankFull = fFull.rank;
  const dfError = n - rankFull;
  const msError = dfError > 0 ? fFull.sse / dfError : 0;

  const ssTotal = f0.sse; // SSE(intercept only) = corrected total SS

  const dfA = a - 1;
  const dfB = b - 1;
  const dfAB = (a - 1) * (b - 1);

  const noError = dfError <= 0 || !Number.isFinite(msError) || msError <= 1e-12;
  if (noError) {
    warnings.push(
      'No residual degrees of freedom or zero residual variance; F-tests are not available.'
    );
  }

  const buildEffect = (
    name: string,
    ss: number,
    df: number,
    rankReduced: number
  ): TwoWayAnovaResult['factorA'] => {
    const estimable = rankReduced < rankFull && !noError && df > 0;
    let effectSs = Math.max(0, ss);
    if (!(rankReduced < rankFull)) effectSs = 0; // non-estimable: no unique contribution
    const ms = df > 0 ? effectSs / df : 0;
    let f: number | null = null;
    let pValue: number | null = null;
    let note: string | undefined;
    if (!estimable) {
      note = !(rankReduced < rankFull) ? 'Not estimable from this design (empty cells / rank deficiency).' : 'No residual degrees of freedom or zero residual variance; F undefined.';
    } else {
      f = ms / msError;
      pValue = Math.max(0, Math.min(1, 1 - fCDF(f, df, dfError)));
    }
    return { f, df, ss: effectSs, ms, pValue, estimable, note };
  };

  // ---- Type-specific sums of squares (extra sum of squares) ----
  let resA: TwoWayAnovaResult['factorA'];
  let resB: TwoWayAnovaResult['factorA'];
  let resAB: TwoWayAnovaResult['factorA'];

  if (type === 'I') {
    // Sequential: A, then B|A, then AB|A+B
    resA = buildEffect('A', f0.sse - fA.sse, dfA, f0.rank);
    resB = buildEffect('B', fA.sse - fAB.sse, dfB, fA.rank);
    resAB = buildEffect('A x B', fAB.sse - fFull.sse, dfAB, fAB.rank);
  } else if (type === 'II') {
    // Hierarchical: A|B, B|A, AB|A+B
    resA = buildEffect('A', fB.sse - fAB.sse, dfA, fB.rank);
    resB = buildEffect('B', fA.sse - fAB.sse, dfB, fA.rank);
    resAB = buildEffect('A x B', fAB.sse - fFull.sse, dfAB, fAB.rank);
  } else {
    // Type III (default): A|B+AB, B|A+AB, AB|A+B
    resA = buildEffect('A', fMB_AB.sse - fFull.sse, dfA, fMB_AB.rank);
    resB = buildEffect('B', fMA_AB.sse - fFull.sse, dfB, fMA_AB.rank);
    resAB = buildEffect('A x B', fAB.sse - fFull.sse, dfAB, fAB.rank);
  }

  return {
    factorALabel,
    factorBLabel,
    type,
    factorA: resA,
    factorB: resB,
    interaction: resAB,
    within: { df: dfError, ss: fFull.sse, ms: msError },
    grandMean,
    cellMeans,
    rank: rankFull,
    dfError,
    dfModel: rankFull > 0 ? rankFull - 1 : 0,
    ssTotal,
    error: null,
    warnings,
  };
}

/** Fit a linear model y ~ X by pseudo-inverse (SVD); returns residual SS and numerical rank. */
function fitSSE(X: number[][], y: number[]): { sse: number; rank: number } {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  if (n === 0 || p === 0) return { sse: 0, rank: 0 };
  const Xm = new Matrix(X);
  const ym = new Matrix(y.map(v => [v]));
  const beta = pseudoInverse(Xm).mmul(ym);
  const yhat = Xm.mmul(beta);
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const r = y[i] - yhat.get(i, 0);
    sse += r * r;
  }
  const svd = new SVD(Xm);
  const smax = svd.diagonal[0] ?? 0;
  const tol = Math.max(n, p) * Number.EPSILON * Math.max(smax, 1e-12);
  let rank = 0;
  for (let k = 0; k < svd.diagonal.length; k++) {
    if (svd.diagonal[k] > tol) rank++;
  }
  return { sse: Math.max(0, sse), rank };
}

function emptyResult(
  factorALabel: string,
  factorBLabel: string,
  type: 'I' | 'II' | 'III',
  msg: string,
  warnings: string[]
): TwoWayAnovaResult {
  const nullEffect: TwoWayAnovaResult['factorA'] = {
    f: null,
    df: 0,
    ss: 0,
    ms: 0,
    pValue: null,
    estimable: false,
    note: msg,
  };
  return {
    factorALabel,
    factorBLabel,
    type,
    factorA: nullEffect,
    factorB: nullEffect,
    interaction: nullEffect,
    within: { df: 0, ss: 0, ms: 0 },
    grandMean: 0,
    cellMeans: [],
    rank: 0,
    dfError: 0,
    dfModel: 0,
    ssTotal: 0,
    error: msg,
    warnings,
  };
}
