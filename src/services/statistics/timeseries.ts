import { SeasonalMKResult, ChangePointResult, GEVResult } from '../../types/earthScience';
import { normalCdf, normalPValue } from './distributions';

// ── Canonical tie handling for Mann-Kendall family statistics ─────────────

/**
 * Relative tolerance for treating two floating-point observations as ties.
 * Two values a and b are ties when |a - b| <= MANN_KENDALL_TIE_TOLERANCE *
 * max(1, |a|, |b|). The max(1, ...) floor keeps an absolute tolerance near
 * zero so very small measurements are not spuriously split, while scaling
 * with magnitude for large measurements. No arbitrary rounding is applied to
 * the measurements; ties are determined solely by this tolerance.
 */
export const MANN_KENDALL_TIE_TOLERANCE = 1e-10;

/**
 * Canonical tie grouping. Values are sorted and single-linkage clustered:
 * consecutive sorted values within the relative tolerance share one tie group.
 * Returns, for each original position, its canonical tie group id plus the
 * size of each group. The group id is the single source of truth used by both
 * the S statistic and the tie-corrected variance, guaranteeing that S and
 * varS agree on exactly which observations are ties.
 */
export function computeTieGroups(
  values: number[],
  tieTolerance: number = MANN_KENDALL_TIE_TOLERANCE
): { groupIds: number[]; groupSizes: number[] } {
  const n = values.length;
  const groupIds = new Array<number>(n);
  const groupSizes: number[] = [];

  const indexed = values.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);

  let currentGroup = 0;
  let groupStart = 0;

  for (let i = 0; i < indexed.length; i++) {
    if (i > 0) {
      const prev = indexed[i - 1].v;
      const curr = indexed[i].v;
      const scale = Math.max(1.0, Math.abs(prev), Math.abs(curr));
      if (Math.abs(curr - prev) > tieTolerance * scale) {
        groupSizes.push(i - groupStart);
        currentGroup++;
        groupStart = i;
      }
    }
    groupIds[indexed[i].i] = currentGroup;
  }
  groupSizes.push(indexed.length - groupStart);

  return { groupIds, groupSizes };
}

/**
 * Sign of (a - b) using canonical tie groups.
 * Returns 0 when a and b are in the same tie group, +1 when a > b, -1 when a < b.
 */
export function tieSign(a: number, b: number, groupIdA: number, groupIdB: number): number {
  if (groupIdA === groupIdB) return 0;
  return a > b ? 1 : -1;
}

/**
 * Standard Seasonal Mann-Kendall test (Hirsch, Slack & Smith 1982)
 * Includes exact centered detrended Hamed-Rao (1998) variance modification and
 * true Yue et al. (2002) Trend-Free Pre-Whitening (TFPW).
 * 
 * Critical fix: preserves original time indices throughout autocorrelation calculations.
 * Never calculates lag-1 correlation across a missing temporal interval as though it were
 * a consecutive observation. Validates regular-spacing assumption and warns when violated.
 */
export function seasonalMannKendall(
  values: number[],
  seasonLength = 12,
  method: 'standard' | 'hamed_rao' | 'tfpw' = 'standard',
  tieTolerance: number = MANN_KENDALL_TIE_TOLERANCE
): SeasonalMKResult {
  const n = values.length;
  const nYears = Math.floor(n / seasonLength);

  if (nYears < 3) {
    return {
      s: 0,
      varS: 1,
      z: 0,
      pValue: 1,
      tau: 0,
      method,
      seasonCount: seasonLength,
    };
  }

  // ── Helper: compute mean ─────────────────────────────────────────────
  const meanArr = (arr: number[]): number =>
    arr.reduce((s, v) => s + v, 0) / arr.length;

  // ── Helper: compute lag-1 autocorrelation on consecutive observations only ─────────────
  // Only correlates y[t] and y[t+1] when both exist AND year difference is exactly 1.
  // Returns 0 if fewer than 2 consecutive pairs available.
  const lag1AutocorrelationConsecutive = (values: number[], years: number[]): number => {
    if (values.length < 3) return 0;
    const m = meanArr(values);
    let num = 0;
    let den = 0;
    let consecutivePairs = 0;
    for (let i = 1; i < values.length; i++) {
      const yrDiff = years[i] - years[i - 1];
      if (yrDiff === 1) {
        consecutivePairs++;
        num += (values[i] - m) * (values[i - 1] - m);
        den += (values[i - 1] - m) ** 2;
      }
    }
    if (consecutivePairs < 2 || den === 0) return 0;
    return num / den;
  };

  // ── Helper: check if a seasonal series has regular (no-gap) spacing ─────────────
  const hasRegularSeasonalSpacing = (years: number[]): boolean => {
    if (years.length < 2) return true;
    for (let i = 1; i < years.length; i++) {
      const diff = years[i] - years[i - 1];
      const validDiff = years[i] > years[i - 1] ? (diff === 0 || diff === 1) : false;
      if (!validDiff) return false;
    }
    return true;
  };

  // ── Yue et al. (2002) Trend-Free Pre-Whitening (TFPW) ─────────────────────
  let workingValues = values;
  let tfpwR1Diagnostic: number | undefined; // lag-1 r value, NOT a variance multiplier
  let tfpwHasIrregularSpacing = false;
  let tfpwInsufficientPairs = false;
  const warnings: string[] = [];
  if (method === 'tfpw') {
    const whitened = [...values];
    const r1SeasonEstimates: number[] = [];
    const r1ConsecutivePairsCount: number[] = [];

    for (let m = 0; m < seasonLength; m++) {
      // Collect valid observations with their original year indices
      const seasonObs: { year: number; idx: number; value: number }[] = [];
      for (let y = 0; y < nYears; y++) {
        const idx = y * seasonLength + m;
        if (idx < n && !isNaN(values[idx]) && isFinite(values[idx])) {
          seasonObs.push({ year: y, idx, value: values[idx] });
        }
      }
      if (seasonObs.length < 4) continue;

      const seasonVals = seasonObs.map(o => o.value);
      const seasonYears = seasonObs.map(o => o.year);

      // Check seasonal spacing regularity
      const hasGaps = !hasRegularSeasonalSpacing(seasonYears);
      if (hasGaps) {
        tfpwHasIrregularSpacing = true;
        warnings.push(`TFPW: season ${m + 1} contains missing months and irregular temporal spacing; pre-whitening skipped for this season.`);
      }

      // 1. Estimate season monotonic trend via Sen slope & detrend
      // Use original year indices to preserve temporal spacing in trend estimation
      const beta = senSlopeExact(seasonVals, seasonYears);
      const detrended = seasonVals.map((v, k) => v - beta * seasonYears[k]);

      // 2. Estimate lag-1 autocorrelation respecting temporal spacing
      // ONLY use observations where consecutive years both exist (year diff = 1)
      const r1 = lag1AutocorrelationConsecutive(detrended, seasonYears);
      r1SeasonEstimates.push(r1);

      // Track how many consecutive pairs were found for diagnostics
      let consecutivePairs = 0;
      for (let k = 1; k < seasonObs.length; k++) {
        if (seasonObs[k].year - seasonObs[k - 1].year === 1) consecutivePairs++;
      }
      r1ConsecutivePairsCount.push(consecutivePairs);
      if (consecutivePairs < 2) tfpwInsufficientPairs = true;

      // 3. If lag-1 autocorrelation exceeds white-noise bound, pre-whiten & re-blend trend
      // Only apply if we have sufficient consecutive pairs for a valid r1 estimate
      if (consecutivePairs >= 2 && Math.abs(r1) > 1.96 / Math.sqrt(detrended.length)) {
        // Pre-whiten: y*_t = y_t - r1 * y_{t-1} for consecutive observations
        // First observation (no predecessor): y*_1 = y_1 - trend_at_year_1
        whitened[seasonObs[0].idx] = detrended[0] + beta * seasonObs[0].year;
        for (let k = 1; k < seasonObs.length; k++) {
          // Only pre-whiten if the predecessor exists and is consecutive in time
          if (seasonObs[k].year - seasonObs[k - 1].year === 1) {
            const pwResidual = detrended[k] - r1 * detrended[k - 1];
            whitened[seasonObs[k].idx] = pwResidual + beta * seasonObs[k].year;
          } else {
            // Non-consecutive: keep original value with trend re-blended
            whitened[seasonObs[k].idx] = detrended[k] + beta * seasonObs[k].year;
          }
        }
      } else if (consecutivePairs < 2) {
        // Not enough consecutive pairs for valid lag-1 estimate;
        // skip TFPW pre-whitening for this season, keep original values
        // (workingValues remains unchanged for this season's indices)
      }
    }

    workingValues = whitened;
    // Use only seasons with sufficient consecutive pairs for the diagnostic
    const validR1Estimates = r1SeasonEstimates.filter(
      (_, i) => r1ConsecutivePairsCount[i] >= 2
    );
    if (validR1Estimates.length > 0) {
      tfpwR1Diagnostic =
        validR1Estimates.reduce((s, v) => s + v, 0) / validR1Estimates.length;
    }
  }

  if (method === 'tfpw' && tfpwHasIrregularSpacing) {
    warnings.push(
      'TFPW: irregular seasonal spacing detected. Lag-1 autocorrelation is computed only on consecutive-year observations; missing months are never treated as adjacent.'
    );
  }
  if (method === 'tfpw' && tfpwInsufficientPairs) {
    warnings.push(
      'TFPW: some seasons have fewer than 2 consecutive-year pairs; lag-1 autocorrelation is unreliable for those seasons and pre-whitening is skipped.'
    );
  }
  if (method === 'hamed_rao' && nYears > 3) {
    const validObsCount = workingValues.filter(v => !isNaN(v) && isFinite(v)).length;
    if (validObsCount < 6) {
      warnings.push('Hamed-Rao: insufficient valid observations (<6) for variance correction.');
    }
  }

  let totalS = 0;
  let totalVarS = 0;
  let totalPairs = 0;

  for (let m = 0; m < seasonLength; m++) {
    const seasonValues: number[] = [];
    for (let y = 0; y < nYears; y++) {
      const idx = y * seasonLength + m;
      if (idx < n && !isNaN(workingValues[idx]) && isFinite(workingValues[idx])) {
        seasonValues.push(workingValues[idx]);
      }
    }

    const ns = seasonValues.length;
    if (ns < 2) continue;

    const { groupIds, groupSizes } = computeTieGroups(seasonValues, tieTolerance);

    let s = 0;
    for (let i = 0; i < ns - 1; i++) {
      const gi = groupIds[i];
      for (let j = i + 1; j < ns; j++) {
        s += tieSign(seasonValues[j], seasonValues[i], groupIds[j], gi);
      }
    }
    totalS += s;

    // Tied groups variance calculation (same canonical tie definition as S).
    let tieTerm = 0;
    for (const size of groupSizes) {
      if (size > 1) tieTerm += size * (size - 1) * (2 * size + 5);
    }

    const varS = Math.max(0, (ns * (ns - 1) * (2 * ns + 5) - tieTerm) / 18);
    totalVarS += varS;
    totalPairs += (ns * (ns - 1)) / 2;
  }

  let autoCorrection = 1.0;
  let hamedRaoRhoDiagnostic: number[] = []; // per-lag rho values for diagnostics

  // Hamed & Rao (1998): centered detrended rank autocorrelation variance factor.
  //
  // Each valid observation is keyed by its ORIGINAL ARRAY INDEX, which encodes
  // both year (Math.floor(i / seasonLength)) and season (i % seasonLength).
  // Using the year alone as a key (the previous approach) caused observations
  // from different seasons in the same year to overwrite one another, and it
  // paired observations at "year + lag" regardless of season -- e.g. January
  // could be correlated with February merely because they fall in adjacent
  // years.
  //
  // Here an observation at index `i` is paired only with the observation of
  // the SAME season exactly `lag` years later, at index `i + lag * seasonLength`.
  // Missing observations are absent from the index map, so no spurious pair is
  // created across a gap, and actual temporal spacing is preserved.
  if (method === 'hamed_rao' && nYears > 3) {
    const validObs: {
      year: number;
      season: number;
      idx: number;
      value: number;
    }[] = [];
    for (let i = 0; i < workingValues.length; i++) {
      if (!isNaN(workingValues[i]) && isFinite(workingValues[i])) {
        validObs.push({
          year: Math.floor(i / seasonLength),
          season: i % seasonLength,
          idx: i,
          value: workingValues[i],
        });
      }
    }
    const mLen = validObs.length;
    if (mLen >= 6) {
      // Detrend using original year indices
      const years = validObs.map((o) => o.year);
      const valuesArr = validObs.map((o) => o.value);
      const overallBeta = senSlopeExact(valuesArr, years);
      const detrended = validObs.map((o) => o.value - overallBeta * o.year);
      const ranks = rankArray(detrended);
      const meanRank = ranks.reduce((a, b) => a + b, 0) / mLen;
      const centered = ranks.map((r) => r - meanRank);
      const denom = centered.reduce((s, r) => s + r * r, 0);

      if (denom > 1e-12) {
        let correctionSum = 0;
        const maxLag = Math.min(12, Math.floor(mLen / 4));
        const crit = 1.96 / Math.sqrt(mLen);

        // Map from original array index (year + season) to position in validObs.
        const indexToPosition = new Map<number, number>();
        validObs.forEach((o, pos) => indexToPosition.set(o.idx, pos));

        for (let lag = 1; lag <= maxLag; lag++) {
          let num = 0;
          let pairCount = 0;
          // Same season, exactly `lag` years apart.
          const step = lag * seasonLength;
          for (let i = 0; i < mLen; i++) {
            const targetIdx = validObs[i].idx + step;
            const j = indexToPosition.get(targetIdx);
            if (j !== undefined) {
              num += centered[i] * centered[j];
              pairCount++;
            }
          }
          const rho = pairCount > 0 ? num / denom : 0;
          hamedRaoRhoDiagnostic.push(rho);
          // Weight by the actual number of same-season pairs at this lag.
          if (Math.abs(rho) > crit && pairCount >= 2) {
            correctionSum +=
              pairCount * (pairCount - 1) * (pairCount - 2) * rho;
          }
        }

        const nOverNStar =
          1 + (2 / (mLen * (mLen - 1) * (mLen - 2))) * correctionSum;
        autoCorrection = Math.max(0.5, Math.min(6.0, nOverNStar));
        totalVarS *= autoCorrection;
      }
    }
  }

  const sqrtVar = Math.sqrt(Math.max(1e-9, totalVarS));
  let z = 0;
  if (totalS > 0) z = (totalS - 1) / sqrtVar;
  else if (totalS < 0) z = (totalS + 1) / sqrtVar;

  const tau = totalPairs > 0 ? totalS / totalPairs : 0;
  const pValue = normalPValue(z);

  // Separate diagnostics from variance correction
  const diagnostics = {
    tfpwR1: method === 'tfpw' && tfpwR1Diagnostic !== undefined
      ? tfpwR1Diagnostic
      : undefined,
    hamedRaoRhoDiagnostic: method === 'hamed_rao' && hamedRaoRhoDiagnostic.length > 0
      ? hamedRaoRhoDiagnostic
      : undefined,
  };

  return {
    s: totalS,
    varS: totalVarS,
    z,
    pValue,
    tau,
    method,
    seasonCount: seasonLength,
    // autocorrelationCorrection is a variance correction factor, not a correlation
    autocorrelationCorrection:
      method === 'hamed_rao'
        ? autoCorrection !== 1.0
          ? autoCorrection
          : undefined
        : undefined, // TFPW does not have a variance correction factor; it pre-whitens
    ...diagnostics,
    warnings: warnings.length > 0 ? warnings : undefined,
  };
}



/** Hamed-Rao modified Mann-Kendall */
export function hamedRaoCorrection(values: number[], seasonLength = 12): SeasonalMKResult {
  return seasonalMannKendall(values, seasonLength, 'hamed_rao');
}

/** Trend-Free Pre-Whitening (TFPW) Mann-Kendall */
export function tfpwMannKendall(values: number[], seasonLength = 12): SeasonalMKResult {
  return seasonalMannKendall(values, seasonLength, 'tfpw');
}

/** Exact Sen's slope without subsampling */
export function senSlopeExact(values: number[], times?: number[]): number {
  const n = values.length;
  if (n < 2) return 0;

  const t = times || Array.from({ length: n }, (_, i) => i);
  const slopes: number[] = [];

  for (let i = 0; i < n; i++) {
    if (isNaN(values[i])) continue;
    for (let j = i + 1; j < n; j++) {
      if (isNaN(values[j])) continue;
      const dt = t[j] - t[i];
      if (dt > 1e-6) {
        slopes.push((values[j] - values[i]) / dt);
      }
    }
  }

  if (slopes.length === 0) return 0;
  slopes.sort((a, b) => a - b);
  const mid = Math.floor(slopes.length / 2);
  return slopes.length % 2 === 0 ? (slopes[mid - 1] + slopes[mid]) / 2 : slopes[mid];
}

/** Trend acceleration and reversal detection */
export function trendAcceleration(values: number[]): {
  acceleration: number;
  isAccelerating: boolean;
  reversalPoint?: number;
} {
  const n = values.length;
  if (n < 6) return { acceleration: 0, isAccelerating: false };

  const half = Math.floor(n / 2);
  const firstHalf = values.slice(0, half);
  const secondHalf = values.slice(half);

  const slope1 = senSlopeExact(firstHalf);
  const slope2 = senSlopeExact(secondHalf);

  const acceleration = slope2 - slope1;
  const isAccelerating = Math.abs(acceleration) > 0.05 * (Math.abs(slope1) + 1e-4);

  // Check for slope sign reversal
  let reversalPoint: number | undefined;
  if ((slope1 > 0 && slope2 < 0) || (slope1 < 0 && slope2 > 0)) {
    reversalPoint = half;
  }

  return { acceleration, isAccelerating, reversalPoint };
}

/** Pettitt's non-parametric change-point test */
export function pettittChangePoint(
  values: number[],
  dates?: string[],
  tieTolerance: number = MANN_KENDALL_TIE_TOLERANCE
): ChangePointResult {
  const n = values.length;
  if (n < 4) {
    return {
      method: 'Pettitt',
      changePointIndex: 0,
      changePointDate: dates?.[0] || 'index_0',
      testStatistic: 0,
      pValue: 1,
      significanceLevel: 'Not Significant',
      meanBefore: 0,
      meanAfter: 0,
      shift: 0,
    };
  }

  // Canonical tie grouping: same tie definition as Mann-Kendall S/variance.
  const { groupIds } = computeTieGroups(values, tieTolerance);

  let maxK = 0;
  let cpIndex = 0;
  let runningU = 0;

  for (let t = 1; t < n; t++) {
    let count = 0;
    const gt = groupIds[t - 1];
    for (let j = t; j < n; j++) {
      count += tieSign(values[j], values[t - 1], groupIds[j], gt);
    }
    runningU += count;
    const absU = Math.abs(runningU);
    if (absU > maxK) {
      maxK = absU;
      cpIndex = t;
    }
  }

  // Pettitt approximate p-value
  const pValue = Math.min(1.0, 2 * Math.exp((-6 * maxK * maxK) / (n * n * n + n * n)));

  const before = values.slice(0, cpIndex);
  const after = values.slice(cpIndex);
  const meanBefore = before.reduce((s, v) => s + v, 0) / before.length;
  const meanAfter = after.reduce((s, v) => s + v, 0) / after.length;
  const shift = meanAfter - meanBefore;

  let significanceLevel = 'Not Significant';
  if (pValue < 0.001) significanceLevel = 'Extremely Significant';
  else if (pValue < 0.01) significanceLevel = 'Highly Significant';
  else if (pValue < 0.05) significanceLevel = 'Significant';
  else if (pValue < 0.10) significanceLevel = 'Marginally Significant';

  return {
    method: 'Pettitt',
    changePointIndex: cpIndex,
    changePointDate: dates?.[cpIndex] || `index_${cpIndex}`,
    testStatistic: maxK,
    pValue,
    significanceLevel,
    meanBefore,
    meanAfter,
    shift,
  };
}

/** CUSUM change point detection */
export function cusumChangePoint(values: number[], dates?: string[]): ChangePointResult {
  const n = values.length;
  if (n < 4) {
    return {
      method: 'CUSUM',
      changePointIndex: 0,
      changePointDate: dates?.[0] || 'index_0',
      testStatistic: 0,
      pValue: 1,
      significanceLevel: 'Not Significant',
      meanBefore: 0,
      meanAfter: 0,
      shift: 0,
    };
  }

  const mean = values.reduce((s, v) => s + v, 0) / n;
  let maxAbsC = 0;
  let cpIndex = 0;
  let cumsum = 0;

  for (let i = 0; i < n; i++) {
    cumsum += values[i] - mean;
    if (Math.abs(cumsum) > maxAbsC) {
      maxAbsC = Math.abs(cumsum);
      cpIndex = i;
    }
  }

  const s = Math.sqrt(values.reduce((ss, v) => ss + (v - mean) ** 2, 0) / (n - 1));
  const testStat = s > 0 ? maxAbsC / (s * Math.sqrt(n)) : 0;
  const pValue = Math.min(1.0, 2 * (1 - normalCdf(testStat)));

  const before = values.slice(0, cpIndex + 1);
  const after = values.slice(cpIndex + 1);
  const meanBefore = before.reduce((ss, v) => ss + v, 0) / (before.length || 1);
  const meanAfter = after.reduce((ss, v) => ss + v, 0) / (after.length || 1);

  return {
    method: 'CUSUM',
    changePointIndex: cpIndex,
    changePointDate: dates?.[cpIndex] || `index_${cpIndex}`,
    testStatistic: testStat,
    pValue,
    significanceLevel: pValue < 0.05 ? 'Significant' : 'Not Significant',
    meanBefore,
    meanAfter,
    shift: meanAfter - meanBefore,
  };
}

/** Shannon Entropy of numeric series using histogram binning */
export function shannonEntropy(values: number[], bins = 10): number {
  const clean = values.filter(v => !isNaN(v) && isFinite(v));
  if (clean.length === 0) return 0;

  const min = Math.min(...clean);
  const max = Math.max(...clean);
  if (max === min) return 0;

  const counts = new Array(bins).fill(0);
  const binWidth = (max - min) / bins;

  for (const v of clean) {
    const idx = Math.min(bins - 1, Math.floor((v - min) / binWidth));
    counts[idx]++;
  }

  let entropy = 0;
  const total = clean.length;
  for (const c of counts) {
    if (c > 0) {
      const p = c / total;
      entropy -= p * Math.log2(p);
    }
  }

  return entropy;
}

/** Sample Entropy (SampEn) for time-series regularity */
export function sampleEntropy(values: number[], m = 2, r?: number): number {
  const clean = values.filter(v => !isNaN(v) && isFinite(v));
  const n = clean.length;
  if (n < m + 2) return 0;

  const std = Math.sqrt(clean.reduce((s, v, _, arr) => {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return s + (v - mean) ** 2;
  }, 0) / (n - 1));

  const tolerance = r !== undefined ? r : 0.2 * std;

  const countMatches = (dim: number): number => {
    let matches = 0;
    const total = n - dim;
    for (let i = 0; i < total; i++) {
      for (let j = i + 1; j < total; j++) {
        let maxDiff = 0;
        for (let k = 0; k < dim; k++) {
          const diff = Math.abs(clean[i + k] - clean[j + k]);
          if (diff > maxDiff) maxDiff = diff;
        }
        if (maxDiff <= tolerance) matches++;
      }
    }
    return matches;
  };

  const A = countMatches(m + 1);
  const B = countMatches(m);

  if (B === 0 || A === 0) return 0;
  return -Math.log(A / B);
}

// ── Helpers ──────────────────────────────────────────────────────────

function rankArray(values: number[]): number[] {
  const indexed = values.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  const ranks = new Array(values.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length && indexed[j].v === indexed[i].v) j++;
    const avgRank = (i + j - 1) / 2 + 1;
    for (let k = i; k < j; k++) {
      ranks[indexed[k].i] = avgRank;
    }
    i = j;
  }
  return ranks;
}

/**
 * Hosking (1990) L-Moments Generalized Extreme Value (GEV) Block-Maxima Fit
 * Fits location (mu), scale (sigma), and shape (xi = -kappa) with return levels.
 */
export function fitGEV(
  values: number[],
  blockSize = 12,
  returnPeriods: number[] = [2, 5, 10, 20, 50, 100]
): GEVResult {
  const blockMaxima: number[] = [];
  for (let i = 0; i + blockSize <= values.length; i += blockSize) {
    const slice = values.slice(i, i + blockSize).filter(v => !isNaN(v) && isFinite(v));
    if (slice.length >= Math.max(2, Math.floor(blockSize / 3))) {
      blockMaxima.push(Math.max(...slice));
    }
  }

  if (blockMaxima.length < 5) {
    const clean = values.filter(v => !isNaN(v) && isFinite(v));
    const step = Math.max(2, Math.floor(clean.length / 8));
    blockMaxima.length = 0;
    for (let i = 0; i + step <= clean.length; i += step) {
      blockMaxima.push(Math.max(...clean.slice(i, i + step)));
    }
  }

  const x = [...blockMaxima].sort((a, b) => a - b);
  const n = x.length;
  if (n < 3) {
    return {
      shapeXi: 0,
      locationMu: x[0] ?? 0,
      scaleSigma: 1,
      family: 'Gumbel (light-tailed)',
      blockMaxima,
      returnLevels: returnPeriods.map(p => ({ period: p, level: x[0] ?? 0 })),
    };
  }

  // Unbiased Probability Weighted Moments b0, b1, b2 (Hosking & Wallis 1997)
  let b0 = 0;
  let b1 = 0;
  let b2 = 0;
  for (let j = 0; j < n; j++) {
    b0 += x[j];
    b1 += (j / (n - 1)) * x[j];
    b2 += ((j * (j - 1)) / ((n - 1) * (n - 2))) * x[j];
  }
  b0 /= n;
  b1 /= n;
  b2 /= n;

  const l1 = b0;
  const l2 = 2 * b1 - b0;
  const l3 = 6 * b2 - 6 * b1 + b0;
  const t3 = l2 > 1e-12 ? l3 / l2 : 0;

  // Hosking rational approximation for GEV shape kappa (xi = -kappa)
  const c = (2 / Math.max(1e-6, 3 + t3)) - Math.log(2) / Math.log(3);
  const kappa = 7.859 * c + 2.9554 * c * c;
  const xi = Math.max(-0.5, Math.min(0.5, -kappa));

  // Gamma approximation via Lanczos or Euler constant fallback for kappa near 0
  const gammaApprox = (z: number): number => {
    const g = 7;
    const p = [
      0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    let zc = z - 1;
    let xSum = p[0];
    for (let i = 1; i < g + 2; i++) xSum += p[i] / (zc + i);
    const t = zc + g + 0.5;
    return Math.sqrt(2 * Math.PI) * Math.pow(t, zc + 0.5) * Math.exp(-t) * xSum;
  };

  let scaleSigma: number;
  let locationMu: number;
  if (Math.abs(kappa) < 1e-4) {
    scaleSigma = Math.max(1e-6, l2 / Math.log(2));
    locationMu = l1 - 0.5772156649 * scaleSigma;
  } else {
    const g1 = gammaApprox(1 + kappa);
    scaleSigma = Math.max(1e-6, (l2 * kappa) / ((1 - Math.pow(2, -kappa)) * g1));
    locationMu = l1 - (scaleSigma * (1 - g1)) / kappa;
  }

  const returnLevels = returnPeriods.map(T => {
    const p = 1 - 1 / Math.max(1.01, T);
    const yP = -Math.log(p);
    const level =
      Math.abs(xi) < 1e-4
        ? locationMu - scaleSigma * Math.log(yP)
        : locationMu - (scaleSigma / xi) * (1 - Math.pow(yP, -xi));
    return { period: T, level };
  });

  const family: GEVResult['family'] =
    xi > 0.05
      ? 'Frechet (heavy-tailed)'
      : xi < -0.05
        ? 'Weibull (bounded)'
        : 'Gumbel (light-tailed)';

  return {
    shapeXi: xi,
    locationMu,
    scaleSigma,
    family,
    blockMaxima,
    returnLevels,
  };
}
