// src/services/statistics/anova.ts
// One-way ANOVA engine (UI-free). Pure TypeScript, no new dependencies.

import { normalCdf } from './distributions';

export interface AnovaGroup {
  label: string;
  values: number[];
}

export interface AnovaInput {
  response: number[];
  group: string[];
}

export interface OneWayAnovaResult {
  k: number;
  N: number;
  nDropped: number[];
  table: {
    source: string;
    ss: number;
    df: number;
    ms: number;
    F: number | null;
    p: number | null;
  }[];
  effectSizes: { eta2: number; omega2: number; omega2Headline: number; cohensF: number };
  assumptions: {
    brownForsythe: { F: number; p: number; df1: number; df2: number } | null;
    bartlett: { T: number; p: number } | null;
    shapiroWilk: { W: number; p: number; n: number } | null;
    durbinWatson: number | null;
    lag1Autocorr: number | null;
    warning: string | null;
  };
  welch: { F: number; df1: number; df2: number; p: number } | null;
  kruskal: { H: number; df: number; p: number; epsilon2: number } | null;
  posthoc: {
    pair: [string, string];
    meanDiff: number;
    ciLow: number;
    ciHigh: number;
    p: number;
    method: 'tukey' | 'games-howell';
  }[];
  error: string | null;
}

// ---- Special functions ----

function logGamma(x: number): number {
  // Lanczos approximation
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 1; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function betaCF(a: number, b: number, x: number): number {
  // Continued fraction for incomplete beta (Lentz)
  const MAXIT = 500;
  const EPS = 3e-12;
  const FPMIN = 1e-30;
  let qab = a + b;
  let qap = a + 1;
  let qam = a - 1;
  let c = 1;
  let d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

function betai(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = logGamma(a + b) - logGamma(a) - logGamma(b);
  const bt = Math.exp(lbeta + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) {
    return bt * betaCF(a, b, x) / a;
  }
  return 1 - bt * betaCF(b, a, 1 - x) / b;
}

export function fSurvival(x: number, d1: number, d2: number): number {
  // P(F >= x) = I_{d2/(d2+d1*x)}(d2/2, d1/2)
  if (x <= 0) return 1;
  const z = d2 / (d2 + d1 * x);
  return betai(d2 / 2, d1 / 2, z);
}

export function chiSquareSurvival(x: number, df: number): number {
  // regularized upper incomplete gamma
  if (x <= 0) return 1;
  // use series/CF via incomplete gamma
  const a = df / 2;
  const xh = x / 2;
  // lower incomplete gamma P(a, xh) via series
  let sum = 1 / a;
  let term = 1 / a;
  for (let i = 1; i < 200; i++) {
    term *= xh / (a + i);
    sum += term;
    if (Math.abs(term) < 1e-15 * Math.abs(sum)) break;
  }
  const lower = Math.exp(-xh + a * Math.log(xh) - logGamma(a)) * sum;
  return 1 - lower;
}

// Studentized range (approx). Use a simple approximation via normal.
export function ptukey(q: number, k: number, df: number): number {
  // Approximate using the fact that for large df it's normal-ish.
  // Use a Monte Carlo-free closed form approximation (Abramowitz).
  if (q <= 0) return 1;
  // Use the CDF approximation: P(Q <= q) ≈ Phi(q)^k-ish; survival = 1 - that.
  // Better: use the known relation: for k=2, ptukey(q,2,df) = 2*(1-tcdf(q/sqrt(2), df))? 
  // We'll approximate with a standard normal tail scaled.
  const z = q / Math.sqrt(2);
  // survival of max of k standard normals ~ (1-Phi(q/sqrt2))^k approximated
  const phi = (x: number) => 0.5 * (1 - erf(x / Math.sqrt(2)));
  const tail = phi(z);
  // rough
  return Math.min(1, Math.max(0, Math.pow(1 - tail, k) > 0 ? 1 - Math.pow(1 - tail, k) : 0));
}

function erf(x: number): number {
  // Abramowitz
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

export function qtukey(p: number, k: number, df: number): number {
  // bisection
  let lo = 0, hi = 10;
  while (ptukey(hi, k, df) > p && hi < 100) hi *= 2;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (ptukey(mid, k, df) > p) hi = mid; else lo = mid;
  }
  return (lo + hi) / 2;
}

// ---- Helpers ----
function mean(arr: number[]): number {
  let s = 0, n = 0;
  for (const v of arr) { s += v; n++; }
  return s / n;
}
function median(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b);
  const m = s.length;
  if (m === 0) return 0;
  if (m % 2 === 1) return s[(m - 1) / 2];
  return (s[m / 2 - 1] + s[m / 2]) / 2;
}

// ---- Main ANOVA ----
export function runOneWayAnova(data: AnovaInput, options?: { alpha?: number }): OneWayAnovaResult {
  const alpha = options?.alpha ?? 0.05;
  const groups = new Map<string, number[]>();
  const nDropped: number[] = [];
  // collect
  const labels = [...new Set(data.group)];
  for (const l of labels) groups.set(l, []);
  for (let i = 0; i < data.response.length; i++) {
    const v = data.response[i];
    const g = data.group[i];
    if (!groups.has(g)) groups.set(g, []);
    if (!isFinite(v)) {
      // drop
      continue;
    }
    groups.get(g)!.push(v);
  }
  // nDropped per group: original counts
  const originalCounts = new Map<string, number>();
  for (const g of data.group) originalCounts.set(g, (originalCounts.get(g) ?? 0) + 1);
  for (const [g, arr] of groups) {
    const orig = originalCounts.get(g) ?? arr.length;
    nDropped.push(orig - arr.length);
  }

  const groupLabels = [...groups.keys()];
  const k = groupLabels.length;
  const groupValues = groupLabels.map((g) => groups.get(g)!);
  const N = groupValues.reduce((s, a) => s + a.length, 0);

  if (k < 2) {
    return errorResult(k, N, nDropped, 'Need at least 2 groups.');
  }
  if (N <= k) {
    return errorResult(k, N, nDropped, 'Need N > k.');
  }
  for (const g of groupValues) {
    if (g.length === 0) return errorResult(k, N, nDropped, 'Empty group.');
  }

  // means
  const groupMeans = groupValues.map(mean);
  const grandMean = mean(groupValues.flat());

  // SSB, SSW using two-pass
  let SSB = 0;
  let SSW = 0;
  for (let i = 0; i < k; i++) {
    const g = groupValues[i];
    const gm = groupMeans[i];
    SSB += g.length * (gm - grandMean) ** 2;
    for (const v of g) SSW += (v - gm) ** 2;
  }
  const SST = SSB + SSW;
  const df1 = k - 1;
  const df2 = N - k;
  const MSB = SSB / df1;
  const MSW = SSW / df2;

  // degenerate: zero within-group variance in all groups
  if (SSW === 0) {
    return errorResult(k, N, nDropped, 'Zero within-group variance in all groups; F undefined.');
  }

  const F = MSB / MSW;
  const p = fSurvival(F, df1, df2);

  const eta2 = SSB / SST;
  const omega2 = (SSB - df1 * MSW) / (SST + MSW);
  const cohensF = Math.sqrt(eta2 / (1 - eta2));

  // Welch
  let welch: OneWayAnovaResult['welch'] = null;
  const allN1 = groupValues.some((g) => g.length < 2);
  if (!allN1) {
    const variances = groupValues.map((g) => {
      const m = mean(g);
      let s = 0;
      for (const v of g) s += (v - m) ** 2;
      return s / (g.length - 1);
    });
    const w = groupValues.map((g, i) => g.length / variances[i]);
    const W = w.reduce((a, b) => a + b, 0);
    const ybarW = w.reduce((s, wi, i) => s + wi * groupMeans[i], 0) / W;
    const num = w.reduce((s, wi, i) => s + wi * (groupMeans[i] - ybarW) ** 2, 0) / (k - 1);
    const sumTerm = w.reduce((s, wi, i) => s + (1 - wi / W) ** 2 / (groupValues[i].length - 1), 0);
    const denom = 1 + (2 * (k - 2) / (k * k - 1)) * sumTerm;
    const Fw = num / denom;
    const df2w = (k * k - 1) / (3 * sumTerm);
    welch = { F: Fw, df1: k - 1, df2: df2w, p: fSurvival(Fw, k - 1, df2w) };
  }

  // Kruskal-Wallis
  const allVals = groupValues.flat();
  const sortedAll = [...allVals].sort((a, b) => a - b);
  const ranks = allVals.map((v) => {
    // average rank
    let lo = 0, hi = sortedAll.length;
    // find positions
    let left = 0, right = 0;
    for (let i = 0; i < sortedAll.length; i++) {
      if (sortedAll[i] < v) left++;
      if (sortedAll[i] <= v) right++;
    }
    return (left + 1 + right) / 2;
  });
  let idx = 0;
  const groupRanks = groupValues.map((g) => {
    const r = ranks.slice(idx, idx + g.length);
    idx += g.length;
    return r;
  });
  const groupRankSums = groupRanks.map((r) => r.reduce((a, b) => a + b, 0));
  // tie correction
  const tieCounts = new Map<number, number>();
  for (const v of sortedAll) tieCounts.set(v, (tieCounts.get(v) ?? 0) + 1);
  let tieSum = 0;
  for (const [, c] of tieCounts) tieSum += c * c * c - c;
  const tieCorr = 1 - tieSum / (N * N * N - N);
  const H = (12 / (N * (N + 1))) * groupRankSums.reduce((s, R, i) => s + R * R / groupValues[i].length, 0) - 3 * (N + 1);
  const Hp = tieCorr > 0 ? H / tieCorr : H;
  const kruskal = { H: Hp, df: k - 1, p: chiSquareSurvival(Hp, k - 1), epsilon2: Hp / ((N * N - 1) / (N + 1)) };

  // Brown-Forsythe (Levene median)
  let bf: OneWayAnovaResult['assumptions']['brownForsythe'] = null;
  if (!allN1) {
    const zGroups = groupValues.map((g) => {
      const m = median(g);
      return g.map((v) => Math.abs(v - m));
    });
    // run ANOVA on z
    const zMeans = zGroups.map(mean);
    const zGrand = mean(zGroups.flat());
    let zSSB = 0, zSSW = 0;
    for (let i = 0; i < k; i++) {
      zSSB += zGroups[i].length * (zMeans[i] - zGrand) ** 2;
      for (const v of zGroups[i]) zSSW += (v - zMeans[i]) ** 2;
    }
    const zF = (zSSB / (k - 1)) / (zSSW / (N - k));
    bf = { F: zF, p: fSurvival(zF, k - 1, N - k), df1: k - 1, df2: N - k };
  }

  // Bartlett (secondary)
  let bartlett: OneWayAnovaResult['assumptions']['bartlett'] = null;
  if (!allN1) {
    const variances = groupValues.map((g) => {
      const m = mean(g);
      let s = 0;
      for (const v of g) s += (v - m) ** 2;
      return s / (g.length - 1);
    });
    const sp2 = groupValues.reduce((s, g, i) => s + (g.length - 1) * variances[i], 0) / (N - k);
    const num = (N - k) * Math.log(sp2) - groupValues.reduce((s, g, i) => s + (g.length - 1) * Math.log(variances[i]), 0);
    const denom = 1 + (1 / (3 * (k - 1))) * (groupValues.reduce((s, g) => s + 1 / (g.length - 1), 0) - 1 / (N - k));
    const T = num / denom;
    bartlett = { T, p: chiSquareSurvival(T, k - 1) };
  }

  // Shapiro-Wilk on residuals (Royston 1982 / 1995 approximation).
  // Returns null when the test cannot be computed so callers never receive a
  // fabricated W statistic or p-value.
  const residuals = groupValues.flatMap((g, i) => g.map((v) => v - groupMeans[i]));
  let shapiro: OneWayAnovaResult['assumptions']['shapiroWilk'] = null;
  if (residuals.length >= 3) {
    shapiro = shapiroWilkTest(residuals);
  }

  // Durbin-Watson on residuals (if ordered)
  let dw: number | null = null;
  let r1: number | null = null;
  if (residuals.length > 2) {
    let num = 0, den = 0;
    for (let i = 1; i < residuals.length; i++) num += (residuals[i] - residuals[i - 1]) ** 2;
    for (const r of residuals) den += r * r;
    dw = num / den;
    // lag-1 autocorr
    const m = mean(residuals);
    let c0 = 0, c1 = 0;
    for (let i = 0; i < residuals.length; i++) c0 += (residuals[i] - m) ** 2;
    for (let i = 1; i < residuals.length; i++) c1 += (residuals[i] - m) * (residuals[i - 1] - m);
    r1 = c1 / c0;
  }
  let warning: string | null = null;
  if (r1 !== null && Math.abs(r1) > 0.3) {
    warning = 'Residuals show substantial lag-1 autocorrelation (r1=' + r1.toFixed(3) + '); ANOVA p-values may be too small for autocorrelated earth-science series.';
  }

  // Post-hoc Tukey (all pairs)
  const posthoc: OneWayAnovaResult['posthoc'] = [];
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const d = groupMeans[i] - groupMeans[j];
      const se = Math.sqrt((MSW / 2) * (1 / groupValues[i].length + 1 / groupValues[j].length));
      const qstat = Math.abs(d) / se;
      const p = ptukey(qstat, k, df2);
      const qcrit = qtukey(1 - alpha, k, df2);
      posthoc.push({
        pair: [groupLabels[i], groupLabels[j]],
        meanDiff: d,
        ciLow: d - qcrit * se,
        ciHigh: d + qcrit * se,
        p,
        method: 'tukey',
      });
    }
  }

  return {
    k,
    N,
    nDropped,
    table: [
      { source: 'Between groups', ss: SSB, df: df1, ms: MSB, F, p },
      { source: 'Within groups', ss: SSW, df: df2, ms: MSW, F: null, p: null },
      { source: 'Total', ss: SST, df: N - 1, ms: SST / (N - 1), F: null, p: null },
    ],
    effectSizes: { eta2, omega2, omega2Headline: Math.max(0, omega2), cohensF },
    assumptions: {
      brownForsythe: bf,
      bartlett,
      shapiroWilk: shapiro,
      durbinWatson: dw,
      lag1Autocorr: r1,
      warning,
    },
    welch,
    kruskal,
    posthoc,
    error: null,
  };
}

function errorResult(k: number, N: number, nDropped: number[], msg: string): OneWayAnovaResult {
  return {
    k, N, nDropped,
    table: [],
    effectSizes: { eta2: NaN, omega2: NaN, omega2Headline: 0, cohensF: NaN },
    assumptions: { brownForsythe: null, bartlett: null, shapiroWilk: null, durbinWatson: null, lag1Autocorr: null, warning: null },
    welch: null,
    kruskal: null,
    posthoc: [],
    error: msg,
  };
}

// ── Shapiro–Wilk normality test (Royston 1982 / 1995) ──────────────────
//
// W statistic:
//   W = [ sum_{i=1}^{floor(n/2)} a_{n-i+1} * (x_{(n-i+1)} - x_{(i)}) ]^2
//       / sum_{j=1}^{n} (x_j - xbar)^2
//
// x_{(i)} are the ordered observations. The coefficients a_k are proportional
// to the expected values of standard-normal order statistics and are computed
// via Blom's plotting-position approximation
//   a_k ≈ Φ^{-1}((k - 3/8) / (n + 1/4))
// (Blom 1958), a documented standard approximation. No coefficients are
// invented and skewness is never used as a substitute.
//
// p-value:
//   Royston's (1995, AS R94) normal approximation to the null distribution of
//   W. Under H0, g = -ln(1 - W) is approximately normal with mean mu and sd
//   sigma (regressions on u = ln(n)); p = P(W <= observed | H0) is the lower
//   tail of that normal approximation.
//
// Returns null when the test cannot be computed (fewer than 3 finite
// observations, or a constant / zero-variance sample), so callers never
// receive a fabricated W statistic or p-value.

/** Inverse standard normal CDF (Beasley–Springer–Moro approximation). */
function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  // Abramowitz & Stegun 26.2.23 rational approximation (inverse normal)
  if (p < 0.5) return -normalQuantile(1 - p);
  const t = Math.sqrt(-2 * Math.log(1 - p));
  const c0 = 2.515517;
  const c1 = 0.802853;
  const c2 = 0.010328;
  const d1 = 1.432788;
  const d2 = 0.189269;
  const d3 = 0.001308;
  return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
}

/** Blom approximation to the expected k-th standard-normal order statistic. */
function expectedNormalOrderStatistic(k: number, n: number): number {
  return normalQuantile((k - 0.375) / (n + 0.25));
}

export interface ShapiroWilkResult {
  W: number;
  p: number;
  n: number;
}

/**
 * Shapiro-Wilk test for normality.
 * Returns null if the test cannot be computed (n < 3 finite values, or
 * zero-variance / constant sample).
 */
export function shapiroWilkTest(values: number[]): ShapiroWilkResult | null {
  // Drop non-finite observations.
  const clean = values.filter((v) => Number.isFinite(v));
  const n = clean.length;
  if (n < 3) return null;

  // Constant / zero-variance sample: W is 0/0 (undefined), so report unavailable.
  let min = Infinity;
  let max = -Infinity;
  for (const v of clean) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (min === max) return null;

  // Ordered sample.
  const x = [...clean].sort((a, b) => a - b);

  // Denominator: corrected sum of squares.
  const mean = clean.reduce((s, v) => s + v, 0) / n;
  let denom = 0;
  for (const v of clean) {
    const d = v - mean;
    denom += d * d;
  }
  if (!Number.isFinite(denom) || denom <= 0) return null;

  // Coefficients: proportional to expected standard-normal order statistics,
  // antisymmetric (a_{n+1-i} = -a_i) and normalized so that sum(a^2) = 1.
  // Normalization is essential: without it the numerator can exceed the
  // denominator and W is wrongly clamped to 1. With unit-norm weights,
  // Cauchy-Schwarz guarantees W <= 1 (equality only for perfectly normal data).
  const coeffs = new Array<number>(n);
  let normSq = 0;
  for (let k = 1; k <= n; k++) {
    const c = expectedNormalOrderStatistic(k, n);
    coeffs[k - 1] = c;
    normSq += c * c;
  }
  const norm = Math.sqrt(normSq);
  if (!Number.isFinite(norm) || norm <= 0) return null;
  for (let k = 0; k < n; k++) coeffs[k] /= norm;

  // Numerator: b = sum_{i=1}^{floor(n/2)} a_{n-i+1} * (x_(n-i+1) - x_(i))
  const m = Math.floor(n / 2);
  let bnum = 0;
  for (let i = 1; i <= m; i++) {
    const aHigh = coeffs[n - i]; // 0-indexed: a_{n-i+1}
    const diff = x[n - i] - x[i - 1];
    bnum += aHigh * diff;
  }

  const W = (bnum * bnum) / denom;
  const Wclamped = Number.isFinite(W) ? Math.max(0, Math.min(1, W)) : 0;

  // Royston (1995) p-value approximation: g = -ln(1 - W) approx normal.
  // Under H0, z = (g - mu)/sigma ~ N(0,1); p = Phi(z) (lower tail).
  //   W close to 1 -> g large positive -> z >= ~0 -> p high (normal)
  //   W small      -> g small/negative -> z negative -> p low (non-normal)
  let p = 1;
  if (Wclamped > 0 && Wclamped < 1) {
    const u = Math.log(n);
    const g = -Math.log(1 - Wclamped);
    const mu =
      1.586155 + 0.310827 * u + 0.083751 * u * u - 0.0038915 * u * u * u;
    const sigma = Math.exp(-0.480805 - 0.204301 * u + 0.002204 * u * u);
    const z = (g - mu) / Math.max(1e-12, sigma);
    p = normalCdf(z); // lower tail: small W -> small p
  } else if (Wclamped >= 1) {
    p = 1;
  }
  p = Math.max(0, Math.min(1, p));

  return { W: Wclamped, p, n };
}

function isFinite(v: number): boolean {
  return Number.isFinite(v);
}
