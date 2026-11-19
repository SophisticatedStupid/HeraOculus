import {
  calculateMannKendall,
} from './index';
import {
  seasonalMannKendall,
  pettittChangePoint,
  computeTieGroups,
  tieSign,
  MANN_KENDALL_TIE_TOLERANCE,
} from './timeseries';
import { normalPValue } from './distributions';
import { TimeSeriesPoint } from '../../types/earthScience';

let failures = 0;
let passes = 0;

function assert(condition: boolean, message: string): void {
  if (condition) {
    passes++;
  } else {
    failures++;
    console.error(`FAIL: ${message}`);
  }
}

function assertClose(actual: number, expected: number, tol: number, message: string): void {
  const ok = Math.abs(actual - expected) <= tol;
  if (ok) {
    passes++;
  } else {
    failures++;
    console.error(`FAIL: ${message} (actual=${actual}, expected=${expected}, tol=${tol})`);
  }
}

function mkPoints(values: (number | null)[]): TimeSeriesPoint[] {
  return values.map((v, i) => ({
    date: `2020-${String((i % 12) + 1).padStart(2, '0')}-01`,
    year: 2020 + Math.floor(i / 12),
    month: (i % 12) + 1,
    raw: v,
    isMissing: v === null,
    anomaly: v === null ? undefined : v,
    movingAvg: undefined,
    trend: undefined,
    ciUpper: undefined,
    ciLower: undefined,
  }));
}

// ── 1. Strictly increasing → positive S ───────────────────────────────────
{
  const r = calculateMannKendall(mkPoints([1, 2, 3, 4, 5]));
  assert(r.s === 10, 'strictly increasing: S === 10');
  assert(r.s > 0, 'strictly increasing: S > 0');
  assertClose(r.varS, 300 / 18, 1e-9, 'strictly increasing: varS === 300/18');
  assertClose(r.z, 9 / Math.sqrt(300 / 18), 1e-6, 'strictly increasing: z with continuity');
  assertClose(r.tau, 1.0, 1e-9, 'strictly increasing: tau === 1');
}

// ── 2. Strictly decreasing → negative S ───────────────────────────────────
{
  const r = calculateMannKendall(mkPoints([5, 4, 3, 2, 1]));
  assert(r.s === -10, 'strictly decreasing: S === -10');
  assert(r.s < 0, 'strictly decreasing: S < 0');
  assertClose(r.varS, 300 / 18, 1e-9, 'strictly decreasing: varS === 300/18');
  assertClose(r.z, -9 / Math.sqrt(300 / 18), 1e-6, 'strictly decreasing: z with continuity');
  assertClose(r.tau, -1.0, 1e-9, 'strictly decreasing: tau === -1');
}

// ── 3. Constant sequence → S = 0, Z = 0 ───────────────────────────────────
{
  const r = calculateMannKendall(mkPoints([3, 3, 3, 3, 3]));
  assert(r.s === 0, 'constant: S === 0');
  assert(r.z === 0, 'constant: z === 0');
  assertClose(r.varS, 0, 1e-9, 'constant: varS === 0 (non-negative)');
  assert(r.tau === 0, 'constant: tau === 0');
  assertClose(r.pValue, 1, 1e-6, 'constant: pValue === 1');
}

// ── 4. Tie correction matches formula ─────────────────────────────────────
// [1, 2, 2, 3, 4]: groups sizes [1,2,1,1], tieTerm = 2*1*(2*2+5) = 18
{
  const r = calculateMannKendall(mkPoints([1, 2, 2, 3, 4]));
  // S: (1,2)x2=+2,(1,3)=+1,(1,4)=+1,(2,2)=0,(2,3)=+1,(2,4)=+1,
  //    (2,3)=+1,(2,4)=+1,(3,4)=+1 => S = 9
  assert(r.s === 9, 'ties [1,2,2,3,4]: S === 9');
  const tieTerm = 2 * 1 * 9;
  assertClose(r.varS, (5 * 4 * 15 - tieTerm) / 18, 1e-9, 'ties: varS matches formula');
  assertClose(r.z, 8 / Math.sqrt((300 - tieTerm) / 18), 1e-6, 'ties: z with continuity');
}

// ── 5. Near-equal observations treated consistently ──────────────────────
// diff within tolerance => tie; diff just above => not tie
{
  const gClose = computeTieGroups([1.0, 1.0 + 1e-11]);
  assert(gClose.groupIds[0] === gClose.groupIds[1], 'near tie (1e-11): same group');
  assert(gClose.groupSizes.includes(2), 'near tie (1e-11): group size 2');

  const gFar = computeTieGroups([1.0, 1.0 + 1e-9]);
  assert(gFar.groupIds[0] !== gFar.groupIds[1], 'not a tie (1e-9): different groups');
  assert(!gFar.groupSizes.includes(2), 'not a tie (1e-9): no group of size 2');
}

// S and variance use the SAME tie definition for near ties (4 points, n>=4)
{
  const rTie = calculateMannKendall(mkPoints([1.0, 1.0 + 1e-11, 2.0, 3.0]));
  // 1.0 & 1.0+1e-11 ties; pairs: tie, +1, +1, +1, +1, +1 => S = 5
  assert(rTie.s === 5, 'near tie S consistent: S === 5');
  // group sizes [2,1,1] => tieTerm = 18 => varS = (156-18)/18
  assertClose(rTie.varS, 138 / 18, 1e-9, 'near tie varS consistent');
}

{
  const rTie = calculateMannKendall(mkPoints([1.0, 1.0 + 1e-11, 2.0, 3.0]));
  const rNoTie = calculateMannKendall(mkPoints([1.0, 1.0 + 1e-9, 2.0, 3.0]));
  // no ties, strictly increasing 4 points => S = 6
  assert(rNoTie.s === 6, 'no near tie: S === 6');
  assertClose(rNoTie.varS, 156 / 18, 1e-9, 'no near tie: varS === 156/18');
  // near vs not-tie produce different results
  assert(rTie.s !== rNoTie.s, 'near vs not-tie produce different S');
}

// ── 6. Very small floating-point measurements ─────────────────────────────
{
  const r = calculateMannKendall(mkPoints([1e-12, 2e-12, 3e-12, 4e-12]));
  // All near zero (scale=1, tol=1e-10), diffs 1e-12 <= 1e-10 => all ties
  assert(r.s === 0, 'tiny values: S === 0 (all ties)');
  assertClose(r.varS, 0, 1e-9, 'tiny values: varS === 0');
  assert(r.z === 0, 'tiny values: z === 0');
}

// ── 7. Large-magnitude measurements ───────────────────────────────────────
{
  const r = calculateMannKendall(mkPoints([1e12, 2e12, 3e12, 4e12]));
  // diffs ~1e12 >> tol*scale (~200) => no ties, strictly increasing
  assert(r.s === 6, 'large values: S === 6 (strictly increasing, no ties)');
  assert(r.s > 0, 'large values: S > 0');
  assertClose(r.varS, (4 * 3 * 13) / 18, 1e-9, 'large values: varS no ties');
  assert(r.z > 0, 'large values: z > 0');
}

// ── 8. Negative values ────────────────────────────────────────────────────
{
  const rInc = calculateMannKendall(mkPoints([-5, -3, -1, 1, 3]));
  assert(rInc.s === 10, 'negative increasing: S === 10');
  assert(rInc.s > 0, 'negative increasing: S > 0');

  const rDec = calculateMannKendall(mkPoints([3, 1, -1, -3, -5]));
  assert(rDec.s === -10, 'negative decreasing: S === -10');
  assert(rDec.s < 0, 'negative decreasing: S < 0');
}

// ── 9. Missing observations excluded ──────────────────────────────────────
{
  const r = calculateMannKendall(mkPoints([1, null, 2, null, 3, null, 4]));
  // valid = [1,2,3,4], n=4
  assert(r.s === 6, 'missing excluded: S === 6');
  assertClose(r.varS, (4 * 3 * 13) / 18, 1e-9, 'missing excluded: varS no ties');
  // n < 4 case
  const rSmall = calculateMannKendall(mkPoints([1, null, 2, null, 3]));
  assert(rSmall.s === 0, 'n<4 with missing: S === 0');
  assertClose(rSmall.varS, 1, 1e-9, 'n<4 with missing: varS === 1');
}

// ── 10. Variance never negative ───────────────────────────────────────────
{
  const r = calculateMannKendall(mkPoints([7, 7, 7, 7, 7, 7, 7]));
  assert(r.varS >= 0, 'all ties: varS >= 0');
  assert(r.varS === 0, 'all ties: varS === 0');
  const r2 = calculateMannKendall(mkPoints([1, 1, 2, 2, 3, 3, 4, 4]));
  assert(r2.varS >= 0, 'many ties: varS >= 0');
}

// ── 11. Continuity correction + two-sided p-value preserved ───────────────
{
  const r = calculateMannKendall(mkPoints([1, 2, 3, 4, 5]));
  const expectedZ = (r.s - 1) / Math.sqrt(r.varS);
  assertClose(r.z, expectedZ, 1e-9, 'continuity correction: z === (S-1)/sqrt(varS)');
  assertClose(r.pValue, normalPValue(r.z), 1e-9, 'p-value === normalPValue(z)');
  assert(r.pValue >= 0 && r.pValue <= 1, 'p-value in [0,1]');
}

// ── 12. Standard vs seasonal MK agree (seasonLength=1) ────────────────────
{
  const vals = [1, 3, 2, 5, 4, 7, 6, 9, 8, 10];
  const standard = calculateMannKendall(mkPoints(vals));
  const seasonal = seasonalMannKendall(vals, 1, 'standard');

  assert(standard.s === seasonal.s, 'seasonal len=1: S matches standard');
  assertClose(standard.varS, seasonal.varS, 1e-9, 'seasonal len=1: varS matches standard');
  assertClose(standard.z, seasonal.z, 1e-9, 'seasonal len=1: z matches standard');
  assertClose(standard.pValue, seasonal.pValue, 1e-9, 'seasonal len=1: pValue matches standard');
}

// ── 13. Tie grouping canonical: groupIds deterministic ────────────────────
{
  const g1 = computeTieGroups([5, 5, 1, 1, 3]);
  const g2 = computeTieGroups([5, 5, 1, 1, 3]);
  assert(JSON.stringify(g1.groupIds) === JSON.stringify(g2.groupIds), 'tie groups deterministic');
  assert(g1.groupIds[0] === g1.groupIds[1], 'two 5s same group');
  assert(g1.groupIds[2] === g1.groupIds[3], 'two 1s same group');
  assert(g1.groupIds[0] !== g1.groupIds[2], '5 and 1 different groups');
}

// ── 14. tieSign uses canonical groups ─────────────────────────────────────
{
  const { groupIds } = computeTieGroups([1.0, 1.0 + 1e-11, 2.0]);
  assert(tieSign(1.0, 1.0 + 1e-11, groupIds[0], groupIds[1]) === 0, 'tieSign tie => 0');
  assert(tieSign(2.0, 1.0, groupIds[2], groupIds[0]) === 1, 'tieSign 2>1 => +1');
  assert(tieSign(1.0, 2.0, groupIds[0], groupIds[2]) === -1, 'tieSign 1<2 => -1');
}

// ── 15. Pettitt uses same tie definition ──────────────────────────────────
{
  const r1 = pettittChangePoint([1, 2, 3, 4, 5]);
  const r2 = pettittChangePoint([1.0, 1.0 + 1e-11, 3.0, 4.0, 5.0]);
  assert(typeof r1.testStatistic === 'number', 'pettitt no-tie: numeric statistic');
  assert(typeof r2.testStatistic === 'number', 'pettitt near-tie: numeric statistic');
  assert(Number.isFinite(r1.pValue), 'pettitt: finite p-value');
  assert(Number.isFinite(r2.pValue), 'pettitt near-tie: finite p-value');
}

// ── Summary ───────────────────────────────────────────────────────────────
console.log(`\n${passes} passed, ${failures} failed`);
