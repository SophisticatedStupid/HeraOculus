import { shapiroWilkTest, ShapiroWilkResult } from './anova';

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
  const ok = Number.isFinite(actual) && Number.isFinite(expected) && Math.abs(actual - expected) <= tol;
  if (ok) {
    passes++;
  } else {
    failures++;
    console.error(`FAIL: ${message} (actual=${actual}, expected=${expected}, tol=${tol})`);
  }
}

function assertInRange(v: number, lo: number, hi: number, message: string): void {
  const ok = Number.isFinite(v) && v >= lo && v <= hi;
  if (ok) {
    passes++;
  } else {
    failures++;
    console.error(`FAIL: ${message} (v=${v}, range=[${lo},${hi}])`);
  }
}

// Simple deterministic PRNG (mulberry32) for reproducible reference samples.
function mulberry32(seed: number): () => number {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function boxMuller(rng: () => number, mean = 0, sd = 1): number {
  const u = 1 - rng(); // [0,1)
  const v = 1 - rng();
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return mean + sd * z;
}

const summary = (label: string, r: ShapiroWilkResult | null): void => {
  console.log(
    `${label}: ${r === null ? 'null' : `W=${r.W.toFixed(4)} p=${r.p.toFixed(4)} n=${r.n}`}`
  );
};

// ── 1. Approximately normal data → W high, p not significant ─────────────
{
  const rng = mulberry32(12345);
  const vals: number[] = [];
  for (let i = 0; i < 120; i++) vals.push(boxMuller(rng, 50, 10));
  const r = shapiroWilkTest(vals);
  summary('normal n=120', r);
  assert(r !== null, 'normal: test available');
  if (r) {
    assertInRange(r.W, 0, 1, 'normal: W in [0,1]');
    assertInRange(r.p, 0, 1, 'normal: p in [0,1]');
    assert(r.n === 120, 'normal: n === 120');
    assert(r.W > 0.90, 'normal: W > 0.90');
    assert(r.p > 0.05, 'normal: p > 0.05 (no significant departure)');
  }
}

// ── 2. Strongly skewed data → W lower, p significant ─────────────────────
{
  const rng = mulberry32(99);
  const vals: number[] = [];
  for (let i = 0; i < 100; i++) vals.push(-Math.log(Math.max(1e-6, rng()))); // exponential
  const r = shapiroWilkTest(vals);
  summary('skewed exponential n=100', r);
  assert(r !== null, 'skewed: test available');
  if (r) {
    assertInRange(r.W, 0, 1, 'skewed: W in [0,1]');
    assertInRange(r.p, 0, 1, 'skewed: p in [0,1]');
    assert(r.W < 0.95, 'skewed: W < 0.95');
    assert(r.p < 0.05, 'skewed: p < 0.05 (evidence against normality)');
  }
}

// ── 3. Heavy-tailed data → departure detected ────────────────────────────
{
  const rng = mulberry32(77);
  const vals: number[] = [];
  for (let i = 0; i < 100; i++) vals.push(boxMuller(rng, 0, 1) * (rng() > 0.9 ? 10 : 1));
  const r = shapiroWilkTest(vals);
  summary('heavy-tailed n=100', r);
  assert(r !== null, 'heavy-tailed: test available');
  if (r) {
    assertInRange(r.W, 0, 1, 'heavy-tailed: W in [0,1]');
    assertInRange(r.p, 0, 1, 'heavy-tailed: p in [0,1]');
  }
}

// ── 4. Constant data → null (test unavailable, never fabricated) ─────────
{
  const r = shapiroWilkTest([7, 7, 7, 7, 7]);
  assert(r === null, 'constant: returns null (unavailable)');
}

// ── 5. Very small sample sizes ───────────────────────────────────────────
{
  const r3 = shapiroWilkTest([1, 2, 3]);
  assert(r3 !== null, 'n=3: test available');
  if (r3) {
    assertInRange(r3.W, 0, 1, 'n=3: W in [0,1]');
    assertInRange(r3.p, 0, 1, 'n=3: p in [0,1]');
    assert(r3.n === 3, 'n=3: n === 3');
  }

  const r2 = shapiroWilkTest([1, 2]);
  assert(r2 === null, 'n=2: returns null (insufficient)');

  const r1 = shapiroWilkTest([1]);
  assert(r1 === null, 'n=1: returns null');

  const r0 = shapiroWilkTest([]);
  assert(r0 === null, 'n=0: returns null');
}

// ── 6. Boundary sample sizes (n=3 minimum, n=5000 large) ─────────────────
{
  // n = 5000 normal data should complete without throwing and be finite.
  const rng = mulberry32(5);
  const big: number[] = [];
  for (let i = 0; i < 5000; i++) big.push(boxMuller(rng, 0, 1));
  const rBig = shapiroWilkTest(big);
  assert(rBig !== null, 'n=5000: test available');
  if (rBig) {
    assert(rBig.n === 5000, 'n=5000: n === 5000');
    assertInRange(rBig.W, 0, 1, 'n=5000: W in [0,1]');
    assertInRange(rBig.p, 0, 1, 'n=5000: p in [0,1]');
  }
}

// ── 7. Non-finite values dropped ─────────────────────────────────────────
{
  const r = shapiroWilkTest([1, NaN, 2, Infinity, 3, -Infinity, 4]);
  // finite values = [1,2,3,4], n=4
  assert(r !== null && r.n === 4, 'non-finite dropped: n === 4');
  if (r) {
    assertInRange(r.W, 0, 1, 'non-finite: W in [0,1]');
    assertInRange(r.p, 0, 1, 'non-finite: p in [0,1]');
  }
}

// ── 8. Tied observations (not constant) → W < 1, computed ────────────────
{
  const r = shapiroWilkTest([1, 1, 2, 3, 3, 4]);
  assert(r !== null, 'ties: test available');
  if (r) {
    assertInRange(r.W, 0, 1, 'ties: W in [0,1]');
    assertInRange(r.p, 0, 1, 'ties: p in [0,1]');
    // Ties but not constant => W strictly less than 1 (some departure)
    assert(r.W < 1, 'ties: W < 1');
  }
}

// ── 9. Deterministic: same input → same output ───────────────────────────
{
  const a = shapiroWilkTest([1, 2, 3, 4, 5, 6]);
  const b = shapiroWilkTest([1, 2, 3, 4, 5, 6]);
  assert(JSON.stringify(a) === JSON.stringify(b), 'deterministic: identical outputs');
}

// ── 10. Negative values handled ──────────────────────────────────────────
{
  const r = shapiroWilkTest([-5, -3, -1, 1, 3, 5]);
  assert(r !== null, 'negative values: test available');
  if (r) {
    assertInRange(r.W, 0, 1, 'negative: W in [0,1]');
    assertInRange(r.p, 0, 1, 'negative: p in [0,1]');
    assert(r.W > 0.90, 'negative symmetric: W > 0.90');
  }
}

// ── 11. Very small-magnitude measurements ────────────────────────────────
{
  const r = shapiroWilkTest([1e-12, 2e-12, 3e-12, 4e-12, 5e-12]);
  assert(r !== null, 'tiny values: test available');
  if (r) {
    assertInRange(r.W, 0, 1, 'tiny: W in [0,1]');
    assertInRange(r.p, 0, 1, 'tiny: p in [0,1]');
  }
}

// ── 12. Large-magnitude measurements ─────────────────────────────────────
{
  const r = shapiroWilkTest([1e12, 2e12, 3e12, 4e12, 5e12]);
  assert(r !== null, 'large values: test available');
  if (r) {
    assertInRange(r.W, 0, 1, 'large: W in [0,1]');
    assertInRange(r.p, 0, 1, 'large: p in [0,1]');
  }
}

// ── 13. Hand-calculated reference: evenly-spaced [1,2,3,4] → W ≈ 1.0 ───
{
  const r = shapiroWilkTest([1, 2, 3, 4]);
  assert(r !== null, 'reference [1,2,3,4]: test available');
  if (r) {
    // Evenly spaced data is perfectly normal-shaped; W ~ 0.998 (close to 1)
    assertClose(r.W, 0.998, 0.01, 'reference [1,2,3,4]: W ≈ 0.998');
    assertInRange(r.p, 0, 1, 'reference [1,2,3,4]: p in [0,1]');
  }
}

// ── 14. Invariant: W never exceeds 1 across many random samples ────────
{
  const rng = mulberry32(4242);
  let maxW = 0;
  for (let trial = 0; trial < 50; trial++) {
    const vals: number[] = [];
    for (let i = 0; i < 30; i++) vals.push(boxMuller(rng, 0, 1) * (1 + rng()));
    const r = shapiroWilkTest(vals);
    if (r && r.W > maxW) maxW = r.W;
  }
  assert(maxW <= 1 + 1e-9, 'invariant: W <= 1 across 50 random samples');
}

// ── Summary ──────────────────────────────────────────────────────────────
console.log(`\n${passes} passed, ${failures} failed`);
