import { twoWayAnova } from './inferential';
import type { TwoWayAnovaResult } from '../../types/earthScience';

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

function assertNull(v: unknown, message: string): void {
  if (v === null) passes++;
  else { failures++; console.error(`FAIL: ${message} (got ${JSON.stringify(v)})`); }
}

function assertNotNull(v: unknown, message: string): void {
  if (v !== null) passes++;
  else { failures++; console.error(`FAIL: ${message} (got null)`); }
}

function rowName(r: TwoWayAnovaResult['factorA']) {
  return r;
}

function obs(...rows: { value: number; levelA: string; levelB: string }[]) {
  return rows;
}

// ── 1. Balanced 2x2 with replication (reference, matches R/statsmodels) ────
// Cells:
//   A1B1: 10, 11  (mean 10.5)   A1B2: 12, 13  (mean 12.5)
//   A2B1: 14, 15  (mean 14.5)   A2B2: 16, 17  (mean 16.5)
// n=8, grandMean=13.5
// Hand-computed: SS_A=32, SS_B=8, SS_AB=0, SS_within=2, SS_total=42
// df: A=1, B=1, AB=1, error=4. MS_error=0.5
// F_A=64, F_B=16, F_AB=0. p_AB=1, p_A<0.01, 0.001<p_B<0.05
// All three types identical for balanced designs.
{
  const data = obs(
    { value: 10, levelA: 'A1', levelB: 'B1' },
    { value: 11, levelA: 'A1', levelB: 'B1' },
    { value: 12, levelA: 'A1', levelB: 'B2' },
    { value: 13, levelA: 'A1', levelB: 'B2' },
    { value: 14, levelA: 'A2', levelB: 'B1' },
    { value: 15, levelA: 'A2', levelB: 'B1' },
    { value: 16, levelA: 'A2', levelB: 'B2' },
    { value: 17, levelA: 'A2', levelB: 'B2' }
  );
  for (const type of ['I', 'II', 'III'] as const) {
    const r = twoWayAnova(data, 'A', 'B', { type });
    assert(r.error === null, `balanced type=${type}: no error`);
    assert(r.type === type, `balanced type=${type}: type echoed`);
    assertClose(r.grandMean, 13.5, 1e-9, `balanced type=${type}: grandMean`);
    assertClose(r.ssTotal, 42, 1e-9, `balanced type=${type}: ssTotal`);
    assertClose(r.factorA.ss, 32, 1e-9, `balanced type=${type}: ssA`);
    assertClose(r.factorB.ss, 8, 1e-9, `balanced type=${type}: ssB`);
    assertClose(r.interaction.ss, 0, 1e-9, `balanced type=${type}: ssAB (no interaction)`);
    assertClose(r.within.ss, 2, 1e-9, `balanced type=${type}: ssWithin`);
    assertClose(r.within.ms, 0.5, 1e-9, `balanced type=${type}: msWithin`);
    assert(r.factorA.df === 1, `balanced type=${type}: dfA=1`);
    assert(r.factorB.df === 1, `balanced type=${type}: dfB=1`);
    assert(r.interaction.df === 1, `balanced type=${type}: dfAB=1`);
    assert(r.within.df === 4, `balanced type=${type}: dfError=4`);
    assert(r.rank === 4, `balanced type=${type}: rank=4`);
    assert(r.factorA.estimable === true, `balanced type=${type}: A estimable`);
    assert(r.factorB.estimable === true, `balanced type=${type}: B estimable`);
    assert(r.interaction.estimable === true, `balanced type=${type}: AB estimable`);
    assertClose(r.factorA.f!, 64, 1e-9, `balanced type=${type}: F_A`);
    assertClose(r.factorB.f!, 16, 1e-9, `balanced type=${type}: F_B`);
    assertClose(r.interaction.f!, 0, 1e-9, `balanced type=${type}: F_AB`);
    assertClose(r.interaction.pValue!, 1, 1e-9, `balanced type=${type}: p_AB=1`);
    assert(r.factorA.pValue! < 0.01, `balanced type=${type}: p_A < 0.01`);
    assert(r.factorB.pValue! > 0.001 && r.factorB.pValue! < 0.05, `balanced type=${type}: 0.001<p_B<0.05`);
    // F = MS / MS_error consistency
    assertClose(r.factorA.f!, r.factorA.ms / r.within.ms, 1e-9, `balanced type=${type}: F_A = MS_A/MS_err`);
    assertClose(r.factorB.f!, r.factorB.ms / r.within.ms, 1e-9, `balanced type=${type}: F_B = MS_B/MS_err`);
  }
  // Types identical for balanced
  const rI = twoWayAnova(data, 'A', 'B', { type: 'I' });
  const rII = twoWayAnova(data, 'A', 'B', { type: 'II' });
  const rIII = twoWayAnova(data, 'A', 'B', { type: 'III' });
  assertClose(rI.factorA.ss, rII.factorA.ss, 1e-9, 'balanced: Type I ssA == Type II ssA');
  assertClose(rI.factorA.ss, rIII.factorA.ss, 1e-9, 'balanced: Type I ssA == Type III ssA');
  assertClose(rI.interaction.ss, rIII.interaction.ss, 1e-9, 'balanced: Type I ssAB == Type III ssAB');
}

// ── 2. Unbalanced complete 2x2: types differ; all effects estimable ────────
{
  const data = obs(
    { value: 10, levelA: 'A1', levelB: 'B1' },
    { value: 12, levelA: 'A1', levelB: 'B1' },
    { value: 14, levelA: 'A1', levelB: 'B2' },
    { value: 16, levelA: 'A2', levelB: 'B1' },
    { value: 18, levelA: 'A2', levelB: 'B2' },
    { value: 20, levelA: 'A2', levelB: 'B2' }
  );
  const rI = twoWayAnova(data, 'A', 'B', { type: 'I' });
  const rII = twoWayAnova(data, 'A', 'B', { type: 'II' });
  const rIII = twoWayAnova(data, 'A', 'B', { type: 'III' });
  for (const r of [rI, rII, rIII]) {
    assert(r.error === null, 'unbalanced: no error');
    assert(r.factorA.estimable === true, 'unbalanced: A estimable');
    assert(r.factorB.estimable === true, 'unbalanced: B estimable');
    assert(r.interaction.estimable === true, 'unbalanced: AB estimable');
    assert(r.factorA.ss >= 0, 'unbalanced: ssA >= 0');
    assert(r.factorB.ss >= 0, 'unbalanced: ssB >= 0');
    assert(r.interaction.ss >= 0, 'unbalanced: ssAB >= 0');
    assert(r.within.df > 0, 'unbalanced: dfError > 0');
    // Internal consistency: F = MS/MS_error, p from F, p in [0,1]
    if (r.factorA.f !== null) {
      assertClose(r.factorA.f, r.factorA.ms / r.within.ms, 1e-6, 'unbalanced: F_A consistency');
      assert(r.factorA.pValue !== null, 'unbalanced: p_A available');
      assert(r.factorA.pValue! >= 0 && r.factorA.pValue! <= 1, 'unbalanced: p_A in [0,1]');
    }
  }
  // Types should generally differ on unbalanced data (at least one SS differs)
  const differ =
    rI.factorA.ss !== rII.factorA.ss ||
    rI.factorB.ss !== rII.factorB.ss ||
    rI.interaction.ss !== rIII.interaction.ss;
  assert(differ, 'unbalanced: Type I/II/III produce different SS');
}

// ── 3. Rank-deficient design (empty cell): interaction not estimable ───────
// A2B2 cell missing → rank < 4 → interaction not independently estimable.
{
  const data = obs(
    { value: 10, levelA: 'A1', levelB: 'B1' },
    { value: 11, levelA: 'A1', levelB: 'B1' },
    { value: 12, levelA: 'A1', levelB: 'B2' },
    { value: 14, levelA: 'A2', levelB: 'B1' }
    // no A2B2
  );
  const r = twoWayAnova(data, 'A', 'B', { type: 'III' });
  assert(r.error === null, 'rank-deficient: no error');
  assert(r.rank < 4, 'rank-deficient: rank < 4 (rank deficient)');
  assert(r.within.df > 0, 'rank-deficient: dfError > 0');
  // Interaction not estimable with an empty cell (Type III)
  assert(r.interaction.estimable === false, 'rank-deficient: interaction not estimable');
  assertNull(r.interaction.f, 'rank-deficient: interaction f is null');
  assertNull(r.interaction.pValue, 'rank-deficient: interaction pValue is null');
  // Main effects may still be estimable depending on design; just check no crash
  assert(Number.isFinite(r.factorA.ss), 'rank-deficient: ssA finite');
  assert(Number.isFinite(r.factorB.ss), 'rank-deficient: ssB finite');
}

// ── 4. Constant response: zero residual variance → F unavailable ────────────
{
  const data = obs(
    { value: 5, levelA: 'A1', levelB: 'B1' },
    { value: 5, levelA: 'A1', levelB: 'B1' },
    { value: 5, levelA: 'A1', levelB: 'B2' },
    { value: 5, levelA: 'A1', levelB: 'B2' },
    { value: 5, levelA: 'A2', levelB: 'B1' },
    { value: 5, levelA: 'A2', levelB: 'B1' },
    { value: 5, levelA: 'A2', levelB: 'B2' },
    { value: 5, levelA: 'A2', levelB: 'B2' }
  );
  const r = twoWayAnova(data, 'A', 'B', { type: 'III' });
  assert(r.error === null, 'constant: no error');
  assert(r.within.ms === 0, 'constant: msError = 0');
  // F undefined because no residual variance
  assert(r.factorA.f === null, 'constant: fA null');
  assert(r.factorB.f === null, 'constant: fB null');
  assert(r.interaction.f === null, 'constant: fAB null');
  assert(r.warnings.length > 0, 'constant: warning emitted');
}

// ── 5. Significant interaction ─────────────────────────────────────────────
// Cell means far from additive → interaction detectable.
{
  const data = obs(
    { value: 10, levelA: 'A1', levelB: 'B1' },
    { value: 11, levelA: 'A1', levelB: 'B1' },
    { value: 12, levelA: 'A1', levelB: 'B2' },
    { value: 13, levelA: 'A1', levelB: 'B2' },
    { value: 14, levelA: 'A2', levelB: 'B1' },
    { value: 14, levelA: 'A2', levelB: 'B1' },
    { value: 20, levelA: 'A2', levelB: 'B2' },
    { value: 20, levelA: 'A2', levelB: 'B2' }
  );
  const r = twoWayAnova(data, 'A', 'B', { type: 'III' });
  assert(r.error === null, 'interaction: no error');
  assert(r.interaction.ss > 0, 'interaction: ssAB > 0');
  assert(r.interaction.estimable === true, 'interaction: estimable');
  assert(r.interaction.pValue !== null, 'interaction: pValue available');
  if (r.interaction.pValue !== null) {
    assert(r.interaction.pValue < 0.05, 'interaction: p < 0.05 (significant departure from additivity)');
  }
}

// ── 6. No interaction (additive) → interaction p = 1 ───────────────────────
{
  const data = obs(
    { value: 10, levelA: 'A1', levelB: 'B1' },
    { value: 12, levelA: 'A1', levelB: 'B1' },
    { value: 14, levelA: 'A1', levelB: 'B2' },
    { value: 16, levelA: 'A1', levelB: 'B2' },
    { value: 18, levelA: 'A2', levelB: 'B1' },
    { value: 20, levelA: 'A2', levelB: 'B1' },
    { value: 22, levelA: 'A2', levelB: 'B2' },
    { value: 24, levelA: 'A2', levelB: 'B2' }
  );
  const r = twoWayAnova(data, 'A', 'B', { type: 'III' });
  assert(r.error === null, 'additive: no error');
  assertClose(r.interaction.ss, 0, 1e-6, 'additive: ssAB = 0');
  assertClose(r.interaction.pValue!, 1, 1e-9, 'additive: p_AB = 1');
}

// ── 7. Insufficient levels → error result ──────────────────────────────────
{
  const r = twoWayAnova([{ value: 1, levelA: 'A1', levelB: 'B1' }], 'A', 'B');
  assert(r.error !== null && r.error!.length > 0, 'insufficient: error message');
  assertNull(r.factorA.f, 'insufficient: fA null');
  assert(r.factorA.estimable === false, 'insufficient: not estimable');
}

// ── 8. No observations → error result ──────────────────────────────────────
{
  const r = twoWayAnova([], 'A', 'B');
  assert(r.error !== null && r.error!.length > 0, 'empty: error message');
  assertNull(r.factorA.f, 'empty: fA null');
}

// ── Summary ────────────────────────────────────────────────────────────────
console.log(`\n${passes} passed, ${failures} failed`);
