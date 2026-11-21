import { describe, expect, it } from 'vitest';
import { runOneWayAnova, ptukey, qtukey } from '../src/services/statistics/anova';

describe('Games-Howell post-hoc — converted from test_gameshowell.ts', () => {
  it('triggers Games-Howell under clearly unequal variances', () => {
    const r = runOneWayAnova({
      response: [
        // Group A: small variance
        10, 11, 12, 10, 11, 12, 10, 11,
        // Group B: large variance
        1, 20, 2, 25, 3, 30, 4, 35, 5, 40,
        // Group C: small variance
        50, 51, 52, 50, 51, 52, 50, 51,
      ],
      group: [
        'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
        'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
        'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
      ],
    });
    expect(r.error).toBeNull();
    expect(r.assumptions.equalVariancesAssumed).toBe(false);
    expect(r.posthoc.length).toBeGreaterThan(0);
    expect(r.posthoc.every((p) => p.method === 'games-howell')).toBe(true);
    r.posthoc.forEach((p) => {
      expect(Number.isFinite(p.p)).toBe(true);
      expect(Number.isFinite(p.df)).toBe(true);
      expect(p.p).toBeGreaterThanOrEqual(0);
      expect(p.p).toBeLessThanOrEqual(1);
    });
  });

  it('ptukey k=2 critical value at df=4 gives p≈0.05', () => {
    // For k=2, Q = sqrt(2)*|T| and P(Q>q) = two-tailed t p-value.
    // q=3.926486, df=4 -> q/sqrt(2) ≈ 2.776 = t_{0.975,4} -> p ≈ 0.05.
    const p1 = ptukey(3.926486, 2, 4);
    expect(p1).toBeCloseTo(0.05, 1);
  });

  it('qtukey k=2 df=4 returns critical value ~3.9265 at alpha=0.05', () => {
    // qtukey(p) returns q such that P(Q > q) = p; alpha=0.05 -> p=0.05.
    const q1 = qtukey(0.05, 2, 4);
    expect(q1).toBeCloseTo(3.9265, 1);
  });

  it('qtukey k=2 df=100 returns critical value ~2.8070 at alpha=0.05', () => {
    const q2 = qtukey(0.05, 2, 100);
    expect(q2).toBeCloseTo(2.8070, 1);
  });

  it('ptukey(qtukey(p)) roundtrip recovers p', () => {
    for (const k of [2, 3, 4]) {
      for (const df of [5, 10, 30]) {
        const p = 0.05;
        const q = qtukey(p, k, df);
        const p2 = ptukey(q, k, df);
        expect(p2).toBeCloseTo(p, 1);
      }
    }
  });
});
