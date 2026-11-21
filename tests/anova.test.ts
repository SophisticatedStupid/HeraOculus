import { describe, expect, it } from 'vitest';
import { runOneWayAnova, ptukey, qtukey } from '../src/services/statistics/anova';

describe('runOneWayAnova — converted from test_anova.ts', () => {
  it('handles two groups k=2 without error', () => {
    const r = runOneWayAnova({
      response: [1, 2, 3, 4, 5, 6],
      group: ['A', 'A', 'A', 'B', 'B', 'B'],
    });
    expect(r.error).toBeNull();
    expect(r.k).toBe(2);
    expect(r.N).toBe(6);
    expect(r.posthoc).toHaveLength(1);
    expect(r.posthoc[0]?.method).toMatch(/^(tukey|games-howell)$/);
  });

  it('handles three groups equal n with finite F and p', () => {
    const r = runOneWayAnova({
      response: [1, 2, 3, 4, 5, 6, 7, 8, 9],
      group: ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    });
    expect(r.error).toBeNull();
    expect(r.k).toBe(3);
    expect(r.N).toBe(9);
    expect(Number.isFinite(r.table[0]?.F ?? -1)).toBe(true);
    expect(Number.isFinite(r.table[0]?.p ?? -1)).toBe(true);
    expect(r.posthoc).toHaveLength(3);
  });

  it('ptukey/qtukey roundtrip recovers the input probability', () => {
    for (const k of [2, 3, 4, 5]) {
      for (const df of [5, 10, 30, 100]) {
        const p = 0.5;
        const q = qtukey(p, k, df);
        const p2 = ptukey(q, k, df);
        expect(p2).toBeCloseTo(p, 1);
      }
    }
  });

  it('unequal variances runs post-hoc without error', () => {
    const r = runOneWayAnova({
      response: [1, 1, 1, 2, 2, 10, 3, 3, 3],
      group: ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    });
    expect(r.error).toBeNull();
    expect(r.posthoc.length).toBeGreaterThan(0);
    r.posthoc.forEach((p) => {
      expect(p.method).toMatch(/^(tukey|games-howell)$/);
      expect(Number.isFinite(p.p)).toBe(true);
      expect(Number.isFinite(p.df)).toBe(true);
    });
  });

  it('small df (k=2, N=4) does not throw', () => {
    const r = runOneWayAnova({
      response: [1, 2, 3, 4],
      group: ['A', 'A', 'B', 'B'],
    });
    expect(r.error).toBeNull();
    expect(r.k).toBe(2);
    expect(r.N).toBe(4);
  });
});
