// src/services/statistics/twoWayAnova.ts
// DEPRECATED: This placeholder module is retained only so that existing
// imports/exports (`export * from './twoWayAnova'` in index.ts) keep working.
// All two-way ANOVA logic now lives in `inferential.ts` (`twoWayAnova`),
// which supports balanced and unbalanced designs with Type I / II / III sums
// of squares and rank-aware handling of empty factor cells.
//
// Prefer importing `twoWayAnova` directly from '../../services/statistics/inferential'
// (it is re-exported from the statistics barrel). This shim will be removed in
// a future release.

import { twoWayAnova, type TwoWayAnovaResult } from './inferential';

export type { TwoWayAnovaResult };

/**
 * @deprecated Use `twoWayAnova(observations, factorALabel, factorBLabel, { type })`
 * from './inferential' instead. Defaults to Type II sums of squares to match this
 * shim's historical default.
 */
export function runTwoWayAnova(data: {
  response: number[];
  factorA: string[];
  factorB: string[];
  type?: 'I' | 'II' | 'III';
}): TwoWayAnovaResult {
  // The legacy shim accepted response/factorA/factorB parallel arrays. The real
  // implementation expects { value, levelA, levelB } observations. Reconstruct
  // them positionally so existing callers (if any) keep working.
  const n = Math.min(data.response.length, data.factorA.length, data.factorB.length);
  const observations: { value: number; levelA: string; levelB: string }[] = [];
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(data.response[i])) {
      observations.push({
        value: data.response[i],
        levelA: data.factorA[i],
        levelB: data.factorB[i],
      });
    }
  }
  return twoWayAnova(observations, 'Factor A', 'Factor B', { type: data.type ?? 'II' });
}
