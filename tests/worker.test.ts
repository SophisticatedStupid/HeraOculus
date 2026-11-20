/**
 * Worker communication and main-thread fallback tests.
 *
 * Validates that runStatsWorkerTask:
 * - falls back to the main thread when Web Workers are unavailable (viaWorker: false),
 * - throws when no worker and no fallback are provided,
 * - dispatches through a Worker when one is available (viaWorker: true),
 * - propagates worker errors,
 * - and matches main-thread statistics results.
 *
 * Each worker test resets the module registry so the client's cached
 * `sharedWorker` singleton does not leak between tests.
 */
import { computeDescriptiveStats } from '../src/services/statistics';
import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';

const originalWorker = globalThis.Worker;

beforeEach(() => {
  globalThis.Worker = originalWorker;
});

afterEach(() => {
  globalThis.Worker = originalWorker;
  vi.resetModules();
});

async function importClient() {
  const mod = await import('../src/workers/statsWorkerClient');
  return mod.runStatsWorkerTask;
}

function installMockWorker(opts: {
  onResponse?: (msg: { id: string; type: string; payload: unknown }) => { id: string; type: string; result: unknown; durationMs: number } | null;
  onError?: () => void;
}): void {
  class MockWorker {
    onmessage: ((event: { data: unknown }) => void) | null = null;
    onerror: ((err: unknown) => void) | null = null;
    postMessage = (msg: { id: string; type: string; payload: unknown }) => {
      queueMicrotask(() => {
        if (opts.onError) {
          opts.onError();
          if (this.onerror) this.onerror(new Error('worker boom'));
          return;
        }
        const response = opts.onResponse?.(msg);
        if (response && this.onmessage) {
          this.onmessage({ data: response });
        }
      });
    };
    terminate = () => {};
  }
  (globalThis as unknown as { Worker: typeof MockWorker }).Worker = MockWorker as unknown as Worker;
}

describe('runStatsWorkerTask — main-thread fallback', () => {
  it('uses main-thread fallback when Worker is unavailable', async () => {
    const runStatsWorkerTask = await importClient();
    const values = [1, 2, 3, 4, 5];
    const expected = computeDescriptiveStats(values);
    const res = await runStatsWorkerTask<ReturnType<typeof computeDescriptiveStats>>(
      'descriptive',
      { values },
      () => expected
    );
    expect(res.viaWorker).toBe(false);
    expect(res.result).toEqual(expected);
    expect(Number.isFinite(res.durationMs)).toBe(true);
  });

  it('throws when Worker is unavailable and no fallback is provided', async () => {
    const runStatsWorkerTask = await importClient();
    await expect(runStatsWorkerTask('descriptive', { values: [1, 2, 3] })).rejects.toThrow(
      'Web Worker unavailable'
    );
  });

  it('propagates an error thrown by the fallback', async () => {
    const runStatsWorkerTask = await importClient();
    await expect(
      runStatsWorkerTask('descriptive', { values: [1, 2, 3] }, () => {
        throw new Error('fallback boom');
      })
    ).rejects.toThrow('fallback boom');
  });
});

describe('runStatsWorkerTask — worker dispatch', () => {
  it('dispatches through a Worker and returns viaWorker: true', async () => {
    const runStatsWorkerTask = await importClient();
    const values = [10, 20, 30];
    const expected = computeDescriptiveStats(values);
    installMockWorker({
      onResponse: msg => ({
        id: msg.id,
        type: msg.type,
        result: expected,
        durationMs: 5,
      }),
    });
    const res = await runStatsWorkerTask<ReturnType<typeof computeDescriptiveStats>>(
      'descriptive',
      { values },
      () => computeDescriptiveStats(values)
    );
    expect(res.viaWorker).toBe(true);
    expect(res.result).toEqual(expected);
    expect(res.durationMs).toBe(5);
  });

  it('propagates worker errors as rejections', async () => {
    const runStatsWorkerTask = await importClient();
    installMockWorker({ onError: () => {} });
    await expect(
      runStatsWorkerTask('descriptive', { values: [1, 2, 3] }, () => [1, 2, 3] as never)
    ).rejects.toThrow('worker boom');
  });
});
