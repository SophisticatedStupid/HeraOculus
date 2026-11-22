/**
 * Vercel proxy function tests.
 *
 * Covers path allowlisting (only /api/v1/*), method allowlisting (GET/POST),
 * server-side X-API-Key injection, X-Forwarded-For forwarding, and 500 when
 * the backend is not configured.
 */
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { handleRequest } from '../api/index.ts';

const originalFetch = globalThis.fetch;
const originalBackendUrl = process.env.BACKEND_URL;
const originalApiKey = process.env.HERAOCULUS_API_KEY;

beforeEach(() => {
  process.env.BACKEND_URL = 'http://backend:8000';
  process.env.HERAOCULUS_API_KEY = 'test-key-123';
});

afterEach(() => {
  vi.unstubAllGlobals();
  if (originalBackendUrl === undefined) {
    delete process.env.BACKEND_URL;
  } else {
    process.env.BACKEND_URL = originalBackendUrl;
  }
  if (originalApiKey === undefined) {
    delete process.env.HERAOCULUS_API_KEY;
  } else {
    process.env.HERAOCULUS_API_KEY = originalApiKey;
  }
});

function makeRequest(
  path: string,
  method = 'POST',
  body: unknown = { test: true }
): Request {
  const url = `http://localhost/api/index?path=${encodeURIComponent(path)}`;
  const init: RequestInit = { method };
  if (method !== 'GET') {
    init.body = JSON.stringify(body);
    init.headers = { 'content-type': 'application/json' };
  }
  return new Request(url, init);
}

function stubFetch(handler: (req: Request) => Promise<Response>): void {
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    let req: Request;
    const finalInit: RequestInit = { ...init };
    if (finalInit.body !== undefined && !('duplex' in finalInit)) {
      finalInit.duplex = 'half';
    }
    if (typeof input === 'string') {
      req = new Request(input, finalInit);
    } else if (input instanceof Request) {
      req = input;
    } else {
      req = new Request(input.href, finalInit);
    }
    return handler(req);
  });
}

describe('proxy path allowlisting', () => {
  it('proxies /api/v1/* paths to the backend', async () => {
    let upstreamUrl = '';
    stubFetch(async (req) => {
      upstreamUrl = req.url;
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    });

    const res = await handleRequest(makeRequest('v1/health', 'GET'));
    expect(res.status).toBe(200);
    expect(upstreamUrl).toBe('http://backend:8000/api/v1/health');
  });

  it('proxies nested /api/v1/* paths', async () => {
    let upstreamUrl = '';
    stubFetch(async (req) => {
      upstreamUrl = req.url;
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    });

    await handleRequest(makeRequest('v1/rasters/dem-diff/upload', 'POST'));
    expect(upstreamUrl).toBe('http://backend:8000/api/v1/rasters/dem-diff/upload');
  });

  it('rejects non-v1 paths with 404', async () => {
    stubFetch(async () => {
      throw new Error('fetch should not be called for disallowed paths');
    });

    const res = await handleRequest(makeRequest('v2/health', 'GET'));
    expect(res.status).toBe(404);
    expect(await res.json()).toEqual({ error: 'Not found' });
  });

  it('rejects paths without the v1 prefix with 404', async () => {
    const res = await handleRequest(makeRequest('health', 'GET'));
    expect(res.status).toBe(404);
  });

  it('rejects disallowed methods with 405', async () => {
    stubFetch(async () => {
      throw new Error('fetch should not be called for disallowed methods');
    });

    const res = await handleRequest(makeRequest('v1/health', 'DELETE'));
    expect(res.status).toBe(405);
    expect(await res.json()).toEqual({ error: 'Method not allowed' });
  });
});

describe('proxy header injection', () => {
  it('injects X-API-Key from server env', async () => {
    let capturedHeaders: Record<string, string> = {};
    stubFetch(async (req) => {
      capturedHeaders = Object.fromEntries(req.headers.entries());
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    });

    await handleRequest(makeRequest('v1/health', 'GET'));
    expect(capturedHeaders['x-api-key']).toBe('test-key-123');
  });

  it('forwards X-Forwarded-For so the backend sees the real client IP', async () => {
    let capturedHeaders: Record<string, string> = {};
    stubFetch(async (req) => {
      capturedHeaders = Object.fromEntries(req.headers.entries());
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    });

    const req = new Request('http://localhost/api/index?path=v1/health', {
      method: 'GET',
      headers: { 'x-forwarded-for': '203.0.113.50' },
    });
    await handleRequest(req);
    expect(capturedHeaders['x-forwarded-for']).toBe('203.0.113.50');
  });

  it('passes through the backend status code', async () => {
    stubFetch(async () => {
      return new Response(JSON.stringify({ detail: 'rate limited' }), {
        status: 429,
        headers: { 'content-type': 'application/json' },
      });
    });

    const res = await handleRequest(makeRequest('v1/health', 'GET'));
    expect(res.status).toBe(429);
    expect(await res.json()).toEqual({ detail: 'rate limited' });
  });

  it('returns 500 when BACKEND_URL is not configured', async () => {
    delete process.env.BACKEND_URL;
    const res = await handleRequest(makeRequest('v1/health', 'GET'));
    expect(res.status).toBe(500);
  });

  it('returns 500 when HERAOCULUS_API_KEY is not configured', async () => {
    delete process.env.HERAOCULUS_API_KEY;
    const res = await handleRequest(makeRequest('v1/health', 'GET'));
    expect(res.status).toBe(500);
  });
});
