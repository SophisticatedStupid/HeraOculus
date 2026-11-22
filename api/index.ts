/**
 * HeraOculus Vercel serverless function — same-origin API proxy.
 *
 * Routes every /api/v1/* request to the FastAPI backend (BACKEND_URL, a
 * server-only environment variable) and injects the X-API-Key header from
 * HERAOCULUS_API_KEY (also server-only). The frontend never sees either
 * value and always calls same-origin /api/v1/*.
 *
 * Security properties:
 * - Only /api/v1/* paths and GET/POST methods are proxied (allowlisted).
 * - X-API-Key is injected server-side; it cannot be read or overridden by the browser.
 * - X-Forwarded-For is forwarded so the backend rate limiter sees the real client
 *   IP (Vercel populates it from the TCP connection, overwriting any client
 *   spoofing attempt).
 * - Request/response bodies are streamed (multipart uploads up to 35 MB work).
 *
 * Deployed at api/index.ts with vercel.json rewrite:
 *   { "source": "/api/:path*", "destination": "/api/index?path=:path*" }
 *
 * Uses the Vercel fetch Web Standard (no @vercel/node dependency required).
 */

const ALLOWED_PREFIX = 'v1/';
const ALLOWED_METHODS = new Set(['GET', 'POST']);

export async function handleRequest(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const path = url.searchParams.get('path');

  // Path allowlist: only /api/v1/* is proxied.
  if (!path || !path.startsWith(ALLOWED_PREFIX)) {
    return new Response(JSON.stringify({ error: 'Not found' }), {
      status: 404,
      headers: { 'content-type': 'application/json' },
    });
  }

  // Method allowlist: only GET and POST.
  const method = request.method;
  if (!ALLOWED_METHODS.has(method)) {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'content-type': 'application/json' },
    });
  }

  const backendUrl = process.env.BACKEND_URL;
  const apiKey = process.env.HERAOCULUS_API_KEY?.trim();
  if (!backendUrl || !apiKey) {
    return new Response(
      JSON.stringify({ error: 'Backend not configured' }),
      {
        status: 500,
        headers: { 'content-type': 'application/json' },
      }
    );
  }

  // Build outgoing headers. Strip hop-by-hop / host headers that fetch will
  // set itself, then inject the server-side API key.
  const headers = new Headers(request.headers);
  headers.delete('host');
  headers.delete('content-length');
  headers.set('x-api-key', apiKey);
  // X-Forwarded-For is preserved: Vercel sets it from the TCP connection and
  // overwrites any client-supplied value, so the backend sees the real client IP.

  const upstreamUrl = backendUrl.replace(/\/+$/, '') + '/api/' + path;

  const fetchInit: RequestInit = { method, headers };
  if (method !== 'GET') {
    // Stream the request body (multipart uploads up to 35 MB).
    // duplex is required by Node.js when a body is present.
    fetchInit.body = request.body as unknown as BodyInit;
    fetchInit.duplex = 'half';
  }

  const upstreamResponse = await fetch(upstreamUrl, fetchInit);

  // Pass through status code and content type; stream the response body.
  const responseHeaders = new Headers(upstreamResponse.headers);
  responseHeaders.delete('connection');
  responseHeaders.delete('keep-alive');

  return new Response(upstreamResponse.body, {
    status: upstreamResponse.status,
    statusText: upstreamResponse.statusText,
    headers: responseHeaders,
  });
}

export default {
  fetch: handleRequest,
};
