import { CompatibleMethod, InterpolationRasterType } from '../../types/earthScience';

const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1/rasters';

async function apiPost(path: string, body: unknown): Promise<any> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

async function apiGet(path: string): Promise<any> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

export { API_BASE, apiPost, apiGet };
export type { CompatibleMethod, InterpolationRasterType };
