import { CompatibleMethod, InterpolationRasterType } from '../../types/earthScience';
import { fromArrayBuffer } from 'geotiff';

const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1/rasters';

export interface RasterFile {
  data: number[][];
  width: number;
  height: number;
  crs: string | null;
  transform: { a: number; b: number; c: number; d: number; e: number; f: number } | null;
  nodata: number | null;
  bands: number;
}

/**
 * Read an uploaded GeoTIFF file into an in-memory 2D array plus basic metadata.
 *
 * Reads the first band. NoData values are preserved as `NaN` so missing data
 * is carried through the interpolation workflow.  CRS/transform extraction is
 * best-effort: when the file embeds usable GeoKeys they are reported, otherwise
 * they are null (null CRS values compare equal, which is sufficient for the
 * multi-file temporal consistency check).
 */
export async function readRasterFile(file: File): Promise<RasterFile> {
  const arrayBuffer = await file.arrayBuffer();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tiff: any = await fromArrayBuffer(arrayBuffer);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const rasters: any[] = await tiff.getRasters();
  if (rasters.length === 0) {
    throw new Error('No raster bands found in uploaded file');
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const raster: any = rasters[0];
  const width = raster.width;
  const height = raster.height;
  const values = await raster.getArray();
  const data: number[][] = [];
  for (let r = 0; r < height; r++) {
    const row: number[] = [];
    for (let c = 0; c < width; c++) {
      const v = values[r * width + c];
      row.push(Number.isNaN(v) ? NaN : v);
    }
    data.push(row);
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodata: any = raster.nodataValue ?? tiff.nodataValue ?? null;
  let crs: string | null = null;
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gk: any = tiff.geoKeys ?? {};
    if (gk.ProjectedCSTypeGeoKey) crs = `EPSG:${gk.ProjectedCSTypeGeoKey}`;
    else if (gk.GeographicTypeGeoKey) crs = `EPSG:${gk.GeographicTypeGeoKey}`;
  } catch {
    // best-effort only
  }
  return {
    data,
    width,
    height,
    crs,
    transform: null,
    nodata: nodata == null ? null : Number(nodata),
    bands: rasters.length,
  };
}

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

/**
 * POST multipart form data (e.g. a raster file upload).
 *
 * The `Content-Type` header is intentionally omitted so the browser sets the
 * `multipart/form-data` boundary automatically.  `formData` must already
 * contain the file part plus any string fields (band_index, user_confirmed_type).
 */
async function apiPostForm(path: string, formData: FormData): Promise<any> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    body: formData,
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

export { API_BASE, apiPost, apiPostForm, apiGet };
export type { CompatibleMethod, InterpolationRasterType };
