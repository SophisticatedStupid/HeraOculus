/**
 * Integration-style tests for the InterpolationLab upload → inspect → interpolate workflow.
 *
 * No live backend or browser DOM is required: `fetch` is mocked and the GeoTIFF
 * reader is stubbed, so these tests verify the request contracts (multipart
 * inspection upload, JSON temporal/spatial interpolation payloads), file
 * validation, timestamp handling, and temporal-cube assembly.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  apiPost,
  apiPostForm,
  readRasterFile,
} from '../src/components/InterpolationLab/api';
import {
  validateFile,
  toDecimalYear,
  assembleTemporalCube,
  type TemporalObservation,
} from '../src/components/InterpolationLab/useInterpolationLab';

vi.mock('geotiff', () => ({
  fromArrayBuffer: vi.fn(),
}));
import { fromArrayBuffer } from 'geotiff';

const mockFetch = vi.fn();
globalThis.fetch = mockFetch as any;

function okJson(data: any, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn(async () => data),
  } as any;
}

function errJson(detail: string, status = 400) {
  return {
    ok: false,
    status,
    json: vi.fn(async () => ({ detail })),
  } as any;
}

function makeFile(
  name: string,
  size: number,
  buffer: ArrayBuffer = new ArrayBuffer(0)
): File {
  return {
    name,
    size,
    type: 'image/tiff',
    arrayBuffer: async () => buffer,
  } as unknown as File;
}

function makeObs(
  id: string,
  timestamp: string,
  time: number,
  data: number[][],
  width: number,
  height: number,
  crs: string | null = 'EPSG:32610',
  nodata: number | null = -9999
): TemporalObservation {
  return {
    id,
    file: makeFile(`${id}.tif`, 1000),
    timestamp,
    time,
    data,
    width,
    height,
    crs,
    nodata,
  };
}

beforeEach(() => {
  mockFetch.mockReset();
  vi.mocked(fromArrayBuffer).mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('apiPostForm — multipart inspection upload', () => {
  it('sends the raster as multipart form data with band_index and user_confirmed_type', async () => {
    mockFetch.mockResolvedValue(okJson({ width: 64, height: 64, band_index: 1 }));
    const file = makeFile('scene.tif', 12345);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('band_index', '1');
    formData.append('user_confirmed_type', 'continuous');

    await apiPostForm('/inspect', formData);

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/v1/rasters/inspect');
    expect(init.method).toBe('POST');
    // Multipart: browser sets the boundary; no explicit Content-Type header.
    expect(init.headers).toBeUndefined();
    expect(init.body).toBe(formData);
    // Assert against the test-local formData (same instance as init.body).
    const entries = [...formData.entries()];
    expect(entries.map(([k]) => k)).toContain('file'); // file part is in the multipart body
    expect(formData.get('band_index')).toBe('1');
    expect(formData.get('user_confirmed_type')).toBe('continuous');
  });

  it('returns the inspection metadata preserved from the backend', async () => {
    const meta = {
      width: 64,
      height: 64,
      bands: 1,
      crs: 'EPSG:32610',
      nodata_value: -9999,
      missing_pixel_count: 819,
      missing_percentage: 20.0,
    };
    mockFetch.mockResolvedValue(okJson(meta));
    const result = await apiPostForm('/inspect', new FormData());
    expect(result).toEqual(meta);
  });

  it('throws a useful error when the backend rejects the upload', async () => {
    mockFetch.mockResolvedValue(errJson('Unsupported raster format', 400));
    await expect(apiPostForm('/inspect', new FormData())).rejects.toThrow(
      'Unsupported raster format'
    );
  });
});

describe('apiPost — JSON interpolation payloads', () => {
  it('spatial interpolation sends a 2D array as JSON', async () => {
    mockFetch.mockResolvedValue(okJson({ method: 'idw', band_index: 1 }));
    const data = [[1, null], [3, 4]];
    await apiPost('/interpolate', {
      data,
      method: 'idw',
      band_index: 1,
      raster_type: 'continuous',
    });
    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers['Content-Type']).toBe('application/json');
    expect(JSON.parse(init.body)).toMatchObject({
      data,
      method: 'idw',
      band_index: 1,
      raster_type: 'continuous',
    });
  });

  it('temporal interpolation sends a 3D cube with times', async () => {
    mockFetch.mockResolvedValue(okJson({ method: 'temporal_linear', band_index: 1 }));
    const cube = [
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
    ];
    await apiPost('/interpolate', {
      data: cube,
      method: 'temporal_linear',
      band_index: 1,
      raster_type: 'temporal',
      times: [2020.0, 2020.5],
      max_gap_steps: 1,
      allow_extrapolation: false,
    });
    const [, init] = mockFetch.mock.calls[0];
    const body = JSON.parse(init.body);
    expect(body.data).toEqual(cube); // (T=2, H=2, W=2)
    expect(body.times).toEqual([2020.0, 2020.5]);
    expect(body.raster_type).toBe('temporal');
    expect(body.method).toBe('temporal_linear');
  });

  it('propagates backend interpolation errors', async () => {
    mockFetch.mockResolvedValue(errJson('Temporal interpolation requires a 3D cube', 400));
    await expect(
      apiPost('/interpolate', { data: [[1]], method: 'temporal_linear' })
    ).rejects.toThrow('Temporal interpolation requires a 3D cube');
  });
});

describe('readRasterFile — GeoTIFF parsing', () => {
  it('parses the first band into a 2D array preserving NoData as NaN', async () => {
    const values = new Float32Array([1, 2, NaN, 4]);
    vi.mocked(fromArrayBuffer).mockResolvedValue({
      getRasters: vi.fn().mockResolvedValue([
        {
          width: 2,
          height: 2,
          getArray: vi.fn().mockResolvedValue(values),
          nodataValue: -9999,
        },
      ]),
    } as any);

    const result = await readRasterFile(makeFile('scene.tif', 100));
    expect(result.width).toBe(2);
    expect(result.height).toBe(2);
    expect(result.data).toEqual([[1, 2], [NaN, 4]]);
    expect(result.nodata).toBe(-9999);
    expect(result.bands).toBe(1);
    expect(vi.mocked(fromArrayBuffer)).toHaveBeenCalledTimes(1);
  });

  it('throws when the file has no bands', async () => {
    vi.mocked(fromArrayBuffer).mockResolvedValue({ getRasters: vi.fn().mockResolvedValue([]) } as any);
    await expect(readRasterFile(makeFile('empty.tif', 10))).rejects.toThrow(
      'No raster bands found'
    );
  });
});

describe('validateFile — pre-upload validation', () => {
  it('accepts supported GeoTIFF extensions', () => {
    for (const ext of ['.tif', '.tiff', '.geotiff']) {
      expect(validateFile(makeFile(`file${ext}`, 1000))).toBeNull();
    }
  });

  it('rejects unsupported formats', () => {
    expect(validateFile(makeFile('scene.png', 1000))).toContain('Unsupported file type');
    expect(validateFile(makeFile('scene.csv', 1000))).toContain('Unsupported file type');
  });

  it('rejects oversized files', () => {
    const big = makeFile('big.tif', 200 * 1024 * 1024); // 200 MB > 100 MB cap
    expect(validateFile(big)).toContain('File too large');
  });
});

describe('toDecimalYear — timestamp handling', () => {
  it('accepts decimal-year numbers', () => {
    expect(toDecimalYear('2020.5')).toBe(2020.5);
    expect(toDecimalYear('2021')).toBe(2021);
  });

  it('converts ISO dates to decimal years', () => {
    const y = toDecimalYear('2020-06-01T00:00:00Z');
    expect(y).toBeGreaterThan(2020);
    expect(y).toBeLessThan(2021);
  });

  it('returns NaN for invalid timestamps', () => {
    expect(Number.isNaN(toDecimalYear('not-a-date'))).toBe(true);
    expect(Number.isNaN(toDecimalYear(''))).toBe(true);
  });
});

describe('assembleTemporalCube — temporal (T,H,W) cube assembly', () => {
  const base = [[1, 2], [3, 4]];

  it('returns null for an empty stack', () => {
    expect(assembleTemporalCube([])).toBeNull();
  });

  it('orders observations by time and preserves timestamps', () => {
    const obs = [
      makeObs('b', '2021.0', 2021.0, base, 2, 2),
      makeObs('a', '2020.0', 2020.0, base, 2, 2),
      makeObs('c', '2020.5', 2020.5, base, 2, 2),
    ];
    const out = assembleTemporalCube(obs);
    expect(out).not.toBeNull();
    expect(out!.times).toEqual([2020.0, 2020.5, 2021.0]);
    expect(out!.cube).toHaveLength(3);
  });

  it('rejects duplicate timestamps', () => {
    const obs = [
      makeObs('a', '2020.0', 2020.0, base, 2, 2),
      makeObs('b', '2020.0', 2020.0, base, 2, 2),
    ];
    expect(assembleTemporalCube(obs)).toBeNull();
  });

  it('builds a single-step cube', () => {
    const obs = [makeObs('a', '2020.0', 2020.0, base, 2, 2)];
    const out = assembleTemporalCube(obs);
    expect(out!.times).toEqual([2020.0]);
    expect(out!.cube).toEqual([base]);
  });

  it('preserves NoData (NaN) within the cube', () => {
    const obs = [makeObs('a', '2020.0', 2020.0, [[1, NaN], [3, 4]], 2, 2)];
    const out = assembleTemporalCube(obs);
    expect(out!.cube[0][0][1]).toBeNaN();
  });
});
