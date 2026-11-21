import React, {
  useState,
  useCallback,
  useMemo,
  useRef,
  useEffect,
} from 'react';
import {
  Layout,
  Download,
  FileText,
  X,
  CheckCircle2,
  AlertCircle,
  Loader2,
  BarChart3,
  Play,
  Sliders,
  Cpu,
  RefreshCw,
  Eye,
} from 'lucide-react';
import {
  InterpolationMethod,
  InterpolationRasterType,
  InterpolationStage,
  RasterInspectionResult,
  InterpolationResult,
  ValidationResult,
  CompatibleMethod,
} from '../types/earthScience';
import { Button } from './ui/Button';
import { Panel } from './ui/Panel';
import { Field } from './ui/Field';

// ═════════════════════════════════════════════════════════════════════
// API Helpers
// ═════════════════════════════════════════════════════════════════════

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

// ═════════════════════════════════════════════════════════════════════
// Synthetic raster generator (for demo/testing without upload)
// ═════════════════════════════════════════════════════════════════════

export function generateSyntheticRaster({
  width = 64,
  height = 64,
  type = 'continuous',
  missingPct = 0.15,
  seed = 42,
}: {
  width?: number;
  height?: number;
  type?: InterpolationRasterType;
  missingPct?: number;
  seed?: number;
}) {
  const rng = (() => {
    let s = seed;
    return () => {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    };
  })();

  const data: number[][] = [];
  for (let r = 0; r < height; r++) {
    const row: number[] = [];
    for (let c = 0; c < width; c++) {
      let val: number;
      if (type === 'categorical') {
        val = Math.floor(rng() * 5);
      } else {
        const grad =
          Math.sin((r / height) * Math.PI * 2) * 10 +
          Math.cos((c / width) * Math.PI * 2) * 10;
        val = 50 + grad + rng() * 5;
      }
      row.push(val);
    }
    data.push(row);
  }

  const missingMask: boolean[][] = [];
  for (let r = 0; r < height; r++) {
    missingMask.push(new Array(width).fill(false));
  }

  const nMissing = Math.floor(width * height * missingPct);
  // Build and shuffle all pixel indices for O(n) missing placement
  const indices: [number, number][] = [];
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      indices.push([r, c]);
    }
  }
  // Fisher-Yates shuffle
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.min(i, Math.floor(rng() * (i + 1)));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  // Mark first nMissing as missing, with clustering
  for (let i = 0; i < nMissing && i < indices.length; i++) {
    const [r, c] = indices[i];
    missingMask[r][c] = true;
    // Cluster: also mark neighbors with some probability
    if (rng() < 0.7) {
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const nr = r + dr,
            nc = c + dc;
          if (
            nr >= 0 &&
            nr < height &&
            nc >= 0 &&
            nc < width &&
            !missingMask[nr][nc] &&
            rng() < 0.5
          ) {
            missingMask[nr][nc] = true;
          }
        }
      }
    }
  }

  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      if (missingMask[r][c]) data[r][c] = NaN;
    }
  }

  return {
    data,
    width,
    height,
    transform: { a: 10, b: 0, c: 500000, d: 0, e: -10, f: 5000000 },
    crs: 'EPSG:32610',
    nodata: -9999,
  };
}

// ═════════════════════════════════════════════════════════════════════
// Array to ImageData
// ═════════════════════════════════════════════════════════════════════

export function arrayToImageData(
  data: number[][],
  options: {
    colormap?: 'viridis' | 'gray' | 'categorical';
    nodataColor?: string;
    vmin?: number;
    vmax?: number;
  } = {}
): ImageData {
  const { colormap = 'viridis', nodataColor = '#1a1a1a', vmin, vmax } = options;
  const h = data.length;
  const w = data[0]?.length ?? 0;
  const imgData = new ImageData(w, h);

  const vals: number[] = [];
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      const v = data[r][c];
      if (v !== null && !Number.isNaN(v)) vals.push(v);
    }

  const minV = vmin ?? (vals.length > 0 ? Math.min(...vals) : 0);
  const maxV = vmax ?? (vals.length > 0 ? Math.max(...vals) : 1);
  const range = maxV - minV || 1;

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const idx = (r * w + c) * 4;
      const v = data[r][c];
      if (v === null || Number.isNaN(v)) {
        const [rr, gg, bb] = hexToRgb(nodataColor);
        imgData.data[idx] = rr;
        imgData.data[idx + 1] = gg;
        imgData.data[idx + 2] = bb;
        imgData.data[idx + 3] = 255;
      } else {
        const t = Math.max(0, Math.min(1, (v - minV) / range));
        const [rr, gg, bb] =
          colormap === 'categorical'
            ? catColor(Math.floor(v))
            : colormap === 'gray'
              ? [t * 255, t * 255, t * 255]
              : viridis(t);
        imgData.data[idx] = rr;
        imgData.data[idx + 1] = gg;
        imgData.data[idx + 2] = bb;
        imgData.data[idx + 3] = 255;
      }
    }
  }
  return imgData;
}

export function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!m) return [0, 0, 0];
  return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)];
}

export function catColor(c: number): [number, number, number] {
  const colors: [number, number, number][] = [
    [255, 107, 107],
    [78, 205, 196],
    [69, 183, 209],
    [156, 122, 245],
    [255, 176, 78],
  ];
  return colors[c % colors.length];
}

export function viridis(t: number): [number, number, number] {
  const r = 68 + Math.round(187 * Math.pow(t, 0.7));
  const g = 18 + Math.round(150 * Math.pow(t, 0.8));
  const b = 114 - Math.round(70 * Math.pow(t, 0.6));
  return [
    Math.min(255, Math.max(0, r)),
    Math.min(255, Math.max(0, g)),
    Math.min(255, Math.max(0, b)),
  ];
}

// ═════════════════════════════════════════════════════════════════════
// Method configuration UI
// ═════════════════════════════════════════════════════════════════════

function MethodConfig({
  method,
  config,
  onChange,
}: {
  method: InterpolationMethod;
  config: Record<string, any>;
  onChange: (key: string, value: any) => void;
}) {
  const set = (key: string, value: any) => onChange(key, value);

  return (
    <div className="space-y-3">
      {method === 'nearest_neighbor' && (
        <Field label="Maximum search distance (pixels)" help="Leave empty for unlimited search.">
          <input
            type="number"
            value={config.max_distance_pixels ?? ''}
            onChange={(e) =>
              set('max_distance_pixels', e.target.value === '' ? undefined : Number(e.target.value))
            }
            className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
          />
        </Field>
      )}
      {method === 'bilinear' && (
        <>
          <Field label="Neighborhood">
            <select
              value={config.neighborhood ?? '3x3'}
              onChange={(e) => set('neighborhood', e.target.value)}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            >
              <option value="3x3">3x3</option>
              <option value="5x5">5x5</option>
            </select>
          </Field>
          <Field label="Edge handling">
            <select
              value={config.edge_handling ?? 'clamp'}
              onChange={(e) => set('edge_handling', e.target.value)}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            >
              <option value="clamp">Clamp to edge</option>
              <option value="nearest_valid">Nearest valid</option>
              <option value="nodata">NoData at edge</option>
            </select>
          </Field>
          <Field label="Maximum search distance (pixels)">
            <input
              type="number"
              value={config.max_distance_pixels ?? ''}
              onChange={(e) =>
                set('max_distance_pixels', e.target.value === '' ? undefined : Number(e.target.value))
              }
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
        </>
      )}
      {method === 'bicubic' && (
        <>
          <Field label="Neighborhood">
            <select
              value={config.neighborhood ?? '4x4'}
              onChange={(e) => set('neighborhood', e.target.value)}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            >
              <option value="4x4">4x4</option>
              <option value="6x6">6x6</option>
            </select>
          </Field>
          <Field label="Clip to observed range" help="Prevent overshooting beyond local min/max.">
            <input
              type="checkbox"
              checked={config.clip_to_observed_range ?? true}
              onChange={(e) => set('clip_to_observed_range', e.target.checked)}
              className="rounded bg-ink-900 border border-ink-700 text-gold-400 focus:ring-gold-400"
            />
          </Field>
          <Field label="Maximum search distance (pixels)">
            <input
              type="number"
              value={config.max_distance_pixels ?? ''}
              onChange={(e) =>
                set('max_distance_pixels', e.target.value === '' ? undefined : Number(e.target.value))
              }
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
        </>
      )}
      {(method === 'local_mean' || method === 'local_median') && (
        <>
          <Field label="Window size">
            <input
              type="number"
              value={config.window_size ?? 3}
              min={3}
              max={21}
              step={2}
              onChange={(e) => set('window_size', Number(e.target.value))}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
          <Field label="Min valid neighbors">
            <input
              type="number"
              value={config.min_valid_neighbors ?? 1}
              min={1}
              onChange={(e) => set('min_valid_neighbors', Number(e.target.value))}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
          <Field label="Neighborhood shape">
            <select
              value={config.neighborhood_shape ?? 'square'}
              onChange={(e) => set('neighborhood_shape', e.target.value)}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            >
              <option value="square">Square</option>
              <option value="circle">Circle</option>
            </select>
          </Field>
          <Field label="Maximum search distance (pixels)">
            <input
              type="number"
              value={config.max_distance_pixels ?? ''}
              onChange={(e) =>
                set('max_distance_pixels', e.target.value === '' ? undefined : Number(e.target.value))
              }
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
        </>
      )}
      {method === 'idw' && (
        <>
          <Field label="Power (p)" help="Higher power = closer neighbors dominate more.">
            <input
              type="number"
              value={config.power ?? 2}
              step={0.1}
              onChange={(e) => set('power', Number(e.target.value))}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
          <Field label="Search radius (pixels)">
            <input
              type="number"
              value={config.search_radius_pixels ?? ''}
              onChange={(e) =>
                set('search_radius_pixels', e.target.value === '' ? undefined : Number(e.target.value))
              }
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
          <Field label="Max neighbors">
            <input
              type="number"
              value={config.max_neighbors ?? 12}
              onChange={(e) => set('max_neighbors', Number(e.target.value))}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
          <Field label="Min neighbors">
            <input
              type="number"
              value={config.min_neighbors ?? 1}
              onChange={(e) => set('min_neighbors', Number(e.target.value))}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
          <Field label="Geographic CRS" help="Use geodesic-aware distance for lat/lon rasters.">
            <input
              type="checkbox"
              checked={config.geographic_crs ?? false}
              onChange={(e) => set('geographic_crs', e.target.checked)}
              className="rounded bg-ink-900 border border-ink-700 text-gold-400 focus:ring-gold-400"
            />
          </Field>
        </>
      )}
      {method === 'ordinary_kriging' && (
        <>
          <Field label="Variogram model">
            <select
              value={config.variogram_model ?? 'spherical'}
              onChange={(e) => set('variogram_model', e.target.value)}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            >
              <option value="spherical">Spherical</option>
              <option value="exponential">Exponential</option>
              <option value="gaussian">Gaussian</option>
              <option value="linear">Linear</option>
            </select>
          </Field>
          <Field label="Nugget" help="Leave empty for automatic fitting.">
            <input
              type="number"
              value={config.variogram_nugget ?? ''}
              step="any"
              onChange={(e) =>
                set('variogram_nugget', e.target.value === '' ? undefined : Number(e.target.value))
              }
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
          <Field label="Sill">
            <input
              type="number"
              value={config.variogram_sill ?? ''}
              step="any"
              onChange={(e) =>
                set('variogram_sill', e.target.value === '' ? undefined : Number(e.target.value))
              }
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
          <Field label="Range">
            <input
              type="number"
              value={config.variogram_range ?? ''}
              step="any"
              onChange={(e) =>
                set('variogram_range', e.target.value === '' ? undefined : Number(e.target.value))
              }
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
          <Field label="Max neighbors">
            <input
              type="number"
              value={config.max_neighbors ?? 12}
              onChange={(e) => set('max_neighbors', Number(e.target.value))}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
          <Field label="Min neighbors">
            <input
              type="number"
              value={config.min_neighbors ?? 3}
              onChange={(e) => set('min_neighbors', Number(e.target.value))}
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
        </>
      )}
      {(method === 'temporal_linear' ||
        method === 'temporal_nearest_date' ||
        method === 'temporal_forward_fill' ||
        method === 'temporal_backward_fill' ||
        method === 'temporal_local_mean') && (
        <>
          <Field label="Max gap steps" help="Maximum consecutive missing time steps to fill.">
            <input
              type="number"
              value={config.max_gap_steps ?? ''}
              min={1}
              onChange={(e) =>
                set('max_gap_steps', e.target.value === '' ? undefined : Number(e.target.value))
              }
              className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
            />
          </Field>
          <Field label="Allow extrapolation" help="Fill leading/trailing gaps (marked as extrapolation).">
            <input
              type="checkbox"
              checked={config.allow_extrapolation ?? false}
              onChange={(e) => set('allow_extrapolation', e.target.checked)}
              className="rounded bg-ink-900 border border-ink-700 text-gold-400 focus:ring-gold-400"
            />
          </Field>
        </>
      )}
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════
// RasterCanvas
// ═════════════════════════════════════════════════════════════════════

function RasterCanvas({
  data,
  title,
  colormap = 'viridis',
}: {
  data: (number | null)[][];
  title: string;
  colormap?: 'viridis' | 'gray' | 'categorical';
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.length) return;
    const h = data.length;
    const w = data[0]?.length ?? 0;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const flat: number[][] = data.map((row) =>
      row.map((v) => (v === null ? NaN : v))
    );
    const imgData = arrayToImageData(flat, { colormap });
    ctx.putImageData(imgData, 0, 0);
  }, [data, colormap]);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="text-xs font-semibold text-silver-400">{title}</div>
      <canvas
        ref={canvasRef}
        className="border border-ink-700 bg-ink-950 block max-w-full max-h-[300px]"
        style={{ imageRendering: 'pixelated' }}
      />
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════
// Main InterpolationLab Component
// ═════════════════════════════════════════════════════════════════════

interface InterpolationLabProps {
  onClose?: () => void;
}

export const InterpolationLab: React.FC<InterpolationLabProps> = ({ onClose }) => {
  const [stage, setStage] = useState<InterpolationStage>('select_dataset');
  const [rasterData, setRasterData] = useState<(number | null)[][] | null>(null);
  const [rasterMeta, setRasterMeta] = useState<{
    width: number;
    height: number;
    crs: string | null;
    transform: any;
    nodata: number | null;
  } | null>(null);
  const [inspection, setInspection] = useState<RasterInspectionResult | null>(null);
  const [method, setMethod] = useState<InterpolationMethod>('nearest_neighbor');
  const [rasterType, setRasterType] = useState<InterpolationRasterType>('continuous');
  const [compatibleMethods, setCompatibleMethods] = useState<CompatibleMethod[]>([]);
  const [config, setConfig] = useState<Record<string, any>>({});
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [result, setResult] = useState<InterpolationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showProvenance, setShowProvenance] = useState(true);
  const [showQuality, setShowQuality] = useState(false);

  useEffect(() => {
    const synth = generateSyntheticRaster({
      width: 64,
      height: 64,
      type: 'continuous',
      missingPct: 0.2,
      seed: 42,
    });
    setRasterData(
      synth.data.map((row) => row.map((v) => (Number.isNaN(v) ? null : v)))
    );
    setRasterMeta({
      width: synth.width,
      height: synth.height,
      crs: synth.crs,
      transform: synth.transform,
      nodata: synth.nodata,
    });
    setStage('select_dataset');
  }, []);

  useEffect(() => {
    let mounted = true;
    apiGet(`/interpolate/compatible-methods?raster_type=${rasterType}`)
      .then((data: any) => {
        if (mounted) setCompatibleMethods(data.methods || []);
      })
      .catch((err: any) => console.warn('Failed to load compatible methods:', err));
    return () => {
      mounted = false;
    };
  }, [rasterType]);

  const handleInspect = useCallback(async () => {
    if (!rasterData) return;
    setIsLoading(true);
    setError(null);
    try {
      const data: any = await apiPost('/inspect', {
        data: rasterData,
        band_index: 1,
        user_confirmed_type: rasterType,
      });
      setInspection(data);
      setStage('inspect_missing');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [rasterData, rasterType]);

  const handleChooseMethod = useCallback((m: InterpolationMethod) => {
    setMethod(m);
    const compat = compatibleMethods.find((cm) => cm.id === m);
    if (compat) {
      setConfig({ ...compat.default_config });
    }
    setStage('choose_method');
  }, [compatibleMethods]);

  const handleConfigure = useCallback(() => {
    setStage('configure');
  }, []);

  const handleValidate = useCallback(async () => {
    if (!rasterData) return;
    setIsLoading(true);
    setError(null);
    try {
      const data: any = await apiPost('/interpolate/validate', {
        data: rasterData,
        method,
        raster_type: rasterType,
        ...config,
      });
      setValidation(data);
      setStage('validate');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [rasterData, method, rasterType, config]);

  const handleExecute = useCallback(async () => {
    if (!rasterData) return;
    setIsLoading(true);
    setError(null);
    try {
      const data: any = await apiPost('/interpolate', {
        data: rasterData,
        method,
        raster_type: rasterType,
        max_distance_pixels: config.max_distance_pixels,
        geographic_crs: config.geographic_crs,
        pixel_size_x: rasterMeta?.transform?.a ?? undefined,
        pixel_size_y: rasterMeta?.transform?.e ? Math.abs(rasterMeta.transform.e) : undefined,
        neighborhood: config.neighborhood,
        edge_handling: config.edge_handling,
        clip_to_observed_range: config.clip_to_observed_range,
        window_size: config.window_size,
        min_valid_neighbors: config.min_valid_neighbors,
        neighborhood_shape: config.neighborhood_shape,
        power: config.power,
        search_radius_pixels: config.search_radius_pixels,
        max_neighbors: config.max_neighbors,
        min_neighbors: config.min_neighbors,
        variogram_model: config.variogram_model,
        variogram_nugget: config.variogram_nugget,
        variogram_sill: config.variogram_sill,
        variogram_range: config.variogram_range,
        max_gap_steps: config.max_gap_steps,
        allow_extrapolation: config.allow_extrapolation,
      });
      setResult(data);
      setStage('execute');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [rasterData, method, rasterType, config, rasterMeta]);

  const stageOrder: InterpolationStage[] = [
    'select_dataset',
    'inspect_missing',
    'choose_method',
    'configure',
    'validate',
    'execute',
    'compare_results',
    'export',
  ];

  const stageLabels: Record<InterpolationStage, string> = {
    select_dataset: '1. Select Dataset',
    inspect_missing: '2. Inspect Missing Data',
    choose_method: '3. Choose Interpolation',
    configure: '4. Configure Parameters',
    validate: '5. Cross-Validation',
    execute: '6. Execute Interpolation',
    compare_results: '7. Compare Results',
    export: '8. Export',
  };

  const currentIndex = stageOrder.indexOf(stage);

  const currentMethodInfo = useMemo(() => {
    return compatibleMethods.find((cm) => cm.id === method);
  }, [method, compatibleMethods]);

  return (
    <div className="min-h-screen bg-ink-950 text-silver-100">
      <header className="border-b border-ink-700 bg-ink-950/95 backdrop-blur-md px-4 lg:px-8 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          <button
            onClick={() => onClose?.()}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-lg border border-ink-700 hover:bg-ink-800 text-xs font-medium text-silver-300 transition-colors"
          >
            <X className="w-4 h-4" />
            <span>Back to Explorer</span>
          </button>
          <div className="flex-1">
            <h1 className="font-display text-lg font-semibold text-silver-100">
              <Layout className="w-5 h-5 text-gold-400 mr-2" />
              Raster Interpolation &amp; Gap-Filling Laboratory
            </h1>
            <p className="text-xs text-silver-500">
              Detect, characterize, reconstruct, validate, and export NoData regions in geospatial rasters.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-1 rounded text-xs font-semibold bg-ink-800 border border-ink-700 text-gold-400">
              Stage {currentIndex + 1} of {stageOrder.length}
            </span>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">
        {error && (
          <div className="mb-6 rounded-xl border border-red-900/40 bg-ink-900 p-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
              <div>
                <h4 className="font-semibold text-red-200">Error</h4>
                <p className="text-xs text-silver-400">{error}</p>
              </div>
            </div>
          </div>
        )}

        <div className="mb-6 flex items-center gap-2">
          {stageOrder.map((s, i) => {
            const isCompleted = i < currentIndex;
            const isActive = i === currentIndex;
            return (
              <div
                key={s}
                className={`flex-1 relative flex flex-col items-center gap-1 px-1 ${
                  isActive ? 'text-gold-400' : 'text-silver-500'
                }`}
              >
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold transition-colors ${
                    isCompleted
                      ? 'bg-gold-400 text-ink-950'
                      : isActive
                        ? 'bg-ink-800 border border-gold-400 text-gold-400'
                        : 'bg-ink-900 border border-ink-700'
                  }`}
                >
                  {i + 1}
                </div>
                {i < stageOrder.length - 1 && (
                  <div className={`absolute -top-2 left-1/2 w-full h-0.5 -translate-x-1/2 ${
                    isCompleted ? 'bg-gold-400' : 'bg-ink-800'
                  }`} />
                )}
                <span className="text-[10px] truncate w-16 text-center">{stageLabels[s]}</span>
              </div>
            );
          })}
        </div>

        <div className="space-y-6">
          {/* STAGE 1: Select Dataset */}
          {stage === 'select_dataset' && (
            <Panel title="Select Raster Dataset">
              <p className="text-sm text-silver-300 mb-4">
                No raster uploaded yet. Generating a synthetic continuous raster (64×64) with ~20% missing data
                for demonstration.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <RasterCanvas
                  data={rasterData ?? []}
                  title="Original Raster (null = NoData)"
                  colormap="viridis"
                />
                <div className="flex flex-col justify-center">
                  <p className="text-xs text-silver-400 mb-2">Raster info:</p>
                  <div className="text-xs text-silver-300 space-y-1">
                    <div>
                      Dimensions: {rasterMeta?.width ?? 0} × {rasterMeta?.height ?? 0}
                    </div>
                    <div>CRS: {rasterMeta?.crs ?? 'None'}</div>
                    <div>NoData: {rasterMeta?.nodata ?? 'None'}</div>
                  </div>
                </div>
              </div>
              <div className="mt-6 flex gap-3">
                <Button onClick={handleInspect} disabled={isLoading}>
                  {isLoading ? <Loader2 className="w-4 h-4" /> : <BarChart3 className="w-4 h-4" />}
                  Inspect Missing Data
                </Button>
                <Button variant="secondary" onClick={() => {
                  const synth = generateSyntheticRaster({
                    width: 64,
                    height: 64,
                    type: 'continuous',
                    missingPct: 0.3,
                    seed: 123,
                  });
                  setRasterData(
                    synth.data.map((row) =>
                      row.map((v) => (Number.isNaN(v) ? null : v))
                    )
                  );
                  setRasterMeta({
                    width: synth.width,
                    height: synth.height,
                    crs: synth.crs,
                    transform: synth.transform,
                    nodata: synth.nodata,
                  });
                  setInspection(null);
                  setStage('select_dataset');
                }}>
                  <RefreshCw className="w-4 h-4" /> New Synthetic Raster
                </Button>
              </div>
            </Panel>
          )}

          {/* STAGE 2: Inspect Missing Data */}
          {stage === 'inspect_missing' && inspection && (
            <Panel title="Missing Data Diagnostics">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                  <div className="text-xs text-silver-500">Valid pixels</div>
                  <div className="text-lg font-mono text-gold-400">{inspection.valid_pixel_count}</div>
                </div>
                <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                  <div className="text-xs text-silver-500">Missing pixels</div>
                  <div className="text-lg font-mono text-red-400">{inspection.missing_pixel_count}</div>
                </div>
                <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                  <div className="text-xs text-silver-500">Missing %</div>
                  <div className="text-lg font-mono text-red-400">{inspection.missing_percentage}%</div>
                </div>
                <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                  <div className="text-xs text-silver-500">Missing regions</div>
                  <div className="text-lg font-mono text-silver-300">{inspection.missing_region_count}</div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-sm font-semibold text-silver-300 mb-3">
                    Region size distribution
                  </h4>
                  <div className="space-y-2">
                    {Object.entries(inspection.missing_region_size_distribution).map(
                      ([key, val]) => (
                        <div key={key} className="flex justify-between text-sm">
                          <span className="text-silver-400">{key.replace(/_/g, ' ')}</span>
                          <span className="text-silver-300 font-mono">{val}</span>
                        </div>
                      )
                    )}
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-silver-300 mb-3">
                    Distance to nearest valid
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-silver-400">Min</span>
                      <span className="text-silver-300">{inspection.distance_to_valid_stats.min ?? 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-silver-400">Mean</span>
                      <span className="text-silver-300">{inspection.distance_to_valid_stats.mean ?? 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-silver-400">Max</span>
                      <span className="text-silver-300">{inspection.distance_to_valid_stats.max ?? 'N/A'}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex gap-3">
                <Button onClick={() => {
                  setRasterType(inspection.raster_type as InterpolationRasterType);
                  setStage('choose_method');
                }}>
                  <BarChart3 className="w-4 h-4" /> Choose Interpolation Method
                </Button>
                <Button variant="secondary" onClick={() => setStage('select_dataset')}>
                  <X className="w-4 h-4" /> Reselect Dataset
                </Button>
              </div>
            </Panel>
          )}

          {/* STAGE 3: Choose Method */}
          {stage === 'choose_method' && (
            <Panel title="Choose Interpolation Method">
              <p className="text-xs text-silver-500 mb-4">
                Raster type: <span className="text-gold-400 font-mono">{rasterType}</span>.{' '}
                {currentMethodInfo ? (
                  <span className="text-silver-300">{currentMethodInfo.description}</span>
                ) : null}
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {compatibleMethods.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => handleChooseMethod(m.id as InterpolationMethod)}
                    className={`flex flex-col items-center p-4 rounded-xl border transition-colors ${
                      method === m.id
                        ? 'bg-gold-400/20 border-gold-400 text-gold-400'
                        : m.compatible
                          ? 'bg-ink-900 border-ink-700 hover:border-ink-500 text-silver-300'
                          : 'bg-ink-900/50 border-ink-800 text-silver-600 cursor-not-allowed'
                    }`}
                    disabled={!m.compatible}
                  >
                    <div className="w-8 h-8 rounded-full flex items-center justify-center mb-2 ${
                      method === m.id ? 'bg-gold-400 text-ink-950' : 'bg-ink-800 border border-ink-700'
                    }">
                      <Cpu className="w-4 h-4" />
                    </div>
                    <div className="text-xs font-semibold text-center">{m.label}</div>
                    {!m.compatible && (
                      <div className="mt-1 text-[10px] text-red-400">Incompatible</div>
                    )}
                  </button>
                ))}
              </div>
              {currentMethodInfo && !currentMethodInfo.compatible && (
                <div className="mt-4 rounded-lg border border-red-900/40 bg-ink-900 p-3">
                  <p className="text-xs text-red-300">
                    <AlertCircle className="w-3 h-3 inline mr-1" />
                    {currentMethodInfo.description}
                  </p>
                </div>
              )}
              <div className="mt-6 flex gap-3">
                <Button onClick={() => handleConfigure()} disabled={!currentMethodInfo?.compatible}>
                  <Sliders className="w-4 h-4" /> Configure Parameters
                </Button>
                <Button variant="secondary" onClick={() => setStage('inspect_missing')}>
                  <X className="w-4 h-4" /> Back to Inspection
                </Button>
              </div>
            </Panel>
          )}

          {/* STAGE 4: Configure */}
          {stage === 'configure' && currentMethodInfo && (
            <Panel title="Configure Parameters">
              <div className="space-y-4">
                <Field label="Raster type">
                  <select
                    value={rasterType}
                    onChange={(e) => setRasterType(e.target.value as InterpolationRasterType)}
                    className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
                  >
                    <option value="continuous">Continuous</option>
                    <option value="categorical">Categorical</option>
                    <option value="unknown">Unknown</option>
                  </select>
                </Field>
                <MethodConfig
                  method={method}
                  config={config}
                  onChange={(k, v) => setConfig({ ...config, [k]: v })}
                />
                <div className="rounded-lg border border-ink-700 bg-ink-900/50 p-3">
                  <p className="text-xs text-silver-400">
                    <strong className="text-silver-300">Warning:</strong> Never modify original valid pixels.{' '}
                    {method === 'ordinary_kriging' &&
                      'Kriging variance is not a calibrated confidence interval.'}
                    {method === 'bicubic' &&
                      'Cubic interpolation may overshoot local observed range.'}
                  </p>
                </div>
              </div>
              <div className="mt-6 flex gap-3 justify-end">
                <Button variant="secondary" onClick={() => setStage('choose_method')}>
                  <X className="w-4 h-4" /> Back to Method
                </Button>
                <Button onClick={handleValidate}>
                  <Cpu className="w-4 h-4" /> Run Cross-Validation
                </Button>
              </div>
            </Panel>
          )}

          {/* STAGE 5: Validate */}
          {stage === 'validate' && validation && (
            <Panel title="Cross-Validation Results">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                {validation.mae !== null && (
                  <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                    <div className="text-xs text-silver-500">MAE</div>
                    <div className="text-lg font-mono text-silver-300">{validation.mae}</div>
                  </div>
                )}
                {validation.rmse !== null && (
                  <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                    <div className="text-xs text-silver-500">RMSE</div>
                    <div className="text-lg font-mono text-silver-300">{validation.rmse}</div>
                  </div>
                )}
                {validation.bias !== null && (
                  <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                    <div className="text-xs text-silver-500">Bias</div>
                    <div className="text-lg font-mono text-silver-300">{validation.bias}</div>
                  </div>
                )}
                {validation.r_squared !== null && (
                  <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                    <div className="text-xs text-silver-500">R²</div>
                    <div className="text-lg font-mono text-silver-300">{validation.r_squared}</div>
                  </div>
                )}
              </div>
              <div className="space-y-4">
                <div>
                  <div className="text-xs font-semibold text-silver-400 mb-2">
                    Validation summary
                  </div>
                  <div className="text-sm text-silver-300 space-y-1">
                    <div>
                      Strategy: <span className="font-mono">{validation.validation_strategy}</span>
                    </div>
                    <div>
                      Samples: <span className="font-mono">{validation.validation_sample_count}</span>
                    </div>
                    <div>
                      Reconstruction: <span className="font-mono">{validation.reconstruction_percentage}%</span>
                    </div>
                    <div>
                      Remaining NoData: <span className="font-mono">{validation.remaining_nodata_percentage}%</span>
                    </div>
                  </div>
                </div>
                {validation.warnings.length > 0 && (
                  <div className="rounded-lg border border-gold-900/40 bg-ink-900 p-3">
                    <div className="text-xs font-semibold text-gold-400 mb-1">Warnings</div>
                    <ul className="text-xs text-silver-400 space-y-1 list-disc list-inside">
                      {validation.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {validation.limitations.length > 0 && (
                  <div className="rounded-lg border border-silver-900/40 bg-ink-900 p-3">
                    <div className="text-xs font-semibold text-silver-400 mb-1">
                      Limitations
                    </div>
                    <ul className="text-xs text-silver-400 space-y-1 list-disc list-inside">
                      {validation.limitations.map((l, i) => (
                        <li key={i}>{l}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
              <div className="mt-6 flex gap-3">
                <Button variant="secondary" onClick={() => setStage('configure')}>
                  <Sliders className="w-4 h-4" /> Adjust Configuration
                </Button>
                <Button onClick={handleExecute}>
                  <Play className="w-4 h-4" /> Execute Interpolation
                </Button>
              </div>
            </Panel>
          )}

          {/* STAGE 6: Execute */}
          {stage === 'execute' && result && (
            <Panel title="Interpolation Complete">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                  <div className="text-xs text-silver-500">Observed</div>
                  <div className="text-lg font-mono text-silver-300">{result.observed_pixel_count}</div>
                </div>
                <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                  <div className="text-xs text-silver-500">Interpolated</div>
                  <div className="text-lg font-mono text-gold-400">{result.interpolated_pixel_count}</div>
                </div>
                <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                  <div className="text-xs text-silver-500">Remaining NoData</div>
                  <div className="text-lg font-mono text-red-400">{result.remaining_nodata_count}</div>
                </div>
                <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                  <div className="text-xs text-silver-500">Excluded</div>
                  <div className="text-lg font-mono text-silver-500">{result.excluded_pixel_count}</div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <RasterCanvas
                  data={rasterData ?? []}
                  title="Original (observed only)"
                  colormap="viridis"
                />
                <RasterCanvas
                  data={result.reconstructed as (number | null)[][]}
                  title="Reconstructed"
                  colormap="viridis"
                />
              </div>

              <div className="mt-6 flex items-center gap-4">
                <div className="flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={showProvenance}
                    onChange={(e) => setShowProvenance(e.target.checked)}
                    className="rounded bg-ink-900 border border-ink-700 text-gold-400 focus:ring-gold-400"
                  />
                  <label className="text-silver-400">Show provenance mask</label>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={showQuality}
                    onChange={(e) => setShowQuality(e.target.checked)}
                    disabled={!result.quality_mask}
                    className="rounded bg-ink-900 border border-ink-700 text-gold-400 focus:ring-gold-400"
                  />
                  <label className={`text-silver-400 ${!result.quality_mask ? 'opacity-50' : ''}`}>
                    Show quality layer
                  </label>
                </div>
              </div>

              {showProvenance && (
                <RasterCanvas
                  data={result.provenance_mask.map((row) =>
                    row.map((cls) => {
                      const m: Record<string, number> = { observed: 1, interpolated: 2, nodata: 3, excluded: 4 };
                      return m[cls] ?? 0;
                    })
                  ) as (number | null)[][]}
                  title="Provenance (1=observed, 2=interpolated, 3=nodata, 4=excluded)"
                  colormap="categorical"
                />
              )}

              <div className="mt-6 flex gap-3">
                <Button onClick={() => setStage('compare_results')}>
                  <Eye className="w-4 h-4" /> Compare Results
                </Button>
                <Button variant="secondary" onClick={() => setStage('validate')}>
                  <X className="w-4 h-4" /> Re-validate
                </Button>
              </div>
            </Panel>
          )}

          {/* STAGE 7: Compare Results */}
          {stage === 'compare_results' && result && (
            <Panel title="Original vs Reconstructed">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <RasterCanvas data={rasterData ?? []} title="Original" colormap="viridis" />
                <RasterCanvas data={result.reconstructed as (number | null)[][]} title="Reconstructed" colormap="viridis" />
                <RasterCanvas
                  data={result.provenance_mask.map((row) =>
                    row.map((cls) => {
                      const m: Record<string, number> = { observed: 1, interpolated: 2, nodata: 3, excluded: 4 };
                      return m[cls] ?? 0;
                    })
                  ) as (number | null)[][]}
                  title="Provenance Mask"
                  colormap="categorical"
                />
              </div>
              <div className="mt-6 rounded-lg border border-ink-700 bg-ink-900 p-4">
                <h4 className="text-sm font-semibold text-silver-300 mb-2">
                  Processing summary
                </h4>
                <div className="text-xs text-silver-400 space-y-1">
                  <div>
                    Method: <span className="font-mono">{result.method}</span>
                  </div>
                  <div>
                    Duration: <span className="font-mono">{result.processing_duration_ms} ms</span>
                  </div>
                  <div>
                    Observed: <span className="font-mono text-silver-300">{result.observed_pixel_count}</span> | Interpolated:{' '}
                    <span className="font-mono text-gold-400">{result.interpolated_pixel_count}</span>
                  </div>
                  <div>
                    Remaining NoData: <span className="font-mono text-red-400">{result.remaining_nodata_count}</span> | Excluded:{' '}
                    <span className="font-mono text-silver-500">{result.excluded_pixel_count}</span>
                  </div>
                </div>
              </div>
              <div className="mt-6 flex gap-3 justify-end">
                <Button variant="secondary" onClick={() => setStage('execute')}>
                  <X className="w-4 h-4" /> Back to Execution
                </Button>
                <Button onClick={() => setStage('export')}>
                  <Download className="w-4 h-4" /> Export Results
                </Button>
              </div>
            </Panel>
          )}

          {/* STAGE 8: Export */}
          {stage === 'export' && result && (
            <Panel title="Export Results">
              <div className="space-y-4">
                <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
                  <p className="text-sm text-silver-300 mb-3">Export options:</p>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 cursor-pointer text-sm">
                      <input type="checkbox" defaultChecked className="form-checkbox" />
                      Reconstructed GeoTIFF (with provenance & quality bands)
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer text-sm">
                      <input type="checkbox" defaultChecked className="form-checkbox" />
                      Provenance mask GeoTIFF
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer text-sm">
                      <input type="checkbox" className="form-checkbox" />
                      Quality report JSON
                    </label>
                  </div>
                </div>
                <div className="rounded-lg border border-ink-700 bg-ink-900 p-4">
                  <p className="text-xs text-silver-400 mb-2">Provenance note:</p>
                  <p className="text-sm text-silver-300">
                    Exported GeoTIFFs include a provenance band (1=observed, 2=interpolated, 3=nodata, 4=excluded)
                    so downstream analyses can distinguish original measurements from interpolated estimates.
                  </p>
                </div>
              </div>
              <div className="mt-6 flex gap-3 justify-end">
                <Button variant="secondary" onClick={() => setStage('compare_results')}>
                  <X className="w-4 h-4" /> Back to Comparison
                </Button>
                <Button onClick={() => {
                  const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const link = document.createElement('a');
                  link.href = url;
                  link.download = `heraoculus_interpolation_${result.method}_${Date.now()}.json`;
                  link.click();
                  URL.revokeObjectURL(url);
                }}>
                  <Download className="w-4 h-4" /> Download Result JSON
                </Button>
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
};
