import React from 'react';
import { InterpolationMethod } from '../../types/earthScience';
import { Field } from '../ui/Field';

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

export default MethodConfig;
