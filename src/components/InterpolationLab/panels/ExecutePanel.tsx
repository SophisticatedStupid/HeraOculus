import React from 'react';
import { InterpolationStage } from '../../../types/earthScience';
import { Button } from '../../ui/Button';
import { Panel } from '../../ui/Panel';
import { Eye, X } from 'lucide-react';
import RasterCanvas from '../RasterCanvas';

interface ExecutePanelProps {
  stage: InterpolationStage;
  result: any;
  rasterData: ((number | null)[][]) | null;
  showProvenance: boolean;
  setShowProvenance: React.Dispatch<React.SetStateAction<boolean>>;
  showQuality: boolean;
  setShowQuality: React.Dispatch<React.SetStateAction<boolean>>;
  setStage: React.Dispatch<React.SetStateAction<InterpolationStage>>;
}

export function ExecutePanel({
  stage,
  result,
  rasterData,
  showProvenance,
  setShowProvenance,
  showQuality,
  setShowQuality,
  setStage,
}: ExecutePanelProps) {
  if (stage !== 'execute' || !result) return null;

  return (
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
          data={result.provenance_mask.map((row: any) =>
            row.map((cls: string) => {
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
  );
}
