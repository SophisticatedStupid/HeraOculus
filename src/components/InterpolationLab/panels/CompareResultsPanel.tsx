import React from 'react';
import { InterpolationStage } from '../../../types/earthScience';
import { Button } from '../../ui/Button';
import { Panel } from '../../ui/Panel';
import { Download, X } from 'lucide-react';
import RasterCanvas from '../RasterCanvas';

interface CompareResultsPanelProps {
  stage: InterpolationStage;
  result: any;
  rasterData: ((number | null)[][]) | null;
  setStage: React.Dispatch<React.SetStateAction<InterpolationStage>>;
}

export function CompareResultsPanel({
  stage,
  result,
  rasterData,
  setStage,
}: CompareResultsPanelProps) {
  if (stage !== 'compare_results' || !result) return null;

  return (
    <Panel title="Original vs Reconstructed">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <RasterCanvas data={rasterData ?? []} title="Original" colormap="viridis" />
        <RasterCanvas data={result.reconstructed as (number | null)[][]} title="Reconstructed" colormap="viridis" />
        <RasterCanvas
          data={result.provenance_mask.map((row: any) =>
            row.map((cls: string) => {
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
  );
}
