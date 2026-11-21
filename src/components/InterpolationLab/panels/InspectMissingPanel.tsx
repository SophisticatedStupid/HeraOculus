import React from 'react';
import { InterpolationRasterType, InterpolationStage } from '../../../types/earthScience';
import { Button } from '../../ui/Button';
import { BarChart3, X } from 'lucide-react';
import { Panel } from '../../ui/Panel';

interface InspectMissingPanelProps {
  stage: InterpolationStage;
  inspection: any;
  rasterType: InterpolationRasterType;
  setRasterType: React.Dispatch<React.SetStateAction<InterpolationRasterType>>;
  setStage: React.Dispatch<React.SetStateAction<InterpolationStage>>;
}

export function InspectMissingPanel({
  stage,
  inspection,
  rasterType,
  setRasterType,
  setStage,
}: InspectMissingPanelProps) {
  if (stage !== 'inspect_missing' || !inspection) return null;

  return (
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
                  <span className="text-silver-300 font-mono">{String(val)}</span>
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
  );
}
