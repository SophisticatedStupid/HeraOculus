import React from 'react';
import { InterpolationStage } from '../../../types/earthScience';
import { Button } from '../../ui/Button';
import { Panel } from '../../ui/Panel';
import { Download, X } from 'lucide-react';

interface ExportPanelProps {
  stage: InterpolationStage;
  result: any;
  setStage: React.Dispatch<React.SetStateAction<InterpolationStage>>;
}

export function ExportPanel({
  stage,
  result,
  setStage,
}: ExportPanelProps) {
  if (stage !== 'export' || !result) return null;

  return (
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
  );
}
