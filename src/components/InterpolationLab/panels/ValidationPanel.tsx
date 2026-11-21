import React from 'react';
import { InterpolationStage } from '../../../types/earthScience';
import { Button } from '../../ui/Button';
import { Panel } from '../../ui/Panel';
import { Sliders, Play, X } from 'lucide-react';

interface ValidationPanelProps {
  stage: InterpolationStage;
  validation: any;
  handleExecute: () => Promise<void>;
  setStage: React.Dispatch<React.SetStateAction<InterpolationStage>>;
}

export function ValidationPanel({
  stage,
  validation,
  handleExecute,
  setStage,
}: ValidationPanelProps) {
  if (stage !== 'validate' || !validation) return null;

  return (
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
              {validation.warnings.map((w: any, i: number) => (
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
              {validation.limitations.map((l: any, i: number) => (
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
  );
}
