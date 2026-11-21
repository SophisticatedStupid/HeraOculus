import React from 'react';
import { InterpolationMethod, InterpolationRasterType, InterpolationStage } from '../../../types/earthScience';
import { Button } from '../../ui/Button';
import { Field } from '../../ui/Field';
import { Panel } from '../../ui/Panel';
import { Cpu, X, Play, Sliders } from 'lucide-react';
import MethodConfig from '../MethodConfig';

interface ConfigurePanelProps {
  stage: InterpolationStage;
  method: InterpolationMethod;
  rasterType: InterpolationRasterType;
  setRasterType: React.Dispatch<React.SetStateAction<InterpolationRasterType>>;
  config: Record<string, any>;
  setConfig: React.Dispatch<React.SetStateAction<Record<string, any>>>;
  currentMethodInfo: any;
  handleValidate: () => Promise<void>;
  setStage: React.Dispatch<React.SetStateAction<InterpolationStage>>;
}

export function ConfigurePanel({
  stage,
  method,
  rasterType,
  setRasterType,
  config,
  setConfig,
  currentMethodInfo,
  handleValidate,
  setStage,
}: ConfigurePanelProps) {
  if (stage !== 'configure' || !currentMethodInfo) return null;

  return (
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
  );
}
