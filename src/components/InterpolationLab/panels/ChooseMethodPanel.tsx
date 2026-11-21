import React from 'react';
import { CompatibleMethod, InterpolationMethod, InterpolationRasterType, InterpolationStage } from '../../../types/earthScience';
import { Button } from '../../ui/Button';
import { Cpu, Sliders, AlertCircle, X, BarChart3 } from 'lucide-react';
import { Panel } from '../../ui/Panel';

interface ChooseMethodPanelProps {
  stage: InterpolationStage;
  method: InterpolationMethod;
  rasterType: InterpolationRasterType;
  compatibleMethods: CompatibleMethod[];
  currentMethodInfo: CompatibleMethod | undefined;
  handleChooseMethod: (m: InterpolationMethod) => void;
  handleConfigure: () => void;
  setStage: React.Dispatch<React.SetStateAction<InterpolationStage>>;
}

export function ChooseMethodPanel({
  stage,
  method,
  rasterType,
  compatibleMethods,
  currentMethodInfo,
  handleChooseMethod,
  handleConfigure,
  setStage,
}: ChooseMethodPanelProps) {
  if (stage !== 'choose_method') return null;

  return (
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
  );
}
