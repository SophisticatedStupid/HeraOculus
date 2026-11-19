import React from 'react';
import { EarthVariable, HotspotRegion, TrendAnalysisResult } from '../types/earthScience';
import {
  Sparkles,
  Satellite,
  Globe2,
  ArrowUpRight,
  AlertCircle,
} from 'lucide-react';

interface PlainEnglishSummaryProps {
  stats: TrendAnalysisResult;
  variable: EarthVariable;
  region: HotspotRegion;
}

export const PlainEnglishSummary: React.FC<PlainEnglishSummaryProps> = ({
  stats,
  variable,
  region,
}) => {
  const isSignificant =
    stats.significanceLevel === 'Extremely Significant' ||
    stats.significanceLevel === 'Highly Significant' ||
    stats.significanceLevel === 'Significant';

  return (
    <div className="bg-ink-900 rounded-2xl border border-ink-700 p-5 sm:p-6 space-y-5">
      <div className="flex items-center justify-between pb-3 border-b border-ink-700">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-ink-950 text-gold-400 border border-ink-700">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-display text-base font-semibold text-silver-100">
              Data detective analysis & real-world impact
            </h3>
            <p className="text-xs text-silver-500">
              Connecting satellite math to physical planetary processes
            </p>
          </div>
        </div>

        <span className="hidden sm:inline-flex px-2.5 py-1 rounded-lg text-[11px] font-medium bg-ink-950 text-gold-400 border border-ink-700">
          Target: {region.name}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-ink-950 rounded-xl border border-ink-700 p-4 space-y-2">
          <div className="flex items-center space-x-2 text-xs font-medium text-gold-400">
            <Globe2 className="w-4 h-4" />
            <span>Underlying process</span>
          </div>
          <p className="text-xs text-silver-300 leading-relaxed">{stats.scientificImplication}</p>
          <div className="pt-2 text-[11px] text-silver-500">
            <strong className="text-silver-300">Key driver:</strong> {region.keyPhenomenon}
          </div>
        </div>

        <div className="bg-ink-950 rounded-xl border border-ink-700 p-4 space-y-2">
          <div className="flex items-center space-x-2 text-xs font-medium text-silver-500">
            <Satellite className="w-4 h-4" />
            <span>Observing missions</span>
          </div>
          <p className="text-xs text-silver-300 leading-relaxed">
            Observed via <strong className="text-silver-100">{variable.missions.join(', ')}</strong>{' '}
            utilizing {variable.measurementType.toLowerCase()} techniques.
          </p>
          <div className="pt-2 text-[11px] text-silver-500 bg-ink-900 p-2 rounded-lg border border-ink-700">
            <strong className="text-silver-300">Sensor specs:</strong> {variable.description}
          </div>
        </div>

        <div className="bg-ink-950 rounded-xl border border-ink-700 p-4 space-y-2">
          <div className="flex items-center space-x-2 text-xs font-medium text-gold-300">
            <AlertCircle className="w-4 h-4" />
            <span>Why significance matters</span>
          </div>
          <p className="text-xs text-silver-300 leading-relaxed">
            {isSignificant ? (
              <>
                Because <strong className="text-silver-100">p &lt; 0.05</strong>, scientists can
                conclusively state that this change is not random chance or an artifact of weather
                cycles. It represents sustained directional forcing on the Earth system.
              </>
            ) : (
              <>
                Because <strong className="text-silver-100">p ≥ 0.05</strong>, the trend cannot yet be
                separated with high statistical confidence from natural climate variability (like
                ENSO or volcanic spikes). A longer time-series baseline is required.
              </>
            )}
          </p>
          <div className="pt-2 text-[11px] text-silver-500">
            Confidence:{' '}
            <strong className="text-silver-300">
              {stats.significanceLevel} ({stats.significanceStars})
            </strong>
          </div>
        </div>
      </div>

      {region.oppositeExplanation && (
        <div className="p-3.5 bg-ink-950 rounded-xl border border-ink-700 flex items-start space-x-3 text-xs text-silver-300">
          <ArrowUpRight className="w-4 h-4 text-gold-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-silver-100 block mb-0.5">
              The opposite region paradox
            </span>
            <p className="leading-relaxed">
              {region.oppositeExplanation} Check out the{' '}
              <strong className="text-silver-100">Opposite trends lab</strong> tab in the navigation
              bar to compare them side-by-side.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
