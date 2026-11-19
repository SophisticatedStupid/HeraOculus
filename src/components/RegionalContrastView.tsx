import React, { useState } from 'react';
import { REGIONAL_CONTRASTS } from '../services/regionalContrasts';
import { HOTSPOT_REGIONS, EARTH_VARIABLES, generateCuratedTimeSeries } from '../services/nasaData';
import { analyzeTrend } from '../services/statistics';
import { TimeSeriesChart } from './TimeSeriesChart';
import { SignificanceBadge } from './SignificanceBadge';
import { GitCompare, Globe2 } from 'lucide-react';

function slopeClass(slope: number): string {
  if (slope > 0) return 'text-gold-400';
  if (slope < 0) return 'text-silver-500';
  return 'text-silver-600';
}

export const RegionalContrastView: React.FC = () => {
  const [selectedScenarioIndex, setSelectedScenarioIndex] = useState<number>(0);
  const scenario = REGIONAL_CONTRASTS[selectedScenarioIndex];

  const [customRegionAId, setCustomRegionAId] = useState<string>(scenario.regionA.hotspotId);
  const [customRegionBId, setCustomRegionBId] = useState<string>(scenario.regionB.hotspotId);
  const [customVariableId, setCustomVariableId] = useState<string>(scenario.variableId);

  const handleSelectScenario = (index: number) => {
    setSelectedScenarioIndex(index);
    setCustomRegionAId(REGIONAL_CONTRASTS[index].regionA.hotspotId);
    setCustomRegionBId(REGIONAL_CONTRASTS[index].regionB.hotspotId);
    setCustomVariableId(REGIONAL_CONTRASTS[index].variableId);
  };

  const regionA = HOTSPOT_REGIONS.find(r => r.id === customRegionAId) || HOTSPOT_REGIONS[0];
  const regionB = HOTSPOT_REGIONS.find(r => r.id === customRegionBId) || HOTSPOT_REGIONS[1];
  const variable = EARTH_VARIABLES.find(v => v.id === customVariableId) || EARTH_VARIABLES[0];

  const pointsA = React.useMemo(() => {
    return generateCuratedTimeSeries(regionA, variable);
  }, [regionA, variable]);

  const statsA = React.useMemo(() => {
    return analyzeTrend(pointsA, variable.id, variable.name, variable.unit, regionA.name, regionA.coordinates, true);
  }, [pointsA, variable, regionA]);

  const pointsB = React.useMemo(() => {
    return generateCuratedTimeSeries(regionB, variable);
  }, [regionB, variable]);

  const statsB = React.useMemo(() => {
    return analyzeTrend(pointsB, variable.id, variable.name, variable.unit, regionB.name, regionB.coordinates, true);
  }, [pointsB, variable, regionB]);

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <div className="flex flex-col items-start justify-between gap-4 rounded-2xl border border-ink-700 bg-ink-900 p-6 md:flex-row md:items-center">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1 rounded-full border border-gold-400/30 bg-ink-800 px-2.5 py-0.5 text-[11px] font-semibold text-gold-400">
              <GitCompare className="h-3 w-3" />
              <span>NASA challenge core feature</span>
            </span>
            <span className="text-xs text-silver-500">Opposite regional trends engine</span>
          </div>
          <h2 className="font-display text-2xl font-semibold tracking-tight text-silver-100">
            One planetary driver, two opposite regional trends
          </h2>
          <p className="max-w-2xl text-xs leading-relaxed text-silver-300 sm:text-sm">
            A measured Earth science data variable can trend one way in one region and the opposite way in another, even when the same process drives both.
          </p>
        </div>

        <div className="flex w-full flex-col gap-2 md:w-auto">
          <span className="text-[11px] font-semibold text-silver-500">
            Curated NASA case scenarios
          </span>
          <div className="flex flex-wrap gap-2">
            {REGIONAL_CONTRASTS.map((sc, idx) => (
              <button
                key={sc.id}
                type="button"
                onClick={() => handleSelectScenario(idx)}
                className={`rounded-xl px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-none ${
                  selectedScenarioIndex === idx
                    ? 'bg-gold-400 text-ink-950'
                    : 'border border-ink-700 bg-ink-950 text-silver-500 hover:bg-ink-800 hover:text-silver-100'
                }`}
              >
                {sc.title.split(':')[0]}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="space-y-3 rounded-2xl border border-ink-700 bg-ink-900 p-5 sm:p-6">
        <div className="flex items-center gap-2.5">
          <div className="rounded-xl border border-gold-400/30 bg-ink-800 p-2 text-gold-400">
            <Globe2 className="h-5 w-5" />
          </div>
          <div>
            <span className="text-[10px] font-semibold text-gold-400">
              Common planetary process
            </span>
            <h3 className="font-display text-lg font-semibold text-silver-100">{scenario.commonDriver}</h3>
          </div>
        </div>

        <p className="rounded-xl border border-ink-700 bg-ink-950/80 p-4 text-xs leading-relaxed text-silver-300 sm:text-sm">
          {scenario.description}
        </p>

        <div className="rounded-xl border border-gold-400/30 bg-gold-400/5 p-3 text-xs text-silver-300">
          <strong className="text-silver-100">Scientific takeaway:</strong> {scenario.scientificTakeaway}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Region A */}
        <div className="space-y-4 rounded-2xl border border-ink-700 bg-ink-900 p-5">
          <div className="flex items-start justify-between border-b border-ink-700 pb-3">
            <div>
              <span className="text-[10px] font-semibold text-silver-500">
                Region A (trend 1)
              </span>
              <h4 className="font-display text-base font-semibold text-silver-100">{regionA.name}</h4>
              <p className="tabular-nums text-xs text-silver-500">
                {regionA.coordinates[0].toFixed(2)}°, {regionA.coordinates[1].toFixed(2)}° · {regionA.countryOrOcean}
              </p>
            </div>
            <SignificanceBadge stats={statsA} />
          </div>

          <div className="flex items-center justify-between rounded-xl border border-ink-700 bg-ink-950 p-3.5">
            <div>
              <span className="mb-0.5 block text-[11px] font-semibold text-silver-500">
                Observed rate of change
              </span>
              <span className={`text-xl font-semibold tabular-nums ${slopeClass(statsA.slopePerDecade)}`}>
                {statsA.slopePerDecade > 0 ? '+' : ''}{statsA.slopePerDecade.toFixed(2)} {variable.unit}/decade
              </span>
            </div>
            <div className="text-right text-xs tabular-nums text-silver-500">
              <span>Total: {statsA.totalChange > 0 ? '+' : ''}{statsA.totalChange.toFixed(1)} {variable.unit}</span>
              <span className="block text-[11px] text-silver-500">MK Z = {statsA.mkZ}</span>
            </div>
          </div>

          <p className="rounded-lg border border-ink-700 bg-ink-950/60 p-3 text-xs leading-relaxed text-silver-300">
            <strong className="text-silver-100">Local mechanism:</strong> {scenario.regionA.mechanism}
          </p>

          <TimeSeriesChart
            points={pointsA}
            variable={variable}
            regionName={regionA.name}
            stats={statsA}
          />
        </div>

        {/* Region B */}
        <div className="space-y-4 rounded-2xl border border-ink-700 bg-ink-900 p-5">
          <div className="flex items-start justify-between border-b border-ink-700 pb-3">
            <div>
              <span className="text-[10px] font-semibold text-gold-400">
                Region B (trend 2)
              </span>
              <h4 className="font-display text-base font-semibold text-silver-100">{regionB.name}</h4>
              <p className="tabular-nums text-xs text-silver-500">
                {regionB.coordinates[0].toFixed(2)}°, {regionB.coordinates[1].toFixed(2)}° · {regionB.countryOrOcean}
              </p>
            </div>
            <SignificanceBadge stats={statsB} />
          </div>

          <div className="flex items-center justify-between rounded-xl border border-ink-700 bg-ink-950 p-3.5">
            <div>
              <span className="mb-0.5 block text-[11px] font-semibold text-silver-500">
                Observed rate of change
              </span>
              <span className={`text-xl font-semibold tabular-nums ${slopeClass(statsB.slopePerDecade)}`}>
                {statsB.slopePerDecade > 0 ? '+' : ''}{statsB.slopePerDecade.toFixed(2)} {variable.unit}/decade
              </span>
            </div>
            <div className="text-right text-xs tabular-nums text-silver-500">
              <span>Total: {statsB.totalChange > 0 ? '+' : ''}{statsB.totalChange.toFixed(1)} {variable.unit}</span>
              <span className="block text-[11px] text-silver-500">MK Z = {statsB.mkZ}</span>
            </div>
          </div>

          <p className="rounded-lg border border-ink-700 bg-ink-950/60 p-3 text-xs leading-relaxed text-silver-300">
            <strong className="text-silver-100">Local mechanism:</strong> {scenario.regionB.mechanism}
          </p>

          <TimeSeriesChart
            points={pointsB}
            variable={variable}
            regionName={regionB.name}
            stats={statsB}
          />
        </div>
      </div>
    </div>
  );
};
