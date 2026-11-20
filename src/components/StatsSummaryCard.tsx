import React from 'react';
import { PROVENANCE_META, ProvenanceTier, TrendAnalysisResult } from '../types/earthScience';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Percent,
  Activity,
  Binary,
  Satellite,
  RefreshCw,
  Upload,
  FlaskConical,
  Waves,
  CopyIcon,
  type LucideIcon,
} from 'lucide-react';
import { SignificanceBadge } from './SignificanceBadge';
import { IrisGauge } from './IrisGauge';

interface StatsSummaryCardProps {
  stats: TrendAnalysisResult;
}

const PROVENANCE_ICONS: Record<ProvenanceTier, LucideIcon> = {
  satellite_observation: Satellite,
  in_situ_gauge: Waves,
  reanalysis_model: RefreshCw,
  user_upload: Upload,
  synthetic_demo: FlaskConical,
};

const PROVENANCE_COLORS: Record<ProvenanceTier, string> = {
  satellite_observation: 'text-silver-500',
  in_situ_gauge: 'text-cyan-400',
  reanalysis_model: 'text-gold-400',
  user_upload: 'text-gold-300',
  synthetic_demo: 'text-silver-600',
};

export const StatsSummaryCard: React.FC<StatsSummaryCardProps> = ({ stats }) => {
  const getDirectionIcon = () => {
    if (stats.direction === 'Increasing') {
      return <TrendingUp className="w-5 h-5 text-gold-400" />;
    }
    if (stats.direction === 'Decreasing') {
      return <TrendingDown className="w-5 h-5 text-silver-500" />;
    }
    return <Minus className="w-5 h-5 text-silver-600" />;
  };

  const getDirectionColor = () => {
    if (stats.direction === 'Increasing') return 'text-gold-400';
    if (stats.direction === 'Decreasing') return 'text-silver-500';
    return 'text-silver-600';
  };

  const ProvenanceIcon = stats.provenance
    ? PROVENANCE_ICONS[stats.provenance]
    : null;

  return (
    <div className="space-y-4">
      {/* Core trend metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Rate of change */}
        <div className="bg-ink-900 rounded-2xl border border-ink-700 p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-silver-500">Rate of change</span>
            <div className="p-1.5 rounded-lg bg-ink-950 border border-ink-700">
              {getDirectionIcon()}
            </div>
          </div>

          <div className="my-2">
            <div className="flex items-baseline space-x-1.5">
              <span
                className={`text-2xl sm:text-3xl font-semibold tabular-nums tracking-tight ${getDirectionColor()}`}
              >
                {stats.slopePerDecade > 0 ? '+' : ''}
                {stats.slopePerDecade.toFixed(2)}
              </span>
              <span className="text-xs text-silver-500 font-medium">{stats.unit}/decade</span>
            </div>
            <p className="text-[11px] text-silver-500 tabular-nums mt-0.5">
              Annual: {stats.slopePerYear > 0 ? '+' : ''}
              {stats.slopePerYear.toFixed(3)} {stats.unit}/yr
            </p>
          </div>

          <div className="pt-2 border-t border-ink-700 flex items-center justify-between text-[11px]">
            <span className="text-silver-500">Theil–Sen slope</span>
            <span className="tabular-nums text-gold-400 font-semibold">
              {stats.sensSlopePerDecade > 0 ? '+' : ''}
              {stats.sensSlopePerDecade.toFixed(2)} {stats.unit}/dec
            </span>
          </div>
        </div>

        {/* Cumulative shift */}
        <div className="bg-ink-900 rounded-2xl border border-ink-700 p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-silver-500">Cumulative shift</span>
            <div className="p-1.5 rounded-lg bg-ink-950 border border-ink-700 text-gold-400">
              <Percent className="w-4 h-4" />
            </div>
          </div>

          <div className="my-2">
            <div className="flex items-baseline space-x-1.5">
              <span className="text-2xl sm:text-3xl font-semibold tabular-nums text-silver-100 tracking-tight">
                {stats.totalChange > 0 ? '+' : ''}
                {stats.totalChange.toFixed(2)}
              </span>
              <span className="text-xs text-silver-500 font-medium">{stats.unit}</span>
            </div>
            <p className="text-[11px] text-silver-500 mt-0.5 flex items-center space-x-1">
              <span>Total departure:</span>
              <strong
                className={`tabular-nums ${
                  stats.percentChange >= 0 ? 'text-gold-400' : 'text-silver-500'
                }`}
              >
                {stats.percentChange >= 0 ? '+' : ''}
                {stats.percentChange.toFixed(1)}%
              </strong>
            </p>
          </div>

          <div className="pt-2 border-t border-ink-700 flex items-center justify-between text-[11px]">
            <span className="text-silver-500">Baseline window</span>
            <span className="tabular-nums text-silver-300 text-right">
              {stats.startDate} → {stats.endDate}
            </span>
          </div>
          {(stats.requestedRange || stats.actualRange) && (
            <div className="pt-2 border-t border-ink-700 flex items-center justify-between text-[11px]">
              <span className="text-silver-500">Time range</span>
              <span className="tabular-nums text-silver-300 text-right">
                {stats.requestedRange ? `Req: ${stats.requestedRange}` : ''}
                {stats.actualRange ? `Actual: ${stats.actualRange}` : ''}
              </span>
            </div>
          )}
          {stats.latestValidObservation && (
            <div className="pt-2 border-t border-ink-700 flex items-center justify-between text-[11px]">
              <span className="text-silver-500">Latest valid observation</span>
              <span className="tabular-nums text-silver-300 text-right">
                {stats.latestValidObservation}
              </span>
            </div>
          )}
        </div>

        {/* Mann-Kendall */}
        <div className="bg-ink-900 rounded-2xl border border-ink-700 p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-silver-500">Mann–Kendall test</span>
            <div className="p-1.5 rounded-lg bg-ink-950 border border-ink-700 text-gold-400">
              <Activity className="w-4 h-4" />
            </div>
          </div>

          <div className="my-2 flex items-center justify-between gap-3">
            <div>
              <div className="flex items-baseline space-x-1.5">
                <span className="text-xl sm:text-2xl font-semibold tabular-nums text-gold-400 tracking-tight">
                  Z = {stats.mkZ > 0 ? '+' : ''}
                  {stats.mkZ.toFixed(2)}
                </span>
              </div>
              <p className="text-[11px] text-silver-500 mt-0.5 tabular-nums">
                Kendall τ = {stats.mkTau > 0 ? '+' : ''}
                {stats.mkTau.toFixed(3)}
              </p>
            </div>
            <IrisGauge
              pValue={stats.mkPValue}
              direction={
                stats.direction === 'Increasing'
                  ? 'increasing'
                  : stats.direction === 'Decreasing'
                    ? 'decreasing'
                    : 'none'
              }
              size={64}
            />
          </div>

          <div className="pt-2 border-t border-ink-700 flex items-center justify-between text-[11px]">
            <span className="text-silver-500">p-value</span>
            <span className="tabular-nums text-silver-100 font-semibold">
              {stats.mkPValue < 0.001 ? '< 0.001 (***)' : `= ${stats.mkPValue.toFixed(4)}`}
            </span>
          </div>
        </div>

        {/* Regression & t-test */}
        <div className="bg-ink-900 rounded-2xl border border-ink-700 p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-silver-500">Regression fit & t-test</span>
            <div className="p-1.5 rounded-lg bg-ink-950 border border-ink-700 text-silver-300">
              <Binary className="w-4 h-4" />
            </div>
          </div>

          <div className="my-2">
            <div className="flex items-baseline space-x-1.5">
              <span className="text-2xl font-semibold tabular-nums text-silver-100 tracking-tight">
                R² = {stats.rSquared.toFixed(3)}
              </span>
            </div>
            <p className="text-[11px] text-silver-500 mt-0.5 tabular-nums">
              t-stat = {stats.tStatistic.toFixed(2)} (p{' '}
              {stats.tPValue < 0.001 ? '< 0.001' : `= ${stats.tPValue.toFixed(3)}`})
            </p>
          </div>

          <div className="pt-2 border-t border-ink-700 flex items-center justify-between text-[11px]">
            <span className="text-silver-500">Std error</span>
            <span className="tabular-nums text-silver-300">
              ±{stats.standardError.toFixed(2)} {stats.unit}
            </span>
          </div>
        </div>
      </div>

      {/* Scientific verdict */}
      <div className="bg-ink-900 rounded-2xl border border-ink-700 p-4 sm:p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-3 max-w-2xl w-full">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs font-medium text-silver-500">Scientific verdict</span>
            <SignificanceBadge stats={stats} />
            {stats.provenance && ProvenanceIcon && (
              <span
                className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-lg text-[10px] font-medium border border-ink-700 bg-ink-950 ${PROVENANCE_COLORS[stats.provenance]}`}
              >
                <ProvenanceIcon className="w-3 h-3" />
                {PROVENANCE_META[stats.provenance].label}
              </span>
            )}
          </div>
          <p className="text-xs sm:text-sm text-silver-100 leading-relaxed font-medium">
            {stats.plainEnglishVerdict}
          </p>
          {stats.statusMessage && (
            <p className="text-[11px] text-silver-500">
              Source{stats.provenanceSource ? ` (${stats.provenanceSource})` : ''}:{' '}
              {stats.statusMessage}
            </p>
          )}
        </div>

        <div className="w-full md:w-auto p-3 rounded-xl bg-ink-950 border border-ink-700 text-xs shrink-0 space-y-1">
          <span className="text-[10px] text-silver-500 font-medium block">Sampling integrity</span>
          <div className="flex items-center justify-between md:justify-start space-x-3 text-silver-300 tabular-nums">
            <span>
              N = <strong className="text-silver-100">{stats.sampleCount}</strong> valid
            </span>
            {(stats.missingCount ?? 0) > 0 && (
              <>
                <span className="text-ink-700">•</span>
                <span className="text-gold-300">{stats.missingCount} missing</span>
              </>
            )}
            <span className="text-ink-700">•</span>
            <span>Monthly resolution</span>
          </div>
        </div>
      </div>
    </div>
  );
};
