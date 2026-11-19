import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import {
  AlertTriangle,
  Terminal,
  Info,
  Satellite,
  RefreshCw,
  Upload,
  FlaskConical,
  type LucideIcon,
} from 'lucide-react';
import { PROVENANCE_META, ProvenanceTier } from '../../types/earthScience';
import { BackendProvenance, Phase2OfflineError, Phase2SeriesHandoff } from '../../services/phase2Api';
import { tokens } from '../../theme/tokens';

ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

/**
 * Chart theming from semantic tokens. Chart.js cannot read Tailwind classes,
 * so these are the single source of truth for axis, grid and series colors.
 */
export const CHART_AXIS_TICK = tokens.silver500;
export const CHART_AXIS_LABEL = tokens.silver300;
export const CHART_GRID = 'rgba(192, 192, 192, 0.10)';
export const CHART_GRID_SOFT = 'rgba(192, 192, 192, 0.07)';

/** Series palette: gold-400 for primary, gold-400/silver-500 for opposing directions. */
export const CHART_SERIES = {
  gold400: tokens.gold400,
  silver500: tokens.silver500,
  gold400Alt: tokens.gold400,
  gold300: tokens.gold300,
  silver600: tokens.silver600,
} as const;

export const CHART_LEGEND = { labels: { color: CHART_AXIS_LABEL, font: { size: 11 } } };

const PROVENANCE_ICONS: Record<ProvenanceTier, LucideIcon> = {
  satellite_observation: Satellite,
  reanalysis_model: RefreshCw,
  user_upload: Upload,
  synthetic_demo: FlaskConical,
};

const PROVENANCE_ACCENT: Record<ProvenanceTier, { shell: string; label: string }> = {
  satellite_observation: {
    shell: 'bg-ink-900/80 border-silver-500/35 text-silver-100',
    label: 'text-silver-500',
  },
  reanalysis_model: {
    shell: 'bg-ink-900/80 border-gold-400/35 text-silver-100',
    label: 'text-gold-400',
  },
  user_upload: {
    shell: 'bg-ink-900/80 border-gold-300/35 text-silver-100',
    label: 'text-gold-300',
  },
  synthetic_demo: {
    shell: 'bg-ink-900/90 border-ink-700 text-silver-300',
    label: 'text-silver-600',
  },
};

export const ProvenanceBanner: React.FC<{ provenance: BackendProvenance }> = ({ provenance }) => {
  const meta = PROVENANCE_META[provenance.tier] ?? PROVENANCE_META.synthetic_demo;
  const tier = (provenance.tier in PROVENANCE_ICONS
    ? provenance.tier
    : 'synthetic_demo') as ProvenanceTier;
  const Icon = PROVENANCE_ICONS[tier];
  const accent = PROVENANCE_ACCENT[tier];

  return (
    <div
      className={`rounded-xl border px-3.5 py-2.5 text-xs flex flex-wrap items-center justify-between gap-2 ${accent.shell}`}
    >
      <div className="flex items-start sm:items-center gap-2.5">
        <Icon className={`w-4 h-4 shrink-0 mt-0.5 sm:mt-0 ${accent.label}`} />
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className={`font-semibold text-[10px] ${accent.label}`}>
              {meta.shortLabel}
            </span>
            <span className="font-semibold text-silver-100">{provenance.source_name}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-ink-950/70 border border-ink-800 text-silver-500 tabular-nums">
              {provenance.dataset_id}
            </span>
            {provenance.is_live && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-silver-500/15 border border-silver-500/40 text-silver-500 text-[10px] font-semibold">
                <span className="w-1.5 h-1.5 rounded-full bg-silver-500 animate-pulse" />
                Live API
              </span>
            )}
          </div>
          <p className="text-[11px] text-silver-300 mt-0.5">{provenance.status_message}</p>
        </div>
      </div>
      {provenance.citation && (
        <span className="text-[10px] text-silver-500">{provenance.citation}</span>
      )}
    </div>
  );
};

export const Phase2ErrorAlert: React.FC<{
  error: unknown;
  onRetry?: () => void;
}> = ({ error, onRetry }) => {
  if (!error) return null;
  const isOffline = error instanceof Phase2OfflineError;
  const msg = error instanceof Error ? error.message : String(error);

  return (
    <div
      className={`rounded-xl border p-4 text-xs space-y-2 ${
        isOffline
          ? 'bg-ink-900/80 border-gold-300/40 text-silver-100'
          : 'bg-ink-900/80 border-gold-400/40 text-silver-100'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2.5">
          {isOffline ? (
            <Terminal className="w-4 h-4 text-gold-300 shrink-0 mt-0.5" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-gold-400 shrink-0 mt-0.5" />
          )}
          <div>
            <div className="font-bold text-sm text-silver-100">
              {isOffline ? 'Python compute backend offline' : 'Computation error'}
            </div>
            <p className="mt-1 leading-relaxed text-silver-300">{msg}</p>
          </div>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-3 py-1 rounded-lg bg-ink-950 hover:bg-ink-800 border border-ink-700 text-silver-100 font-semibold shrink-0 cursor-pointer"
          >
            Retry
          </button>
        )}
      </div>
      {isOffline && (
        <div className="rounded-lg bg-ink-950/90 border border-ink-800 p-2.5 text-[11px] text-gold-400">
          cd backend &amp;&amp; pip install -r requirements.txt &amp;&amp; uvicorn app.main:app --reload --port 8000
        </div>
      )}
    </div>
  );
};

export const MetricTile: React.FC<{
  label: string;
  value: string | number;
  sub?: string;
  accent?: string;
  title?: string;
}> = ({ label, value, sub, accent = 'text-silver-100', title }) => (
  <div
    title={title}
    className="bg-ink-950/75 border border-ink-800 rounded-xl p-3 flex flex-col justify-between"
  >
    <span className="text-[10px] font-semibold text-silver-500">
      {label}
    </span>
    <div className={`text-lg font-black tabular-nums mt-1 ${accent}`}>{value}</div>
    {sub && <span className="text-[10px] text-silver-600 mt-0.5 tabular-nums">{sub}</span>}
  </div>
);

export const InfoBanner: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="rounded-xl bg-ink-900/70 border border-ink-800 px-3.5 py-2.5 text-[11px] text-silver-300 flex items-start gap-2">
    <Info className="w-4 h-4 text-gold-400 shrink-0 mt-0.5" />
    <div>{children}</div>
  </div>
);

export type BenchmarkSeriesPresetId = 'arctic_temp' | 'co2_seasonal' | 'extreme_precip';

export function buildBenchmarkSeries(preset: BenchmarkSeriesPresetId): Phase2SeriesHandoff {
  const dates: string[] = [];
  const values: (number | null)[] = [];
  const startYear = 2012;
  const totalMonths = 144; // 12 years monthly

  // Deterministic pseudo-random helper so results are reproducible
  let seed = preset === 'arctic_temp' ? 17 : preset === 'co2_seasonal' ? 31 : 73;
  const nextRand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296 - 0.5;
  };

  for (let i = 0; i < totalMonths; i++) {
    const year = startYear + Math.floor(i / 12);
    const month = (i % 12) + 1;
    dates.push(`${year}-${String(month).padStart(2, '0')}`);
    const tYr = i / 12;

    if (preset === 'arctic_temp') {
      // Polar amplification with 2019 acceleration & occasional missing satellite gaps
      const postBreak = Math.max(0, tYr - 6.5);
      const val =
        -11.8 +
        0.095 * tYr +
        0.08 * postBreak +
        11.2 * Math.sin((2 * Math.PI * (month - 4)) / 12) +
        nextRand() * 2.1;
      values.push(i === 52 || i === 109 ? null : Number(val.toFixed(2)));
    } else if (preset === 'co2_seasonal') {
      const val =
        393.5 +
        2.35 * tYr +
        0.025 * tYr * tYr +
        3.4 * Math.sin((2 * Math.PI * (month - 2)) / 12) +
        nextRand() * 0.7;
      values.push(Number(val.toFixed(2)));
    } else {
      // Heavy-tailed extreme monthly rainfall index (mm/day max)
      const u = Math.max(0.02, 0.5 + nextRand());
      const burst = u < 0.09 ? 38.0 / (u * 8.0) : 0;
      const val =
        42.0 +
        1.1 * tYr +
        26.0 * Math.max(0, Math.sin((2 * Math.PI * (month - 11)) / 12)) +
        burst +
        Math.abs(nextRand()) * 14.0;
      values.push(Number(val.toFixed(2)));
    }
  }

  const meta: Record<BenchmarkSeriesPresetId, { title: string; unit: string; regionName: string }> = {
    arctic_temp: {
      title: 'Arctic Shelf Surface Temperature Benchmark',
      unit: '°C',
      regionName: 'Svalbard / Barents Sea (Benchmark)',
    },
    co2_seasonal: {
      title: 'Global Atmospheric Column Benchmark',
      unit: 'ppm',
      regionName: 'Mauna Loa Baseline (Benchmark)',
    },
    extreme_precip: {
      title: 'Heavy-Tailed Coastal Precipitation Maxima',
      unit: 'mm',
      regionName: 'Western Sierra Watershed (Benchmark)',
    },
  };

  return {
    ...meta[preset],
    dates,
    values,
    provenanceTier: 'synthetic_demo',
  };
}
