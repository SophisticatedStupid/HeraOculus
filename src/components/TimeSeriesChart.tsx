import React, { useState, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { EarthVariable, PROVENANCE_META, TimeSeriesPoint, TrendAnalysisResult } from '../types/earthScience';
import {
  Eye,
  EyeOff,
  Download,
  TrendingUp,
  Activity,
  Calendar,
  Layers,
  Sparkles,
  Loader2
} from 'lucide-react';
import { tokens, trendColor } from '../theme/tokens';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface TimeSeriesChartProps {
  points: TimeSeriesPoint[];
  variable: EarthVariable;
  regionName: string;
  stats: TrendAnalysisResult;
  isLoading?: boolean;
  loadingMessage?: string | null;
  loadingElapsedSeconds?: number;
}

const silver600Soft = `${tokens.silver600}73`; // ~45%
const silver600Fill = `${tokens.silver600}1F`; // ~12%
const silver500Soft = `${tokens.silver500}14`;
const mistAxis = tokens.silver500;
const gridLine = `${tokens.ink700}40`;

export const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  points,
  variable,
  regionName,
  stats,
  isLoading = false,
  loadingMessage,
  loadingElapsedSeconds,
}) => {
  const chartRef = useRef<any>(null);

  // Toggle states
  const [showRaw, setShowRaw] = useState<boolean>(true);
  const [showMovingAvg, setShowMovingAvg] = useState<boolean>(true);
  const [showAnomaly, setShowAnomaly] = useState<boolean>(false);
  const [showTrend, setShowTrend] = useState<boolean>(true);
  const [showCI, setShowCI] = useState<boolean>(true);

  // Range filter
  const [timeRange, setTimeRange] = useState<'all' | '20y' | '10y'>('all');

  const firstYear = points[0]?.year;
  const lastYear = points[points.length - 1]?.year;
  const totalYears = firstYear && lastYear ? Math.max(1, lastYear - firstYear + 1) : 0;

  const rangeOptions = [
    { key: 'all' as const, label: totalYears > 0 ? `All (${totalYears}y)` : 'All' },
    ...(totalYears > 20 && lastYear
      ? [{ key: '20y' as const, label: `${lastYear - 20}–${lastYear}` }]
      : []),
    ...(totalYears > 10 && lastYear
      ? [{ key: '10y' as const, label: `${lastYear - 10}–${lastYear}` }]
      : []),
  ];

  const activeRange = rangeOptions.some(opt => opt.key === timeRange) ? timeRange : 'all';

  // Filter points based on selected range
  const filteredPoints = React.useMemo(() => {
    if (points.length === 0) return [];
    if (activeRange === '10y') {
      const minYear = points[points.length - 1].year - 10;
      return points.filter(p => p.year >= minYear);
    }
    if (activeRange === '20y') {
      const minYear = points[points.length - 1].year - 20;
      return points.filter(p => p.year >= minYear);
    }
    return points;
  }, [points, activeRange]);

  const visibleStart = filteredPoints[0]?.date ?? stats.startDate;
  const visibleEnd = filteredPoints[filteredPoints.length - 1]?.date ?? stats.endDate;
  const visibleCount = filteredPoints.length;

  const labels = filteredPoints.map(p => p.date);
  const trendStroke = stats.mkPValue < 0.05 ? tokens.gold400 : tokens.silver300;
  const trendDash = stats.mkPValue < 0.05 ? [] : [6, 4];
  const ciFill = stats.mkPValue < 0.05 ? 'rgba(212, 175, 55, 0.12)' : 'rgba(192, 192, 192, 0.12)';

  // Datasets preparation
  const datasets: any[] = [];

  // 1. Raw Observations
  if (showRaw && !showAnomaly) {
    datasets.push({
      label: `Raw Monthly ${variable.shortName} (${variable.unit})`,
      data: filteredPoints.map(p => p.raw),
      borderColor: silver600Soft,
      backgroundColor: ciFill,
      borderWidth: 1.2,
      pointRadius: 1,
      pointHoverRadius: 4,
      tension: 0.2,
      spanGaps: false,
      order: 3,
    });
  }

  // 2. 12-Month Moving Average
  if (showMovingAvg && !showAnomaly) {
    datasets.push({
      label: '12-Month Moving Average',
      data: filteredPoints.map(p => p.movingAvg ?? null),
      borderColor: variable.color || tokens.gold400,
      backgroundColor: 'transparent',
      borderWidth: 2.2,
      pointRadius: 0,
      pointHoverRadius: 5,
      tension: 0.3,
      spanGaps: false,
      order: 2,
    });
  }

  // 3. Deseasonalized Baseline Anomaly
  if (showAnomaly) {
    datasets.push({
      label: `Deseasonalized Anomaly (${variable.unit} from 30-yr monthly norm)`,
      data: filteredPoints.map(p => p.anomaly ?? null),
      borderColor: tokens.silver500,
      backgroundColor: silver500Soft,
      fill: true,
      borderWidth: 2,
      pointRadius: 1.5,
      pointHoverRadius: 5,
      tension: 0.25,
      spanGaps: false,
      order: 2,
    });
  }

  // 4. Linear Trend Line
  if (showTrend) {
    datasets.push({
      label: `OLS Trend Line (${stats.slopePerDecade > 0 ? '+' : ''}${stats.slopePerDecade.toFixed(2)} ${variable.unit}/decade)`,
      data: filteredPoints.map(p => p.trend),
      borderColor: trendStroke,
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      borderDash: trendDash,
      pointRadius: 0,
      tension: 0,
      order: 1,
    });
  }

  // 5. 95% Confidence Interval Upper Bound
  if (showCI && showTrend) {
    datasets.push({
      label: '95% CI Upper Bound',
      data: filteredPoints.map(p => p.ciUpper),
      borderColor: 'transparent',
      backgroundColor: ciFill,
      fill: '+1', // fill down to the next dataset (CI Lower Bound)
      pointRadius: 0,
      tension: 0,
      order: 4,
    });

    // 6. 95% Confidence Interval Lower Bound
    datasets.push({
      label: '95% CI Lower Bound',
      data: filteredPoints.map(p => p.ciLower),
      borderColor: 'transparent',
      backgroundColor: 'transparent',
      fill: false,
      pointRadius: 0,
      tension: 0,
      order: 5,
    });
  }

  const chartOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 600,
    },
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: tokens.silver300,
          boxWidth: 12,
          boxHeight: 12,
          usePointStyle: true,
          font: {
            family: "'Bricolage Grotesque', ui-sans-serif, system-ui, sans-serif",
            size: 11,
          },
        },
      },
      tooltip: {
        backgroundColor: tokens.ink950,
        titleColor: tokens.gold400,
        bodyColor: tokens.silver100,
        borderColor: tokens.ink700,
        borderWidth: 1,
        padding: 10,
        boxPadding: 4,
        usePointStyle: true,
        callbacks: {
          label: (context: any) => {
            const label = context.dataset.label || '';
            const val = context.parsed.y;
            if (val === null || val === undefined) return '';
            return ` ${label.split('(')[0].trim()}: ${val.toFixed(2)} ${variable.unit}`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: gridLine,
          drawBorder: false,
        },
        ticks: {
          color: mistAxis,
          maxTicksLimit: 12,
          font: {
            family: "ui-sans-serif, system-ui, sans-serif",
            size: 10,
          },
        },
      },
      y: {
        grid: {
          color: gridLine,
          drawBorder: false,
        },
        ticks: {
          color: mistAxis,
          font: {
            family: "ui-sans-serif, system-ui, sans-serif",
            size: 10,
          },
          callback: (value: any) => `${value} ${variable.unit}`,
        },
      },
    },
  };

  // Download chart as PNG image
  const handleDownloadImage = () => {
    if (chartRef.current) {
      const url = chartRef.current.toBase64Image();
      const link = document.createElement('a');
      link.download = `HERAOCULUS_${variable.id}_${regionName.replace(/\s+/g, '_')}.png`;
      link.href = url;
      link.click();
    }
  };

  return (
    <div className="space-y-4 rounded-2xl border border-ink-700 bg-ink-900 p-4 sm:p-6">
      {/* Top Header & Controls Bar */}
      <div className="flex flex-col justify-between gap-4 border-b border-ink-700 pb-3 lg:flex-row lg:items-center">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-display flex flex-wrap items-center gap-x-2 text-base font-semibold text-silver-100 sm:text-lg">
              <span>{variable.name}</span>
              <span className="font-normal text-silver-500">in</span>
              <span className="text-gold-400">{regionName}</span>
            </h3>
            {isLoading && (
              <span className="inline-flex animate-pulse items-center gap-1.5 rounded-lg border border-gold-400/40 bg-ink-800 px-2.5 py-0.5 text-[11px] font-medium text-gold-300">
                <Loader2 className="h-3 w-3 animate-spin" />
                <span className="tabular-nums">
                  {loadingMessage || 'Fetching live feed...'}
                  {loadingElapsedSeconds !== undefined && loadingElapsedSeconds > 0 ? ` (${loadingElapsedSeconds}s)` : ''}
                </span>
              </span>
            )}
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-silver-500">
            <span>
              Missions: {variable.missions.join(' • ')} • Window: {visibleStart} to {visibleEnd} ({visibleCount} pts)
            </span>
            {stats.provenance && (
              <span
                className={`rounded-lg border border-ink-700 bg-ink-950 px-2 py-0.5 text-[10px] font-medium ${PROVENANCE_META[stats.provenance].colorClass}`}
                title={stats.statusMessage || PROVENANCE_META[stats.provenance].description}
              >
                {PROVENANCE_META[stats.provenance].icon}{' '}
                {stats.provenanceSource || PROVENANCE_META[stats.provenance].label}
              </span>
            )}
            {(stats.missingCount ?? 0) > 0 && (
              <span className="rounded-lg border border-gold-400/30 bg-ink-800 px-2 py-0.5 text-[10px] tabular-nums text-gold-300">
                {stats.missingCount} missing / polar-night slots preserved
              </span>
            )}
          </div>
        </div>

        {/* Action Buttons: Time Range & Image Download */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Time Range Filter */}
          <div className="flex items-center rounded-xl border border-ink-700 bg-ink-950 p-1 text-xs">
            <Calendar className="ml-1.5 mr-1 h-3.5 w-3.5 text-silver-500" />
            {rangeOptions.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setTimeRange(key)}
                className={`rounded-lg px-2.5 py-1 transition-colors tabular-nums ${
                  activeRange === key
                    ? 'bg-ink-700 font-semibold text-silver-100'
                    : 'text-silver-500 hover:text-silver-100'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <button
            onClick={handleDownloadImage}
            className="flex items-center gap-1 rounded-xl border border-ink-700 bg-ink-800 px-3 py-1.5 text-xs font-medium text-silver-300 transition-colors hover:bg-ink-700 hover:text-silver-100"
            title="Download high-resolution chart snapshot"
          >
            <Download className="h-3.5 w-3.5 text-gold-400" />
            <span className="hidden sm:inline">Export PNG</span>
          </button>
        </div>
      </div>

      {/* Interactive Layer Toggles */}
      <div className="flex flex-wrap items-center gap-2 pt-1 text-xs">
        <span className="mr-1 flex items-center gap-1 font-medium text-silver-500">
          <Layers className="h-3.5 w-3.5 text-silver-500" />
          <span>View layers</span>
        </span>

        {/* Deseasonalized Anomaly Toggle */}
        <button
          onClick={() => setShowAnomaly(!showAnomaly)}
          className={`flex items-center gap-1.5 rounded-lg border px-3 py-1 font-medium transition-colors ${
            showAnomaly
              ? 'border-silver-500/50 bg-ink-800 text-silver-500'
              : 'border-ink-700 bg-ink-950 text-silver-500 hover:text-silver-300'
          }`}
        >
          <Sparkles className="h-3 w-3 text-silver-500" />
          <span>Deseasonalized anomaly</span>
        </button>

        {!showAnomaly && (
          <>
            {/* Raw Observations Toggle */}
            <button
              onClick={() => setShowRaw(!showRaw)}
              className={`flex items-center gap-1.5 rounded-lg border px-3 py-1 font-medium transition-colors ${
                showRaw
                  ? 'border-ink-700 bg-ink-800 text-silver-300'
                  : 'border-ink-700 bg-ink-950 text-silver-500 line-through'
              }`}
            >
              {showRaw ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
              <span>Raw seasonal data</span>
            </button>

            {/* 12-Month Moving Avg Toggle */}
            <button
              onClick={() => setShowMovingAvg(!showMovingAvg)}
              className={`flex items-center gap-1.5 rounded-lg border px-3 py-1 font-medium transition-colors ${
                showMovingAvg
                  ? 'border-gold-400/50 bg-ink-800 text-gold-400'
                  : 'border-ink-700 bg-ink-950 text-silver-500 line-through'
              }`}
            >
              <Activity className="h-3 w-3" />
              <span>12-mo moving avg</span>
            </button>
          </>
        )}

        {/* Linear Trend Toggle */}
        <button
          onClick={() => setShowTrend(!showTrend)}
          className={`flex items-center gap-1.5 rounded-lg border px-3 py-1 font-medium transition-colors ${
            showTrend
              ? 'border-gold-400/50 bg-ink-800 text-gold-300'
              : 'border-ink-700 bg-ink-950 text-silver-500 line-through'
          }`}
        >
          <TrendingUp className="h-3 w-3" />
          <span>OLS trend line</span>
        </button>

        {/* 95% Confidence Interval Toggle */}
        {showTrend && (
          <button
            onClick={() => setShowCI(!showCI)}
            className={`flex items-center gap-1.5 rounded-lg border px-3 py-1 font-medium transition-colors ${
              showCI
                ? 'border-silver-600/50 bg-ink-800 text-silver-600'
                : 'border-ink-700 bg-ink-950 text-silver-500 line-through'
            }`}
          >
            <span>95% CI band</span>
          </button>
        )}
      </div>

      {/* Main Chart Canvas */}
      <div className="relative h-72 w-full pt-2 sm:h-80 md:h-96">
        {isLoading && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center rounded-xl border border-ink-700 bg-ink-950/90 p-6 text-center backdrop-blur-sm">
            <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl border border-gold-400/40 bg-ink-800">
              <Loader2 className="h-6 w-6 animate-spin text-gold-300" />
            </div>
            <div className="font-display flex items-center gap-2 text-sm font-semibold text-silver-100 sm:text-base">
              <span>Retrieving live observations for {variable.name}</span>
            </div>
            <p className="mt-1.5 max-w-lg text-xs text-gold-300">
              {loadingMessage || `Connecting to satellite instrument feeds for ${regionName}...`}
            </p>

            {loadingElapsedSeconds !== undefined && loadingElapsedSeconds >= 3 ? (
              <div className="mt-3 max-w-md rounded-xl border border-ink-700 bg-ink-900 p-3 text-xs leading-relaxed text-silver-300">
                <strong className="mb-0.5 block text-silver-100">High-resolution observation extraction in progress</strong>
                NASA/USGS remote servers are slicing multi-year satellite granules on the fly. This can take 10–25 seconds. Please wait while data is validated.
              </div>
            ) : (
              <span className="mt-2 text-[11px] text-silver-500">
                Verifying live NASA feeds before rendering chart and statistical trends...
              </span>
            )}

            {loadingElapsedSeconds !== undefined && loadingElapsedSeconds > 0 && (
              <div className="mt-2 text-[10px] text-silver-500">
                Elapsed:{' '}
                <span className="tabular-nums font-semibold text-gold-300">{loadingElapsedSeconds}s</span>
              </div>
            )}
          </div>
        )}
        <Line ref={chartRef} data={{ labels, datasets }} options={chartOptions} />
      </div>

      {/* Deseasonalization Scientific Note */}
      {showAnomaly && (
        <div className="flex items-start gap-2 rounded-xl border border-silver-500/30 bg-ink-800 p-3 text-xs text-silver-300">
          <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-silver-500" />
          <p>
            <strong className="text-silver-100">Scientific deseasonalization active:</strong> Each monthly value has been subtracted from its 30-year climatological monthly baseline. This eliminates the repeating summer/winter cycle, isolating true long-term climate warming or cooling.
          </p>
        </div>
      )}
    </div>
  );
};
