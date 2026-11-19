import React from 'react';
import { chartSeries, tokens } from '../../theme/tokens';

export interface BoxPlotGroup {
  label: string;
  values: number[];
  color?: string;
}

interface BoxPlotChartProps {
  groups: BoxPlotGroup[];
  title?: string;
}

interface BoxStats {
  label: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  outliers: number[];
  color: string;
}

export const BoxPlotChart: React.FC<BoxPlotChartProps> = ({
  groups,
  title = 'Box & Whisker Distribution',
}) => {
  const statsList: BoxStats[] = groups.map((g, idx) => {
    const clean = g.values.filter(v => !isNaN(v) && isFinite(v)).sort((a, b) => a - b);
    const n = clean.length;
    if (n === 0) {
      return {
        label: g.label,
        min: 0,
        q1: 0,
        median: 0,
        q3: 0,
        max: 0,
        outliers: [],
        color: g.color || tokens.gold400,
      };
    }

    const q1 = clean[Math.floor(n * 0.25)];
    const median = n % 2 === 0 ? (clean[n / 2 - 1] + clean[n / 2]) / 2 : clean[Math.floor(n / 2)];
    const q3 = clean[Math.floor(n * 0.75)];
    const iqr = q3 - q1;
    const lowerWhisker = Math.max(clean[0], q1 - 1.5 * iqr);
    const upperWhisker = Math.min(clean[n - 1], q3 + 1.5 * iqr);

    const outliers = clean.filter(v => v < lowerWhisker || v > upperWhisker);

    return {
      label: g.label,
      min: lowerWhisker,
      q1,
      median,
      q3,
      max: upperWhisker,
      outliers,
      color: g.color || chartSeries[idx % chartSeries.length],
    };
  });

  const allVals = statsList.flatMap(s => [s.min, s.q1, s.median, s.q3, s.max, ...s.outliers]);
  const globalMin = allVals.length > 0 ? Math.min(...allVals) : 0;
  const globalMax = allVals.length > 0 ? Math.max(...allVals) : 1;
  const range = globalMax - globalMin || 1;

  const toPct = (val: number) => ((val - globalMin) / range) * 100;

  return (
    <div className="w-full space-y-4 rounded-xl border border-ink-700 bg-ink-900/60 p-4">
      {title && (
        <h4 className="font-display text-xs font-semibold text-silver-300">
          {title}
        </h4>
      )}

      <div className="space-y-6 pt-2">
        {statsList.map((box, i) => {
          const leftQ1 = toPct(box.q1);
          const widthBox = Math.max(2, toPct(box.q3) - leftQ1);
          const medianPos = toPct(box.median);
          const minPos = toPct(box.min);
          const maxPos = toPct(box.max);

          return (
            <div key={i} className="space-y-1">
              <div className="flex justify-between text-xs text-silver-500">
                <span className="font-medium text-silver-100">{box.label}</span>
                <span className="tabular-nums text-[11px]">
                  Med: {box.median.toFixed(2)} | IQR: {(box.q3 - box.q1).toFixed(2)}
                </span>
              </div>

              {/* Box plot visual container */}
              <div className="relative flex h-9 items-center overflow-hidden rounded-lg border border-ink-700 bg-ink-950/60">
                {/* Horizontal whisker line */}
                <div
                  className="absolute h-0.5 bg-silver-600"
                  style={{ left: `${minPos}%`, width: `${Math.max(1, maxPos - minPos)}%` }}
                />

                {/* Min whisker tick */}
                <div
                  className="absolute h-4 w-0.5 bg-silver-500"
                  style={{ left: `${minPos}%` }}
                />

                {/* Max whisker tick */}
                <div
                  className="absolute h-4 w-0.5 bg-silver-500"
                  style={{ left: `${maxPos}%` }}
                />

                {/* Interquartile Box (Q1 to Q3) */}
                <div
                  className="absolute flex h-6 items-center justify-center rounded border transition-all"
                  style={{
                    left: `${leftQ1}%`,
                    width: `${widthBox}%`,
                    backgroundColor: `${box.color}33`,
                    borderColor: box.color,
                  }}
                >
                  {/* Median bar */}
                  <div
                    className="absolute h-full w-1 bg-silver-100"
                    style={{ left: `${Math.max(0, Math.min(100, ((medianPos - leftQ1) / widthBox) * 100))}%` }}
                  />
                </div>

                {/* Outliers as scatter points */}
                {box.outliers.map((out, oi) => (
                  <div
                    key={oi}
                    className="absolute h-2 w-2 rounded-full border border-gold-300 bg-gold-400"
                    style={{ left: `${toPct(out)}%`, top: '50%', transform: 'translate(-50%, -50%)' }}
                    title={`Outlier: ${out.toFixed(2)}`}
                  />
                ))}
              </div>

              {/* Numeric labels beneath */}
              <div className="flex justify-between px-1 text-[10px] tabular-nums text-silver-500">
                <span>Min: {box.min.toFixed(1)}</span>
                <span>Q1: {box.q1.toFixed(1)}</span>
                <span>Q3: {box.q3.toFixed(1)}</span>
                <span>Max: {box.max.toFixed(1)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
