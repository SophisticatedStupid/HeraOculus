import React, { useRef, useEffect } from 'react';

interface HeatmapChartProps {
  matrix: number[][]; // correlation matrix values [-1, 1]
  labels: string[];
  title?: string;
}

export const HeatmapChart: React.FC<HeatmapChartProps> = ({
  matrix,
  labels,
  title = 'Correlation Matrix Heatmap',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const n = labels.length;
    if (n === 0 || matrix.length === 0) return;

    const size = Math.min(canvas.width, canvas.height);
    const cellSize = (size - 60) / n;
    const offsetX = 50;
    const offsetY = 20;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Color mapper from -1 (silver-600) through 0 (ink-900) to +1 (gold-400)
    const getColor = (val: number) => {
      const clamped = Math.max(-1, Math.min(1, val));
      if (clamped >= 0) {
        // 0 to 1: ink-900 to gold-400
        const t = clamped;
        const r = Math.round(10 + t * (212 - 10));
        const g = Math.round(10 + t * (175 - 10));
        const b = Math.round(10 + t * (55 - 10));
        return `rgb(${r}, ${g}, ${b})`;
      } else {
        // -1 to 0: silver-600 to ink-900
        const mag = Math.abs(clamped);
        const t = mag;
        const r = Math.round(10 + t * (94 - 10));
        const g = Math.round(10 + t * (94 - 10));
        const b = Math.round(10 + t * (94 - 10));
        return `rgb(${r}, ${g}, ${b})`;
      }
    };

    // Draw cells
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const val = matrix[r]?.[c] ?? 0;
        const x = offsetX + c * cellSize;
        const y = offsetY + r * cellSize;

        ctx.fillStyle = getColor(val);
        ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

        // Draw correlation text
        ctx.fillStyle = '#F2F2F2';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(val.toFixed(2), x + cellSize / 2, y + cellSize / 2);
      }

      // Draw Row label
      ctx.fillStyle = '#8A8A8A';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      const label = labels[r].length > 6 ? labels[r].substring(0, 6) + '..' : labels[r];
      ctx.fillText(label, offsetX - 8, offsetY + r * cellSize + cellSize / 2);

      // Draw Column label
      ctx.save();
      ctx.translate(offsetX + r * cellSize + cellSize / 2, offsetY + n * cellSize + 10);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = 'right';
      ctx.fillText(label, 0, 0);
      ctx.restore();
    }
  }, [matrix, labels]);

  return (
    <div className="w-full bg-ink-900/60 rounded-xl border border-ink-700 p-4 space-y-3">
      {title && (
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-semibold text-silver-300">
            {title}
          </h4>
          <div className="flex items-center space-x-2 text-[10px] text-silver-500">
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-sm bg-gold-400 inline-block"></span>
              <span>-1 (Neg)</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-sm bg-ink-800 inline-block"></span>
              <span>0 (Neutral)</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-sm bg-gold-400 inline-block"></span>
              <span>+1 (Pos)</span>
            </span>
          </div>
        </div>
      )}

      <div className="flex justify-center overflow-x-auto">
        <canvas
          ref={canvasRef}
          width={380}
          height={380}
          className="max-w-full"
        />
      </div>
    </div>
  );
};
