import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { tokens } from '../../theme/tokens';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface HistogramChartProps {
  data: number[];
  title?: string;
  bins?: number;
  color?: string;
}

export const HistogramChart: React.FC<HistogramChartProps> = ({
  data,
  title = 'Frequency Distribution',
  bins = 15,
  color = tokens.gold400,
}) => {
  const chartData = useMemo(() => {
    const clean = data.filter(v => !isNaN(v) && isFinite(v));
    if (clean.length === 0) {
      return { labels: [], datasets: [] };
    }

    const min = Math.min(...clean);
    const max = Math.max(...clean);
    const numBins = Math.max(5, Math.min(30, bins));
    const binWidth = (max - min) / numBins || 1;

    const binCounts = new Array(numBins).fill(0);
    const labels: string[] = [];

    for (let i = 0; i < numBins; i++) {
      const bMin = min + i * binWidth;
      const bMax = min + (i + 1) * binWidth;
      labels.push(`${bMin.toFixed(1)} - ${bMax.toFixed(1)}`);
    }

    for (const v of clean) {
      let idx = Math.floor((v - min) / binWidth);
      if (idx >= numBins) idx = numBins - 1;
      if (idx < 0) idx = 0;
      binCounts[idx]++;
    }

    return {
      labels,
      datasets: [
        {
          label: 'Frequency',
          data: binCounts,
          backgroundColor: color + '99',
          borderColor: color,
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    };
  }, [data, bins, color]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: {
        display: !!title,
        text: title,
        color: tokens.silver100,
        font: { size: 13, weight: 'bold' as const, family: "'Bricolage Grotesque', ui-sans-serif, system-ui, sans-serif" },
      },
      tooltip: {
        backgroundColor: tokens.ink900,
        titleColor: tokens.gold400,
        bodyColor: tokens.silver100,
        borderColor: tokens.ink700,
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        grid: { color: `${tokens.ink700}40` },
        ticks: { color: tokens.silver500, font: { size: 10 }, maxRotation: 45 },
      },
      y: {
        grid: { color: `${tokens.ink700}40` },
        ticks: { color: tokens.silver500, font: { size: 10 } },
      },
    },
  };

  return (
    <div className="h-64 w-full">
      <Bar data={chartData} options={options} />
    </div>
  );
};
