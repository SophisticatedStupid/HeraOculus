import React from 'react';
import { EarthVariable, HotspotRegion, TimeSeriesPoint, TrendAnalysisResult } from '../types/earthScience';
import {
  X,
  Download,
  FileSpreadsheet,
  FileText,
  Printer,
  CheckCircle2,
  Copy,
} from 'lucide-react';

interface DataExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  region: HotspotRegion;
  variable: EarthVariable;
  points: TimeSeriesPoint[];
  stats: TrendAnalysisResult;
}

export const DataExportModal: React.FC<DataExportModalProps> = ({
  isOpen,
  onClose,
  region,
  variable,
  points,
  stats,
}) => {
  const [copiedCitation, setCopiedCitation] = React.useState<boolean>(false);

  if (!isOpen) return null;

  // Generate clean RFC-style CSV without # comment headers so it re-imports cleanly
  const generateCsv = () => {
    const lat = region.coordinates[0].toFixed(4);
    const lon = region.coordinates[1].toFixed(4);
    let csv = `Date,Year,Month,Latitude,Longitude,Raw_${variable.id},MovingAvg_12Mo,Deseasonalized_Anomaly,Trend_OLS,CI95_Lower,CI95_Upper\n`;

    points.forEach((p) => {
      csv += `${p.date},${p.year},${p.month},${lat},${lon},${p.raw ?? ''},${p.movingAvg ?? ''},${p.anomaly ?? ''},${p.trend ?? ''},${p.ciLower ?? ''},${p.ciUpper ?? ''}\n`;
    });

    return csv;
  };

  const handleDownloadCsv = () => {
    const csvData = generateCsv();
    const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `HeraOculus_${variable.id}_${region.name.replace(/\s+/g, '_')}_${stats.startDate}_${stats.endDate}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadJson = () => {
    const jsonPayload = {
      metadata: {
        tool: 'HeraOculus Earth Science Change Detective',
        event: 'HeraOculus',
        exportDate: new Date().toISOString(),
        region: {
          name: region.name,
          category: region.category,
          coordinates: region.coordinates,
          phenomenon: region.keyPhenomenon,
        },
        variable: {
          id: variable.id,
          name: variable.name,
          unit: variable.unit,
          missions: variable.missions,
          category: variable.category,
        },
        statistics: stats,
      },
      timeSeries: points,
    };

    const blob = new Blob([JSON.stringify(jsonPayload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `HeraOculus_${variable.id}_${region.name.replace(/\s+/g, '_')}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const citationText = `HeraOculus (2026). Decadal Trend and Significance Analysis of ${variable.name} in ${region.name} (${stats.startDate} to ${stats.endDate}). Based on ${variable.missions.join(', ')} observations.`;

  const handleCopyCitation = () => {
    navigator.clipboard.writeText(citationText);
    setCopiedCitation(true);
    setTimeout(() => setCopiedCitation(false), 2500);
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink-950/85 backdrop-blur-md">
      <div className="relative w-full max-w-2xl bg-ink-900 border border-ink-700 rounded-2xl p-6 text-silver-100 max-h-[90vh] overflow-y-auto">
        <button
          type="button"
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 text-silver-500 hover:text-silver-100 rounded-lg hover:bg-ink-800 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center space-x-3 mb-5 pb-3 border-b border-ink-700">
          <div className="p-2.5 rounded-xl bg-ink-950 text-gold-400 border border-ink-700">
            <Download className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-display text-lg font-semibold text-silver-100">
              Export planetary science dossier
            </h3>
            <p className="text-xs text-silver-500">
              Download clean time-series datasets, statistical tables, and citations
            </p>
          </div>
        </div>

        <div className="bg-ink-950 rounded-xl p-4 border border-ink-700 space-y-2.5 text-xs">
          <div className="flex justify-between border-b border-ink-700 pb-2">
            <span className="text-silver-500">Target region</span>
            <span className="font-semibold text-silver-100">{region.name}</span>
          </div>
          <div className="flex justify-between border-b border-ink-700 pb-2">
            <span className="text-silver-500">Observed variable</span>
            <span className="font-semibold text-gold-400">
              {variable.name} ({variable.unit})
            </span>
          </div>
          <div className="flex justify-between border-b border-ink-700 pb-2">
            <span className="text-silver-500">Rate of change</span>
            <span className="tabular-nums font-semibold text-silver-100">
              {stats.slopePerDecade > 0 ? '+' : ''}
              {stats.slopePerDecade.toFixed(2)} {variable.unit}/decade
            </span>
          </div>
          <div className="flex justify-between border-b border-ink-700 pb-2">
            <span className="text-silver-500">Statistical significance</span>
            <span className="font-semibold text-gold-400">
              {stats.significanceLevel} ({stats.significanceStars}) • p{' '}
              {stats.mkPValue < 0.001 ? '< 0.001' : `= ${stats.mkPValue.toFixed(4)}`}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-silver-500">Sample records</span>
            <span className="tabular-nums text-silver-300">
              {stats.sampleCount} monthly data points ({stats.startDate} → {stats.endDate})
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 my-5">
          <button
            type="button"
            onClick={handleDownloadCsv}
            className="flex flex-col items-center justify-center p-4 bg-ink-950 hover:bg-ink-800 border border-ink-700 rounded-xl text-center transition-colors group"
          >
            <FileSpreadsheet className="w-6 h-6 text-silver-500 group-hover:scale-105 transition-transform mb-2" />
            <span className="text-xs font-semibold text-silver-100 block">Download CSV</span>
            <span className="text-[10px] text-silver-500">Raw, anomaly & trend data</span>
          </button>

          <button
            type="button"
            onClick={handleDownloadJson}
            className="flex flex-col items-center justify-center p-4 bg-ink-950 hover:bg-ink-800 border border-ink-700 rounded-xl text-center transition-colors group"
          >
            <FileText className="w-6 h-6 text-gold-400 group-hover:scale-105 transition-transform mb-2" />
            <span className="text-xs font-semibold text-silver-100 block">Download JSON</span>
            <span className="text-[10px] text-silver-500">Structured statistical API payload</span>
          </button>

          <button
            type="button"
            onClick={handlePrint}
            className="flex flex-col items-center justify-center p-4 bg-ink-950 hover:bg-ink-800 border border-ink-700 rounded-xl text-center transition-colors group"
          >
            <Printer className="w-6 h-6 text-silver-300 group-hover:scale-105 transition-transform mb-2" />
            <span className="text-xs font-semibold text-silver-100 block">Print / PDF report</span>
            <span className="text-[10px] text-silver-500">Formatted dossier summary</span>
          </button>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-silver-500">Suggested citation</span>
            <button
              type="button"
              onClick={handleCopyCitation}
              className="flex items-center space-x-1 text-xs text-gold-400 hover:text-silver-300 transition-colors"
            >
              {copiedCitation ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-silver-500" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
              <span>{copiedCitation ? 'Copied to clipboard' : 'Copy citation'}</span>
            </button>
          </div>
          <div className="p-3 bg-ink-950 rounded-xl border border-ink-700 text-[11px] text-silver-300 leading-relaxed">
            {citationText}
          </div>
        </div>

        <div className="mt-6 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-5 py-2 bg-ink-800 hover:bg-ink-700 text-silver-100 font-semibold text-xs rounded-xl transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
