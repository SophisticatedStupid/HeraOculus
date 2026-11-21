/**
 * HeraOculus → Google Earth Engine script modal.
 *
 * Pure client-side code generation: produces a ready-to-run GEE Code Editor
 * (JavaScript) or earthengine-api (Python) script from the user's current
 * Explorer selection (region/point, date range, variable). No GEE credentials,
 * no backend, no network calls.
 */

import React from 'react';
import {
  X,
  Download,
  Copy,
  Code2,
  AlertCircle,
  CheckCircle2,
} from 'lucide-react';
import {
  EarthVariable,
  HotspotRegion,
  VariableId,
} from '../types/earthScience';
import {
  generateJsScript,
  generatePythonScript,
  GeeScriptError,
  sanitizeExportName,
} from '../services/gee/scriptGenerator';
import {
  listGeeDatasets,
  resolveGeeDataset,
  GeeScriptOptions,
  GeeExportTarget,
  GeeTemporalAgg,
  GeeReducer,
} from '../services/gee/datasetMap';

type Lang = 'js' | 'py';

interface GeeScriptModalProps {
  isOpen: boolean;
  onClose: () => void;
  region: HotspotRegion;
  variable: EarthVariable;
  startDate: string;
  endDate: string;
}

const DEFAULT_LANG: Lang = 'js';
const DEFAULT_TEMPORAL: GeeTemporalAgg = 'none';
const DEFAULT_SPATIAL: GeeReducer = 'mean';
const DEFAULT_EXPORT: GeeExportTarget = 'chart';

export const GeeScriptModal: React.FC<GeeScriptModalProps> = ({
  isOpen,
  onClose,
  region,
  variable,
  startDate: propStartDate,
  endDate: propEndDate,
}) => {
  const [lang, setLang] = React.useState<Lang>(DEFAULT_LANG);
  const [datasetIndex, setDatasetIndex] = React.useState<number>(0);
  const [temporalAggregation, setTemporalAggregation] = React.useState<GeeTemporalAgg>(
    DEFAULT_TEMPORAL
  );
  const [spatialReducer, setSpatialReducer] = React.useState<GeeReducer>(
    DEFAULT_SPATIAL
  );
  const [exportTarget, setExportTarget] = React.useState<GeeExportTarget>(
    DEFAULT_EXPORT
  );
  const [script, setScript] = React.useState<string>('');
  const [error, setError] = React.useState<string | null>(null);
  const [copied, setCopied] = React.useState<boolean>(false);

  const datasets = React.useMemo(() => listGeeDatasets(variable.id as VariableId), [
    variable.id,
  ]);
  const hasSupport = datasets.length > 0;

  const geometry = {
    kind: 'point' as const,
    lat: region.coordinates[0],
    lon: region.coordinates[1],
  };

  const languageButtons = (['js', 'py'] as const).map((l) => (
    <button
      key={l}
      type="button"
      onClick={() => setLang(l)}
      className={`px-3 py-1.5 text-xs rounded ${lang === l ? 'bg-gold-400 text-ink-950' : 'bg-ink-800 border border-ink-700 text-silver-400 hover:text-silver-100'}`}
    >
      {l === 'js' ? 'JS (Editor)' : 'Python'}
    </button>
  ));
  const reducerButtons = (['mean', 'median', 'max'] as const).map((r) => (
    <button
      key={r}
      type="button"
      onClick={() => setSpatialReducer(r)}
      className={`px-2.5 py-1 text-xs rounded border ${spatialReducer === r ? 'bg-gold-400/20 border-gold-400 text-gold-400' : 'bg-ink-800 border-ink-700 text-silver-400 hover:border-ink-500'}`}
    >
      {r}
    </button>
  ));

  const [startDate, setStartDate] = React.useState<string>(propStartDate);
  const [endDate, setEndDate] = React.useState<string>(propEndDate);

  React.useEffect(() => {
    setStartDate(propStartDate);
    setEndDate(propEndDate);
  }, [propStartDate, propEndDate]);

  const generate = () => {
    setError(null);
    try {
      const config = resolveGeeDataset(variable.id as VariableId, datasetIndex);
      if (!config) {
        setError(
          `No GEE equivalent configured for variable '${variable.id}'.`
        );
        setScript('');
        return;
      }
      const options: GeeScriptOptions = {
        geometry,
        startDate,
        endDate,
        variableId: variable.id as VariableId,
        datasetIndex,
        temporalAggregation,
        spatialReducer,
        exportTarget,
      };
      const out = lang === 'js' ? generateJsScript(options) : generatePythonScript(options);
      setScript(out);
    } catch (err) {
      if (err instanceof GeeScriptError) {
        setError(err.message);
      } else {
        setError(String(err));
      }
      setScript('');
    }
  };

  React.useEffect(() => {
    generate();
  }, [variable.id, datasetIndex, temporalAggregation, spatialReducer, exportTarget, lang, geometry, startDate, endDate]);

  const handleCopy = () => {
    if (!script) return;
    navigator.clipboard.writeText(script);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!script) return;
    const ext = lang === 'js' ? 'js' : 'py';
    const mime = lang === 'js' ? 'text/javascript' : 'text/x-python';
    const blob = new Blob([script], { type: `${mime};charset=utf-8` });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const safeVar = sanitizeExportName(variable.id);
    link.href = url;
    link.download = `HeraOculus_${safeVar}_${startDate}_${endDate}.${ext}`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (!isOpen) return null;

  const noSupportMsg =
    'No Google Earth Engine equivalent is configured for this variable, so a runnable GEE script cannot be generated. ' +
    'This keeps exported scripts honest — we never fabricate a dataset or band that does not exist in the public GEE catalog.';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink-950/90 backdrop-blur-md">
      <div className="relative w-full max-w-[980px] bg-ink-900 border border-ink-700 rounded-2xl p-6 text-silver-100 max-h-[92vh] overflow-y-auto">
        <button
          type="button"
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 text-silver-500 hover:text-silver-100 rounded-lg hover:bg-ink-800 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center space-x-3 mb-4 pb-3 border-b border-ink-700">
          <div className="p-2.5 rounded-xl bg-ink-950 text-gold-400 border border-ink-700">
            <Code2 className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-display text-lg font-semibold text-silver-100">
              Generate GEE script
            </h3>
            <p className="text-xs text-silver-500">
              Ready-to-run Earth Engine Code Editor (JS) or earthengine-api (Python)
            </p>
          </div>
        </div>

        {/* Selection summary */}
        <div className="bg-ink-950 rounded-xl border border-ink-700 p-4 text-xs space-y-2 mb-5">
          <div className="flex justify-between border-b border-ink-700 pb-1">
            <span className="text-silver-500">Region</span>
            <span className="font-semibold text-silver-100 truncate">
              {region.name} ({region.coordinates[0].toFixed(3)}°, {region.coordinates[1].toFixed(3)}°)
            </span>
          </div>
          <div className="flex justify-between border-b border-ink-700 pb-1">
            <span className="text-silver-500">Variable</span>
            <span className="font-semibold text-silver-100 truncate">
              {variable.name}
            </span>
          </div>
          <div className="flex justify-between border-b border-ink-700 pb-1">
            <span className="text-silver-500">Date range</span>
            <span className="font-semibold text-silver-100">
              {startDate} → {endDate}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-silver-500">Language</span>
            <span className="font-semibold text-gold-400">{lang === 'js' ? 'JavaScript (Code Editor)' : 'Python (earthengine-api)'}</span>
          </div>
        </div>

        {!hasSupport ? (
          // No GEE equivalent
          <div className="rounded-lg border border-gold-900/40 bg-ink-900 p-5 mb-5">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-gold-400 mt-0.5" />
              <div>
                <h4 className="font-semibold text-gold-300">No GEE equivalent for this variable</h4>
                <p className="text-xs text-silver-400 leading-relaxed">
                  {noSupportMsg}
                </p>
              </div>
            </div>
          </div>
        ) : null}

        {hasSupport && (
          <>
            {/* Controls */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
              <div className="col-span-2">
                <label className="text-xs text-silver-500 block mb-1">Dataset</label>
                <select
                  value={datasetIndex}
                  onChange={(e) => setDatasetIndex(Number(e.target.value))}
                  className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
                >
                  {datasets.map((d, i) => (
                    <option key={i} value={i}>
                      {d.label}{d.deprecated ? ' (deprecated)' : ''}
                    </option>
                  ))}
                </select>
                <p className="text-[10px] text-silver-500 mt-1">
                  {resolveGeeDataset(variable.id as VariableId, datasetIndex)?.notes ?? ''}
                </p>
              </div>
              <div>
                <label className="text-xs text-silver-500 block mb-1">Language</label>
                <div className="flex gap-1">
                  {languageButtons}
                </div>
              </div>
              <div>
                <label className="text-xs text-silver-500 block mb-1">Temporal</label>
                <select
                  value={temporalAggregation}
                  onChange={(e) => setTemporalAggregation(e.target.value as GeeTemporalAgg)}
                  className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
                >
                  <option value="none">None (native)</option>
                  <option value="monthly">Monthly mean</option>
                  <option value="annual">Annual mean</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-silver-500 block mb-1">Export</label>
                <select
                  value={exportTarget}
                  onChange={(e) => setExportTarget(e.target.value as GeeExportTarget)}
                  className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
                >
                  <option value="chart">Time-series chart</option>
                  <option value="csv-drive">CSV to Drive</option>
                  <option value="geotiff-drive">GeoTIFF to Drive</option>
                </select>
              </div>
            </div>

            <div className="flex gap-2 mb-3">
              <label className="text-xs text-silver-500">Spatial reducer</label>
              {reducerButtons}
              <span className="ml-2 text-[10px] text-silver-500">
                (point geometry = point sample)
              </span>
            </div>

            {/* Script output */}
            <div className="rounded-xl border border-ink-700 bg-ink-950 overflow-hidden">
              {error && (
                <div className="p-3 rounded-lg border border-red-900/40 bg-ink-900 mb-3">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 text-red-400 mt-0.5" />
                    <p className="text-xs text-red-200">{error}</p>
                  </div>
                </div>
              )}
              {!error && script ? (
                <div className="relative">
                  <pre className="p-4 text-[11px] font-mono text-silver-300 overflow-x-auto leading-relaxed">
                    {script}
                  </pre>
                  <div className="absolute top-3 right-3 flex gap-1.5">
                    <button
                      type="button"
                      onClick={handleCopy}
                      className="p-1.5 text-xs rounded bg-ink-800 hover:bg-ink-700 text-silver-300 border border-ink-700 hover:text-silver-100 transition-colors"
                      title="Copy to clipboard"
                    >
                      {copied ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-silver-500" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={handleDownload}
                      className="p-1.5 text-xs rounded bg-ink-800 hover:bg-ink-700 text-silver-300 border border-ink-700 hover:text-silver-100 transition-colors"
                      title="Download"
                    >
                      <Download className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="p-4 text-xs text-silver-500 text-center">
                  Generating script…
                </div>
              )}
            </div>

            <p className="text-[10px] text-silver-500 mt-3 text-center">
              Paste into your own <a
                href="https://code.earthengine.google.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gold-400 hover:underline"
              >GEE Code Editor</a>{lang === 'js' ? '' : ' or run with earthengine-api'} — Geo-Iris does not execute this code.
            </p>
          </>
        )}

        <div className="mt-5 flex justify-end">
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

export default GeeScriptModal;
