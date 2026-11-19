import React, { useEffect, useState } from 'react';
import { Grid, Play, Loader2, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { phase2Api, SpatialFdrResponse } from '../../services/phase2Api';
import { InfoBanner, MetricTile, Phase2ErrorAlert, ProvenanceBanner } from './shared';

interface SpatialFdrPanelProps {
  externalCube?: {
    label: string;
    cube: (number | null)[][][];
  } | null;
  onClearExternalCube?: () => void;
}

const REGION_PRESETS = [
  { label: 'Arctic & Boreal Sub-Grid (Svalbard/Barents)', lat: 74.0, lon: 22.0 },
  { label: 'Amazon Basin Tropical Grid', lat: -3.5, lon: -61.0 },
  { label: 'Sahel & Horn Semi-Arid Transect', lat: 14.0, lon: 15.0 },
];

export const SpatialFdrPanel: React.FC<SpatialFdrPanelProps> = ({
  externalCube,
  onClearExternalCube,
}) => {
  const [presetIdx, setPresetIdx] = useState(0);
  const [method, setMethod] = useState<'by' | 'bh' | 'holm' | 'bonferroni'>('by');
  const [alpha, setAlpha] = useState(0.05);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [res, setRes] = useState<SpatialFdrResponse | null>(null);
  const [inspectedCell, setInspectedCell] = useState<{ r: number; c: number } | null>(null);
  const [cubeTimeStep, setCubeTimeStep] = useState(1.0);

  // Auto-detect time step from selected region preset or manual override

  const runAnalysis = async () => {
    setLoading(true);
    setError(null);
    setInspectedCell(null);
    try {
      const region = REGION_PRESETS[presetIdx];
      const out = await phase2Api.runSpatialFdr({
        region_name: externalCube ? externalCube.label : region.label,
        center_lat: region.lat,
        center_lon: region.lon,
        grid_rows: 20,
        grid_cols: 24,
        n_years: 20,
        alpha,
        correction_method: method,
        cube: externalCube ? externalCube.cube : undefined,
        time_step_years: externalCube ? cubeTimeStep : undefined,
      });
      setRes(out);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void runAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIdx, method, alpha, externalCube, cubeTimeStep]);

  const cellInfo =
    res && inspectedCell
      ? {
          lat: res.grid.latitudes[inspectedCell.r],
          lon: res.grid.longitudes[inspectedCell.c],
          slope: res.grid.slope_decade[inspectedCell.r]?.[inspectedCell.c],
          p: res.grid.p_values[inspectedCell.r]?.[inspectedCell.c],
          q: res.grid.q_values[inspectedCell.r]?.[inspectedCell.c],
          rawSig: res.grid.raw_significant[inspectedCell.r]?.[inspectedCell.c],
          fdrSig: res.grid.fdr_significant[inspectedCell.r]?.[inspectedCell.c],
        }
      : null;

  return (
    <div className="space-y-4">
      <div className="bg-ink-900/90 border border-ink-700 rounded-2xl p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 items-end">
        <div className="lg:col-span-2 text-[11px] text-silver-300 space-y-1">
          <span className="block text-silver-500 font-semibold">Spatial Grid Source</span>
          {externalCube ? (
            <div className="flex items-center justify-between gap-2 px-3 py-1.5 rounded-lg bg-ink-800/35 border border-gold-400/40 text-gold-300 text-xs">
              <span className="truncate font-semibold">NetCDF Cube: {externalCube.label}</span>
              {onClearExternalCube && (
                <button
                  onClick={onClearExternalCube}
                  className="text-[10px] underline shrink-0 cursor-pointer"
                >
                  Clear
                </button>
              )}
            </div>
          ) : (
            <select
              value={presetIdx}
              onChange={(e) => setPresetIdx(Number(e.target.value))}
              className="w-full px-2.5 py-1.5 rounded-lg bg-ink-950 border border-ink-700 text-xs text-silver-100"
            >
              {REGION_PRESETS.map((p, idx) => (
                <option key={p.label} value={idx}>
                  {p.label}
                </option>
              ))}
            </select>
          )}
        </div>

        <label className="text-[11px] text-silver-300 space-y-1">
          <span className="block text-silver-500 font-semibold">Multiple-Testing Control</span>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as 'by' | 'bh' | 'holm' | 'bonferroni')}
            className="w-full px-2.5 py-1.5 rounded-lg bg-ink-950 border border-ink-700 text-xs text-silver-100"
          >
            <option value="by">Benjamini-Yekutieli (Spatial Corr.)</option>
            <option value="bh">Benjamini-Hochberg (FDR)</option>
            <option value="holm">Holm-Bonferroni (FWER)</option>
            <option value="bonferroni">Bonferroni (Strict FWER)</option>
          </select>
        </label>

        {externalCube && (
          <label className="text-[11px] text-silver-300 space-y-1">
            <span className="block text-silver-500 font-semibold">Temporal Resolution</span>
            <select
              value={cubeTimeStep}
              onChange={(e) => setCubeTimeStep(Number(e.target.value))}
              className="w-full px-2.5 py-1.5 rounded-lg bg-ink-950 border border-ink-700 text-xs text-silver-100"
            >
              <option value={1.0}>Annual (1.0)</option>
              <option value={0.5}>Semi-Annual (0.5)</option>
              <option value={0.25}>Quarterly (0.25)</option>
              <option value={0.08333333}>Monthly (1/12 ≈ 0.0833)</option>
            </select>
          </label>
        )}

        <label className="text-[11px] text-silver-300 space-y-1">
          <span className="block text-silver-500 font-semibold">Target Alpha (&alpha;)</span>
          <select
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            className="w-full px-2.5 py-1.5 rounded-lg bg-ink-950 border border-ink-700 text-xs text-silver-100"
          >
            <option value={0.01}>0.01 (Strict)</option>
            <option value={0.05}>0.05 (Standard)</option>
            <option value={0.10}>0.10 (Exploratory)</option>
          </select>
        </label>

        <button
          onClick={() => void runAnalysis()}
          disabled={loading}
          className="flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl bg-gold-400 hover:bg-gold-400 disabled:opacity-50 text-xs font-bold text-silver-100 cursor-pointer"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          Run Gridded MK + FDR
        </button>
      </div>

      <InfoBanner>
        Testing hundreds of correlated satellite grid pixels simultaneously inflates Type-I false positives. Wilks
        (2016) Benjamini-Yekutieli FDR prunes spurious random detections across space while preserving true regional
        forced signals.
      </InfoBanner>

      <Phase2ErrorAlert error={error} onRetry={() => void runAnalysis()} />

      {res && (
        <>
          <ProvenanceBanner provenance={res.provenance} />

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <MetricTile
              label="Grid Dimensions"
              value={`${res.dimensions.rows} × ${res.dimensions.cols}`}
              sub={`${res.summary.total_valid_cells} valid cells · T=${res.dimensions.time_steps}`}
            />
            <MetricTile
              label="Raw MK Significant"
              value={`${res.summary.raw_significant_cells} (${res.summary.pct_area_raw_sig}%)`}
              sub="Uncorrected p ≤ α"
              accent="text-gold-300"
            />
            <MetricTile
              label="FDR Surviving"
              value={`${res.summary.fdr_significant_cells} (${res.summary.pct_area_fdr_sig}%)`}
              sub="Adjusted q ≤ α"
              accent="text-gold-400"
            />
            <MetricTile
              label="False Positives Pruned"
              value={res.summary.false_positives_pruned}
              sub="Spurious pixels removed"
              accent="text-gold-300"
            />
            <MetricTile
              label="Wilks Global Alpha"
              value={res.summary.wilks_effective_alpha.toExponential(2)}
              sub={`Min adj q = ${res.summary.min_adjusted_q.toExponential(2)}`}
            />
            <MetricTile
              label="Field Significance"
              value={res.summary.field_significant ? 'SIGNIFICANT' : 'NONE'}
              accent={res.summary.field_significant ? 'text-silver-300' : 'text-silver-500'}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-ink-900/90 border border-ink-700 rounded-2xl p-4 space-y-2">
              <div className="flex items-center justify-between text-xs font-bold text-silver-100">
                <span>Raw Gridded Mann-Kendall Slope (Uncorrected p &le; {res.alpha})</span>
                <span className="text-[10px] text-gold-300">
                  {res.summary.raw_significant_cells} cells flagged
                </span>
              </div>
              <img
                src={res.images.trend_png}
                alt="Uncorrected spatial trend field"
                className="w-full rounded-xl border border-ink-700 bg-ink-950"
              />
            </div>

            <div className="bg-ink-900/90 border border-ink-700 rounded-2xl p-4 space-y-2">
              <div className="flex items-center justify-between text-xs font-bold text-silver-100">
                <span>Spatial FDR Field Mask ({res.method.toUpperCase()} Corrected q &le; {res.alpha})</span>
                <span className="text-[10px] text-silver-300">
                  {res.summary.fdr_significant_cells} verified cells
                </span>
              </div>
              <img
                src={res.images.fdr_png}
                alt="FDR-filtered spatial field"
                className="w-full rounded-xl border border-ink-700 bg-ink-950"
              />
            </div>
          </div>

          {/* Interactive Pixel Inspector */}
          <div className="bg-ink-900/90 border border-ink-700 rounded-2xl p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-xs font-bold text-silver-100">
                <Grid className="w-4 h-4 text-gold-400" />
                Interactive Cell FDR Inspector (Click any cell to compare Raw MK vs Adjusted q)
              </div>
              {cellInfo && (
                <div className="text-[11px] bg-ink-950 border border-ink-700 px-3 py-1 rounded-lg flex flex-wrap items-center gap-3">
                  <span className="text-silver-300">
                    ({cellInfo.lat?.toFixed(2)}°, {cellInfo.lon?.toFixed(2)}°)
                  </span>
                  <span className="text-gold-400">Slope: {cellInfo.slope}/dec</span>
                  <span className="text-silver-500">raw p={cellInfo.p?.toFixed(4)}</span>
                  <span className="text-gold-400">adj q={cellInfo.q?.toFixed(4)}</span>
                  <span className={cellInfo.fdrSig ? 'text-silver-500 font-bold' : cellInfo.rawSig ? 'text-gold-400 font-bold' : 'text-silver-300'}>
                    {cellInfo.fdrSig ? (
                      <span className="inline-flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" /> FDR Verified
                      </span>
                    ) : cellInfo.rawSig ? (
                      <span className="inline-flex items-center gap-1">
                        <ShieldAlert className="w-3 h-3" /> Pruned False Positive
                      </span>
                    ) : (
                      'Non-significant'
                    )}
                  </span>
                </div>
              )}
            </div>

            <div className="overflow-x-auto">
              <div
                className="inline-grid gap-0.5 bg-ink-950 p-2 rounded-xl border border-ink-700"
                style={{
                  gridTemplateColumns: `repeat(${res.dimensions.cols}, minmax(12px, 16px))`,
                }}
              >
                {[...res.grid.slope_decade].reverse().map((row, revR) => {
                  const r = res.dimensions.rows - 1 - revR;
                  return row.map((val, c) => {
                    const isFdr = res.grid.fdr_significant[r]?.[c];
                    const isRaw = res.grid.raw_significant[r]?.[c];
                    const active = inspectedCell?.r === r && inspectedCell?.c === c;
                    let bg = 'bg-ink-800/70';
                    if (isFdr) {
                      bg = val >= 0 ? 'bg-gold-400' : 'bg-gold-400';
                    } else if (isRaw) {
                      bg = 'bg-gold-400/40 ring-1 ring-gold-400/50';
                    }
                    return (
                      <button
                        key={`${r}-${c}`}
                        onClick={() => setInspectedCell({ r, c })}
                        title={`Lat ${res.grid.latitudes[r]}°, Lon ${res.grid.longitudes[c]}° | Slope ${val}/dec`}
                        className={`h-3.5 rounded-[2px] cursor-pointer transition ${bg} ${
                          active ? 'ring-2 ring-white scale-110 z-10' : 'hover:opacity-80'
                        }`}
                      />
                    );
                  });
                })}
              </div>
            </div>
            <div className="flex flex-wrap gap-4 text-[10px] text-silver-500">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-gold-400" /> FDR Significant (+)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-gold-400" /> FDR Significant (&minus;)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-gold-400/40 ring-1 ring-gold-400/50" /> Raw Sig &rarr; Pruned by FDR
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-ink-800" /> Background / Noise
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
