import React, { useCallback, useState } from 'react';
import { InterpolationStage } from '../../../types/earthScience';
import { API_BASE } from '../api';
import { Button } from '../../ui/Button';
import { Panel } from '../../ui/Panel';
import { Download, Loader2, X } from 'lucide-react';

interface ExportPanelProps {
  stage: InterpolationStage;
  result: any;
  crs: string | null;
  transform: any;
  nodata: number | null;
  setStage: React.Dispatch<React.SetStateAction<InterpolationStage>>;
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function ExportPanel({
  stage,
  result,
  crs,
  transform,
  nodata,
  setStage,
}: ExportPanelProps) {
  if (stage !== 'export' || !result) return null;

  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [includeReconstructed, setIncludeReconstructed] = useState(true);
  const [includeProvenanceStandalone, setIncludeProvenanceStandalone] =
    useState(false);
  const [includeQualityJson, setIncludeQualityJson] = useState(false);

  const hasProvenance = !!result?.provenance_mask;
  const hasQuality = !!result?.quality_mask;
  const nodataValue = nodata ?? -9999;
  const baseName = `heraoculus_export_${result.method ?? 'unknown'}_${Date.now()}`;

  const handleDownload = useCallback(async () => {
    setIsExporting(true);
    setExportError(null);
    try {
      const selected: Array<{ key: string; label: string }> = [];
      if (includeReconstructed) {
        selected.push({
          key: 'full',
          label: 'Reconstructed GeoTIFF (with provenance & quality bands)',
        });
      }
      if (includeProvenanceStandalone && hasProvenance) {
        selected.push({ key: 'provenance', label: 'Provenance mask GeoTIFF' });
      }
      if (includeQualityJson && hasQuality) {
        selected.push({ key: 'quality', label: 'Quality report JSON' });
      }
      if (selected.length === 0) {
        setExportError('Select at least one export option.');
        return;
      }

      const nodataNum = Number.isFinite(nodataValue) ? nodataValue : -9999;

      for (let i = 0; i < selected.length; i++) {
        const sel = selected[i];
        if (sel.key === 'full') {
          const resp = await fetch(`${API_BASE}/interpolate/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              output_filename: `${baseName}.tif`,
              nodata: nodataNum,
              include_provenance_band: true,
              include_quality_band: hasQuality,
              reconstructed_array: result.reconstructed,
              provenance_mask: result.provenance_mask,
              quality_mask: result.quality_mask ?? null,
              crs,
              transform,
              method: result.method ?? 'unknown',
            }),
          });
          if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || `Export failed (${resp.status})`);
          }
          const blob = await resp.blob();
          downloadBlob(blob, `${baseName}.tif`);
        } else if (sel.key === 'provenance') {
          const resp = await fetch(`${API_BASE}/interpolate/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              output_filename: `${baseName}_provenance.tif`,
              nodata: nodataNum,
              export_type: 'provenance',
              provenance_mask: result.provenance_mask,
              crs,
              transform,
              method: result.method ?? 'unknown',
            }),
          });
          if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || `Export failed (${resp.status})`);
          }
          const blob = await resp.blob();
          downloadBlob(blob, `${baseName}_provenance.tif`);
        } else if (sel.key === 'quality') {
          // Quality report derived from server-computed interpolation result.
          const report = {
            method: result.method ?? 'unknown',
            raster_type: result.raster_type ?? 'continuous',
            nodata: nodataNum,
            provenance_counts: result.provenance_mask
              ? (() => {
                  const counts: Record<string, number> = {
                    observed: 0, interpolated: 0, nodata: 0, excluded: 0,
                  };
                  for (const row of result.provenance_mask) {
                    for (const v of row) {
                      if (v in counts) counts[v] = (counts[v] ?? 0) + 1;
                    }
                  }
                  return counts;
                })()
              : null,
            quality_available: hasQuality,
            interpolated_pixel_count: result.interpolated_pixel_count ?? null,
            remaining_nodata_count: result.remaining_nodata_count ?? null,
            processing_duration_ms: result.processing_duration_ms ?? null,
            generated_at: new Date().toISOString(),
          };
          const blob = new Blob([JSON.stringify(report, null, 2)], {
            type: 'application/json',
          });
          downloadBlob(blob, `${baseName}_quality_report.json`);
        }

        // Stagger downloads so the browser does not coalesce/block them.
        if (i < selected.length - 1) {
          await new Promise((r) => setTimeout(r, 250));
        }
      }
    } catch (e: any) {
      setExportError(e.message || 'Export failed');
    } finally {
      setIsExporting(false);
    }
  }, [
    includeReconstructed,
    includeProvenanceStandalone,
    includeQualityJson,
    hasProvenance,
    hasQuality,
    nodataValue,
    baseName,
    result,
    crs,
    transform,
  ]);

  const disabledOpts: Record<string, string> = {};
  if (!hasProvenance) disabledOpts.provenance = 'No provenance mask available.';
  if (!hasQuality) disabledOpts.quality = 'No quality data available for this result.';

  return (
    <Panel title="Export Results">
      <div className="space-y-4">
        <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
          <p className="text-sm text-silver-300 mb-3">Export options:</p>
          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer text-sm">
              <input
                type="checkbox"
                defaultChecked
                checked={includeReconstructed}
                onChange={(e) => setIncludeReconstructed(e.target.checked)}
                className="form-checkbox"
                disabled={isExporting}
              />
              Reconstructed GeoTIFF (with provenance &amp; quality bands)
            </label>
            <label className="flex items-center gap-2 cursor-pointer text-sm">
              <input
                type="checkbox"
                checked={includeProvenanceStandalone}
                onChange={(e) => setIncludeProvenanceStandalone(e.target.checked)}
                className="form-checkbox"
                disabled={isExporting || !hasProvenance}
              />
              Provenance mask GeoTIFF
              {!hasProvenance && (
                <span className="text-xs text-red-400 ml-2">
                  (unavailable: no provenance mask)
                </span>
              )}
            </label>
            <label className="flex items-center gap-2 cursor-pointer text-sm">
              <input
                type="checkbox"
                checked={includeQualityJson}
                onChange={(e) => setIncludeQualityJson(e.target.checked)}
                className="form-checkbox"
                disabled={isExporting || !hasQuality}
              />
              Quality report JSON
              {!hasQuality && (
                <span className="text-xs text-red-400 ml-2">
                  (unavailable: no quality data)
                </span>
              )}
            </label>
          </div>
          {Object.entries(disabledOpts).length > 0 && (
            <p className="mt-2 text-xs text-silver-400">
              Unavailable options are disabled and will not be exported.
            </p>
          )}
        </div>

        <div className="bg-ink-900 rounded-xl border border-ink-700 p-4">
          <p className="text-xs text-silver-400 mb-3">
            Provenance band classification (GeoTIFF band 2):
          </p>
          <div className="text-xs text-silver-300 space-y-1 ml-2">
            <div>1 = observed</div>
            <div>2 = interpolated</div>
            <div>3 = NoData</div>
            <div>4 = excluded</div>
          </div>
        </div>

        {exportError && (
          <div className="rounded-xl border border-red-900/40 bg-ink-900 p-3">
            <p className="text-xs text-red-400">{exportError}</p>
          </div>
        )}

        <div className="mt-6 flex gap-3 justify-end">
          <Button variant="secondary" onClick={() => setStage('compare_results')}>
            <X className="w-4 h-4" /> Back to Comparison
          </Button>
          <Button
            onClick={handleDownload}
            disabled={isExporting}
          >
            {isExporting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Exporting…
              </>
            ) : (
              <>
                <Download className="w-4 h-4" /> Download Selected
              </>
            )}
          </Button>
        </div>
      </div>
    </Panel>
  );
}
