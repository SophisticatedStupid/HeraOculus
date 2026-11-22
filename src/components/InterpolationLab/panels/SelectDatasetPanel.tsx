import React from 'react';
import { InterpolationRasterType, InterpolationStage } from '../../../types/earthScience';
import { Button } from '../../ui/Button';
import { Loader2, BarChart3, RefreshCw, X, Upload, Clock, Trash2 } from 'lucide-react';
import { Panel } from '../../ui/Panel';
import { generateSyntheticRaster } from '../synthetic';
import RasterCanvas from '../RasterCanvas';

interface SelectDatasetPanelProps {
  stage: InterpolationStage;
  rasterData: ((number | null)[][]) | null;
  rasterMeta: {
    width: number;
    height: number;
    crs: string | null;
    transform: any;
    nodata: number | null;
  } | null;
  isLoading: boolean;
  uploadMode: 'demo' | 'upload';
  selectedFile: File | null;
  selectedFileError: string | null;
  handleInspect: () => Promise<void>;
  handleRegenerate: () => void;
  handleSelectFile: (file: File) => Promise<void>;
  handleRemoveFile: () => void;
  setUploadMode: (mode: 'demo' | 'upload') => void;
  setStage: React.Dispatch<React.SetStateAction<InterpolationStage>>;
  // temporal
  temporalObservations: any[];
  temporalCube: number[][][] | null;
  temporalTimes: number[] | null;
  uploadError: string | null;
  setUploadError: (msg: string | null) => void;
  handleAddTemporalObservation: (file: File, timestamp: string) => Promise<void>;
  handleRemoveTemporalObservation: (id: string) => void;
  handleClearTemporalObservations: () => void;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function SelectDatasetPanel({
  stage,
  rasterData,
  rasterMeta,
  isLoading,
  uploadMode,
  selectedFile,
  selectedFileError,
  handleInspect,
  handleRegenerate,
  handleSelectFile,
  handleRemoveFile,
  setUploadMode,
  setStage,
  temporalObservations,
  temporalCube,
  temporalTimes,
  uploadError,
  setUploadError,
  handleAddTemporalObservation,
  handleRemoveTemporalObservation,
  handleClearTemporalObservations,
}: SelectDatasetPanelProps) {
  if (stage !== 'select_dataset') return null;

  const isDemo = uploadMode === 'demo';

  return (
    <Panel title="Select Raster Dataset">
      <div className="mb-4 flex gap-2 items-center">
        <Button variant="secondary" onClick={() => setUploadMode('demo')} disabled={isDemo || isLoading}>
          <Clock className="w-4 h-4" /> Demo (synthetic)
        </Button>
        <Button onClick={() => setUploadMode('upload')} disabled={!isDemo || isLoading}>
          <Upload className="w-4 h-4" /> Upload raster
        </Button>
      </div>

      {isDemo ? (
        <div className="mb-4 text-sm text-silver-300">
          Generating a synthetic continuous raster (64×64) with ~20% missing data for demonstration.
        </div>
      ) : (
        <>
          <div className="mb-4">
            <label className="block text-xs text-silver-400 mb-1">
              GeoTIFF file (.tif, .tiff, .geotiff, max 100 MB)
            </label>
            <input
              type="file"
              accept=".tif,.tiff,.geotiff"
              disabled={isLoading}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleSelectFile(file);
                e.target.value = '';
              }}
            />
            {selectedFileError && (
              <p className="mt-1 text-xs text-red-400">{selectedFileError}</p>
            )}
          </div>
          {selectedFile && (
            <div className="mb-4 bg-ink-900 rounded-xl border border-ink-700 p-3">
              <div className="flex items-center gap-2 mb-1">
                <Upload className="w-4 h-4 text-silver-400" />
                <span className="text-sm text-silver-300 truncate" title={selectedFile.name}>
                  {selectedFile.name}
                </span>
                <Button variant="secondary" onClick={handleRemoveFile}>
                  <Trash2 className="w-3 h-3" />
                </Button>
              </div>
              <div className="text-xs text-silver-500">
                {formatFileSize(selectedFile.size)}
              </div>
            </div>
          )}
          <div className="mt-6 flex gap-3">
            <Button onClick={handleInspect} disabled={isLoading || !selectedFile}>
              {isLoading ? <Loader2 className="w-4 h-4" /> : <BarChart3 className="w-4 h-4" />}
              Inspect Missing Data
            </Button>
            <Button variant="secondary" onClick={handleRegenerate}>
              <RefreshCw className="w-4 h-4" /> New Synthetic Raster
            </Button>
          </div>
        </>
      )}

      {/* Temporal raster support (multiple observations with timestamps) */}
      <div className="mt-8 border-t border-ink-700 pt-6">
        <h3 className="text-xs font-semibold text-silver-400 mb-3 uppercase tracking-wider">
          Temporal raster stack
        </h3>
        <p className="text-xs text-silver-500 mb-3">
          Upload multiple rasters, each with a timestamp. Dimensions and CRS must match across
          observations; timestamps are preserved and sent to the backend as the temporal cube
          (T, H, W) + times.
        </p>
        <div className="space-y-3">
          {temporalObservations.map((obs) => (
            <div key={obs.id} className="bg-ink-900 rounded-xl border border-ink-700 p-3 flex flex-col md:flex-row md:items-center gap-3">
              <div className="flex-1 min-w-0">
                <div className="text-xs text-silver-400 truncate" title={obs.file.name}>
                  {obs.file.name}
                </div>
                <div className="text-xs text-silver-300 font-mono truncate">
                  {obs.timestamp} &rarr; t = {obs.time.toFixed(4)}
                </div>
                <div className="text-[10px] text-silver-500">
                  {obs.width}×{obs.height}{obs.crs ? ` · CRS: ${obs.crs}` : ''}
                </div>
              </div>
              <Button variant="secondary" onClick={() => handleRemoveTemporalObservation(obs.id)}>
                <Trash2 className="w-3 h-3" />
              </Button>
            </div>
          ))}
          {temporalObservations.length === 0 && (
            <p className="text-xs text-silver-500 italic">No temporal observations yet.</p>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
            <div>
              <label className="block text-xs text-silver-400 mb-1">Timestamp (year or ISO date)</label>
              <input
                id="temporal-timestamp"
                type="text"
                placeholder="e.g. 2020.0"
                className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-100 focus:border-gold-400"
              />
            </div>
            <div>
              <label className="block text-xs text-silver-400 mb-1">GeoTIFF file</label>
              <input
                id="temporal-file"
                type="file"
                accept=".tif,.tiff,.geotiff"
                className="w-full bg-ink-900 border border-ink-700 rounded px-2 py-1.5 text-xs text-silver-400 focus:border-gold-400"
              />
            </div>
          </div>
          <div className="mt-2 flex gap-2">
            <Button
              onClick={() => {
                const fileInput = document.getElementById(
                  'temporal-file'
                ) as HTMLInputElement | null;
                const tsInput = document.getElementById(
                  'temporal-timestamp'
                ) as HTMLInputElement | null;
                const file = fileInput?.files?.[0];
                const timestamp = tsInput?.value?.trim();
                if (!file) {
                  setUploadError('Select a GeoTIFF file');
                  return;
                }
                if (!timestamp) {
                  setUploadError('Enter a timestamp');
                  return;
                }
                if (!tsInput) return;
                void handleAddTemporalObservation(file, timestamp);
                fileInput.value = '';
                tsInput.value = '';
              }}
              disabled={isLoading}
            >
              <Upload className="w-4 h-4" /> Add observation
            </Button>
            <Button variant="secondary" onClick={handleClearTemporalObservations} disabled={temporalObservations.length === 0}>
              <X className="w-4 h-4" /> Clear stack
            </Button>
          </div>
          {uploadError && <p className="mt-2 text-xs text-red-400">{uploadError}</p>}
          {temporalCube && temporalTimes && (
            <div className="mt-3 bg-ink-900/50 rounded border border-ink-700/50 p-3">
              <div className="text-xs text-silver-400 mb-1">Assembled temporal cube</div>
              <div className="text-xs text-silver-300 space-y-1">
                <div>Steps (T): <span className="font-mono">{temporalCube.length}</span></div>
                <div>Dimensions: <span className="font-mono">{temporalCube[0]?.length ?? 0}×{temporalCube[0]?.[0]?.length ?? 0}</span></div>
                <div>Times: <span className="font-mono">{temporalTimes.map((t) => t.toFixed(4)).join(', ')}</span></div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Panel>
  );
}
