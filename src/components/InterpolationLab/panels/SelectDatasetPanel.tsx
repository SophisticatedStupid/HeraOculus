import React from 'react';
import { InterpolationRasterType, InterpolationStage } from '../../../types/earthScience';
import { Button } from '../../ui/Button';
import { Loader2, BarChart3, RefreshCw, X } from 'lucide-react';
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
  handleInspect: () => Promise<void>;
  handleRegenerate: () => void;
  setStage: React.Dispatch<React.SetStateAction<InterpolationStage>>;
}

export function SelectDatasetPanel({
  stage,
  rasterData,
  rasterMeta,
  isLoading,
  handleInspect,
  handleRegenerate,
  setStage,
}: SelectDatasetPanelProps) {
  if (stage !== 'select_dataset') return null;

  return (
    <Panel title="Select Raster Dataset">
      <p className="text-sm text-silver-300 mb-4">
        No raster uploaded yet. Generating a synthetic continuous raster (64×64) with ~20% missing data
        for demonstration.
      </p>
      <div className="grid grid-cols-2 gap-4">
        <RasterCanvas
          data={rasterData ?? []}
          title="Original Raster (null = NoData)"
          colormap="viridis"
        />
        <div className="flex flex-col justify-center">
          <p className="text-xs text-silver-400 mb-2">Raster info:</p>
          <div className="text-xs text-silver-300 space-y-1">
            <div>
              Dimensions: {rasterMeta?.width ?? 0} × {rasterMeta?.height ?? 0}
            </div>
            <div>CRS: {rasterMeta?.crs ?? 'None'}</div>
            <div>NoData: {rasterMeta?.nodata ?? 'None'}</div>
          </div>
        </div>
      </div>
      <div className="mt-6 flex gap-3">
        <Button onClick={handleInspect} disabled={isLoading}>
          {isLoading ? <Loader2 className="w-4 h-4" /> : <BarChart3 className="w-4 h-4" />}
          Inspect Missing Data
        </Button>
        <Button variant="secondary" onClick={handleRegenerate}>
          <RefreshCw className="w-4 h-4" /> New Synthetic Raster
        </Button>
      </div>
    </Panel>
  );
}
