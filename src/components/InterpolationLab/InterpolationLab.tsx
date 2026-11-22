import React from 'react';
import { Layout, AlertCircle, X } from 'lucide-react';
import { useInterpolationLab } from './useInterpolationLab';
import { SelectDatasetPanel } from './panels/SelectDatasetPanel';
import { InspectMissingPanel } from './panels/InspectMissingPanel';
import { ChooseMethodPanel } from './panels/ChooseMethodPanel';
import { ConfigurePanel } from './panels/ConfigurePanel';
import { ValidationPanel } from './panels/ValidationPanel';
import { ExecutePanel } from './panels/ExecutePanel';
import { CompareResultsPanel } from './panels/CompareResultsPanel';
import { ExportPanel } from './panels/ExportPanel';

interface InterpolationLabProps {
  onClose?: () => void;
}

export const InterpolationLab: React.FC<InterpolationLabProps> = ({ onClose }) => {
  const {
    stage,
    setStage,
    rasterData,
    rasterMeta,
    inspection,
    method,
    rasterType,
    setRasterType,
    compatibleMethods,
    config,
    setConfig,
    validation,
    result,
    isLoading,
    error,
    showProvenance,
    setShowProvenance,
    showQuality,
    setShowQuality,
    handleInspect,
    handleChooseMethod,
    handleConfigure,
    handleValidate,
    handleExecute,
    handleRegenerate,
    // dataset source
    uploadMode,
    setUploadMode,
    selectedFile,
    selectedFileError,
    handleSelectFile,
    handleRemoveFile,
    // temporal
    temporalObservations,
    temporalCube,
    temporalTimes,
    uploadError,
    setUploadError,
    handleAddTemporalObservation,
    handleRemoveTemporalObservation,
    handleClearTemporalObservations,
    currentMethodInfo,
    currentIndex,
    stageOrder,
    stageLabels,
  } = useInterpolationLab({ onClose });

  return (
    <div className="min-h-screen bg-ink-950 text-silver-100">
      <header className="border-b border-ink-700 bg-ink-950/95 backdrop-blur-md px-4 lg:px-8 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          <button
            onClick={() => onClose?.()}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-lg border border-ink-700 hover:bg-ink-800 text-xs font-medium text-silver-300 transition-colors"
          >
            <X className="w-4 h-4" />
            <span>Back to Explorer</span>
          </button>
          <div className="flex-1">
            <h1 className="font-display text-lg font-semibold text-silver-100">
              <Layout className="w-5 h-5 text-gold-400 mr-2" />
              Raster Interpolation &amp; Gap-Filling Laboratory
            </h1>
            <p className="text-xs text-silver-500">
              Detect, characterize, reconstruct, validate, and export NoData regions in geospatial rasters.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-1 rounded text-xs font-semibold bg-ink-800 border border-ink-700 text-gold-400">
              Stage {currentIndex + 1} of {stageOrder.length}
            </span>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">
        {error && (
          <div className="mb-6 rounded-xl border border-red-900/40 bg-ink-900 p-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
              <div>
                <h4 className="font-semibold text-red-200">Error</h4>
                <p className="text-xs text-silver-400">{error}</p>
              </div>
            </div>
          </div>
        )}

        <div className="mb-6 flex items-center gap-2">
          {stageOrder.map((s, i) => {
            const isCompleted = i < currentIndex;
            const isActive = i === currentIndex;
            return (
              <div
                key={s}
                className={`flex-1 relative flex flex-col items-center gap-1 px-1 ${
                  isActive ? 'text-gold-400' : 'text-silver-500'
                }`}
              >
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold transition-colors ${
                    isCompleted
                      ? 'bg-gold-400 text-ink-950'
                      : isActive
                        ? 'bg-ink-800 border border-gold-400 text-gold-400'
                        : 'bg-ink-900 border border-ink-700'
                  }`}
                >
                  {i + 1}
                </div>
                {i < stageOrder.length - 1 && (
                  <div className={`absolute -top-2 left-1/2 w-full h-0.5 -translate-x-1/2 ${
                    isCompleted ? 'bg-gold-400' : 'bg-ink-800'
                  }`} />
                )}
                <span className="text-[10px] truncate w-16 text-center">{stageLabels[s]}</span>
              </div>
            );
          })}
        </div>

        <div className="space-y-6">
          <SelectDatasetPanel
            stage={stage}
            rasterData={rasterData ?? []}
            rasterMeta={rasterMeta}
            isLoading={isLoading}
            uploadMode={uploadMode}
            selectedFile={selectedFile}
            selectedFileError={selectedFileError}
            handleInspect={handleInspect}
            handleRegenerate={handleRegenerate}
            handleSelectFile={handleSelectFile}
            handleRemoveFile={handleRemoveFile}
            setUploadMode={setUploadMode}
            setStage={setStage}
            temporalObservations={temporalObservations}
            temporalCube={temporalCube}
            temporalTimes={temporalTimes}
            uploadError={uploadError}
            setUploadError={setUploadError}
            handleAddTemporalObservation={handleAddTemporalObservation}
            handleRemoveTemporalObservation={handleRemoveTemporalObservation}
            handleClearTemporalObservations={handleClearTemporalObservations}
          />
          <InspectMissingPanel
            stage={stage}
            inspection={inspection}
            rasterType={rasterType}
            setRasterType={setRasterType}
            setStage={setStage}
          />
          <ChooseMethodPanel
            stage={stage}
            method={method}
            rasterType={rasterType}
            compatibleMethods={compatibleMethods}
            currentMethodInfo={currentMethodInfo}
            handleChooseMethod={handleChooseMethod}
            handleConfigure={handleConfigure}
            setStage={setStage}
          />
          <ConfigurePanel
            stage={stage}
            method={method}
            rasterType={rasterType}
            setRasterType={setRasterType}
            config={config}
            setConfig={setConfig}
            currentMethodInfo={currentMethodInfo}
            handleValidate={handleValidate}
            setStage={setStage}
          />
          <ValidationPanel
            stage={stage}
            validation={validation}
            handleExecute={handleExecute}
            setStage={setStage}
          />
          <ExecutePanel
            stage={stage}
            result={result}
            rasterData={rasterData ?? []}
            showProvenance={showProvenance}
            setShowProvenance={setShowProvenance}
            showQuality={showQuality}
            setShowQuality={setShowQuality}
            setStage={setStage}
          />
          <CompareResultsPanel
            stage={stage}
            result={result}
            rasterData={rasterData ?? []}
            setStage={setStage}
          />
          <ExportPanel
            stage={stage}
            result={result}
            crs={rasterMeta?.crs ?? null}
            transform={rasterMeta?.transform ?? null}
            nodata={rasterMeta?.nodata ?? null}
            setStage={setStage}
          />
        </div>
      </div>
    </div>
  );
};

export default InterpolationLab;
