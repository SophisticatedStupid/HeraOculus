import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  CompatibleMethod,
  InterpolationMethod,
  InterpolationRasterType,
  InterpolationStage,
  RasterInspectionResult,
  ValidationResult,
  ValidationStrategy,
} from '../../types/earthScience';
import { apiGet, apiPost, apiPostForm, readRasterFile } from './api';
import { generateSyntheticRaster } from './synthetic';

export interface TemporalObservation {
  id: string;
  file: File;
  timestamp: string; // user-provided timestamp string (preserved)
  time: number; // decimal-year time coordinate (sent to backend as `times`)
  data: number[][]; // 2D array, NoData preserved as NaN
  width: number;
  height: number;
  crs: string | null;
  nodata: number | null;
}

export interface UseInterpolationLabReturn {
  stage: InterpolationStage;
  setStage: React.Dispatch<React.SetStateAction<InterpolationStage>>;
  rasterData: ((number | null)[][]) | null;
  rasterMeta: {
    width: number;
    height: number;
    crs: string | null;
    transform: any;
    nodata: number | null;
  } | null;
  inspection: RasterInspectionResult | null;
  method: InterpolationMethod;
  rasterType: InterpolationRasterType;
  setRasterType: React.Dispatch<React.SetStateAction<InterpolationRasterType>>;
  compatibleMethods: CompatibleMethod[];
  config: Record<string, any>;
  setConfig: React.Dispatch<React.SetStateAction<Record<string, any>>>;
  validation: ValidationResult | null;
  result: any;
  isLoading: boolean;
  error: string | null;
  showProvenance: boolean;
  setShowProvenance: React.Dispatch<React.SetStateAction<boolean>>;
  showQuality: boolean;
  setShowQuality: React.Dispatch<React.SetStateAction<boolean>>;
  // dataset source mode: 'demo' (synthetic) or 'upload' (real raster file)
  uploadMode: 'demo' | 'upload';
  setUploadMode: (mode: 'demo' | 'upload') => void;
  selectedFile: File | null;
  selectedFileError: string | null;
  handleSelectFile: (file: File) => Promise<void>;
  handleRemoveFile: () => void;
  handleInspect: () => Promise<void>;
  handleRegenerate: () => void;
  // temporal raster support
  temporalObservations: TemporalObservation[];
  temporalCube: number[][][] | null;
  temporalTimes: number[] | null;
  uploadError: string | null;
  setUploadError: (msg: string | null) => void;
  handleAddTemporalObservation: (file: File, timestamp: string) => Promise<void>;
  handleRemoveTemporalObservation: (id: string) => void;
  handleClearTemporalObservations: () => void;
  handleChooseMethod: (m: InterpolationMethod) => void;
  handleConfigure: () => void;
  handleValidate: () => Promise<void>;
  handleExecute: () => Promise<void>;
  currentMethodInfo: CompatibleMethod | undefined;
  currentIndex: number;
  stageOrder: InterpolationStage[];
  stageLabels: Record<InterpolationStage, string>;
}

const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100 MB
const RASTER_EXTENSIONS = ['.tif', '.tiff', '.geotiff'];

export function validateFile(file: File): string | null {
  const name = file.name.toLowerCase();
  if (!RASTER_EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return 'Unsupported file type. Upload a GeoTIFF (.tif, .tiff, .geotiff).';
  }
  if (file.size > MAX_FILE_SIZE) {
    return 'File too large. Maximum 100 MB.';
  }
  return null;
}

/** Convert a user-provided timestamp to a decimal-year time coordinate. */
export function toDecimalYear(timestamp: string): number {
  if (!timestamp || !timestamp.trim()) return NaN;
  const n = Number(timestamp);
  if (Number.isFinite(n)) return n;
  const d = new Date(timestamp);
  if (!Number.isNaN(d.getTime())) {
    const start = new Date(d.getFullYear(), 0, 0);
    const end = new Date(d.getFullYear() + 1, 0, 0);
    return d.getFullYear() + (d.getTime() - start.getTime()) / (end.getTime() - start.getTime());
  }
  return NaN;
}

/**
 * Assemble a (T, H, W) temporal cube from temporal observations.
 *
 * Validates that all observations share dimensions and CRS, that timestamps
 * are unique, and orders by time.  Returns null when the cube cannot be
 * assembled (empty input, duplicate timestamp).  Missing/NoData values within
 * each 2D array are preserved as NaN.
 */
export function assembleTemporalCube(
  observations: TemporalObservation[]
): { cube: number[][][]; times: number[] } | null {
  if (observations.length === 0) return null;
  const sorted = [...observations].sort((a, b) => a.time - b.time);
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].time === sorted[i - 1].time) return null; // duplicate timestamp
  }
  return {
    cube: sorted.map((o) => o.data),
    times: sorted.map((o) => o.time),
  };
}

export function useInterpolationLab({ onClose }: { onClose?: () => void }): UseInterpolationLabReturn {
  const [stage, setStage] = useState<InterpolationStage>('select_dataset');
  const [rasterData, setRasterData] = useState<(number | null)[][] | null>(null);
  const [rasterMeta, setRasterMeta] = useState<{
    width: number;
    height: number;
    crs: string | null;
    transform: any;
    nodata: number | null;
  } | null>(null);
  const [inspection, setInspection] = useState<RasterInspectionResult | null>(null);
  const [method, setMethod] = useState<InterpolationMethod>('nearest_neighbor');
  const [rasterType, setRasterType] = useState<InterpolationRasterType>('continuous');
  const [compatibleMethods, setCompatibleMethods] = useState<CompatibleMethod[]>([]);
  const [config, setConfig] = useState<Record<string, any>>({});
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [result, setResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showProvenance, setShowProvenance] = useState(true);
  const [showQuality, setShowQuality] = useState(false);

  // dataset source mode
  const [uploadMode, setUploadModeState] = useState<'demo' | 'upload'>('demo');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedFileError, setSelectedFileError] = useState<string | null>(null);

  // temporal raster support
  const [temporalObservations, setTemporalObservations] = useState<TemporalObservation[]>([]);
  const [temporalCube, setTemporalCube] = useState<number[][][] | null>(null);
  const [temporalTimes, setTemporalTimes] = useState<number[] | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // prevents duplicate submissions while an upload/inspect/execute is in flight
  const submissionLock = useRef(false);

  useEffect(() => {
    if (uploadMode !== 'demo') return;
    const synth = generateSyntheticRaster({
      width: 64,
      height: 64,
      type: 'continuous',
      missingPct: 0.2,
      seed: 42,
    });
    setRasterData(
      synth.data.map((row) => row.map((v) => (Number.isNaN(v) ? null : v)))
    );
    setRasterMeta({
      width: synth.width,
      height: synth.height,
      crs: synth.crs,
      transform: synth.transform,
      nodata: synth.nodata,
    });
    setStage('select_dataset');
  }, [uploadMode]);

  useEffect(() => {
    let mounted = true;
    apiGet(`/interpolate/compatible-methods?raster_type=${rasterType}`)
      .then((data: any) => {
        if (mounted) setCompatibleMethods(data.methods || []);
      })
      .catch((err: any) => console.warn('Failed to load compatible methods:', err));
    return () => {
      mounted = false;
    };
  }, [rasterType]);

  const handleSelectFile = useCallback(async (file: File) => {
    const err = validateFile(file);
    if (err) {
      setSelectedFileError(err);
      setSelectedFile(null);
      setRasterData(null);
      setRasterMeta(null);
      return;
    }
    setIsLoading(true);
    setSelectedFileError(null);
    try {
      const raster = await readRasterFile(file);
      setSelectedFile(file);
      setRasterData(
        raster.data.map((row) =>
          row.map((v) => (Number.isNaN(v) ? null : v))
        )
      );
      setRasterMeta({
        width: raster.width,
        height: raster.height,
        crs: raster.crs,
        transform: raster.transform,
        nodata: raster.nodata,
      });
      setSelectedFileError(null);
    } catch (e: any) {
      setSelectedFileError(e.message || 'Failed to read raster file');
      setSelectedFile(null);
      setRasterData(null);
      setRasterMeta(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleRemoveFile = useCallback(() => {
    setSelectedFile(null);
    setSelectedFileError(null);
    setRasterData(null);
    setRasterMeta(null);
    setInspection(null);
    setResult(null);
    setError(null);
    setStage('select_dataset');
  }, []);

  const handleInspect = useCallback(async () => {
    if (uploadMode === 'demo') {
      if (!rasterData) {
        setError('No raster data available');
        return;
      }
      // Demo mode: fabricate the inspection result client-side from the
      // synthetic raster.  No backend round-trip is needed (and impossible,
      // because /inspect expects a real multipart file upload).
      const flat = (rasterData as number[][]).flat();
      const valid = flat.filter((v) => v !== null).length;
      const missing = flat.length - valid;
      const inspection: RasterInspectionResult = {
        width: rasterMeta?.width ?? 0,
        height: rasterMeta?.height ?? 0,
        bands: 1,
        crs: rasterMeta?.crs ?? null,
        transform: rasterMeta?.transform ?? null,
        pixel_size_x: null,
        pixel_size_y: null,
        extent: null,
        data_type: 'float32',
        nodata_value: rasterMeta?.nodata ?? null,
        has_alpha_band: false,
        band_index: 1,
        valid_pixel_count: valid,
        missing_pixel_count: missing,
        missing_percentage: (missing / flat.length) * 100,
        valid_percentage: (valid / flat.length) * 100,
        band_stats: [
          {
            band: 1,
            min: null,
            max: null,
            mean: null,
            std: null,
            valid_count: valid,
            missing_count: missing,
          },
        ],
        raster_type: rasterType,
        missing_region_count: 0,
        missing_region_stats: [],
        largest_missing_region_pixels: 0,
        average_missing_region_size: 0,
        missing_region_size_distribution: {},
        isolated_missing_pixels: 0,
        edge_gaps: 0,
        interior_gaps: 0,
        distance_to_valid_stats: {},
        missing_spatial_distribution: {},
        processing_duration_ms: 0,
        inspected_at: new Date().toISOString(),
      };
      setInspection(inspection);
      setStage('inspect_missing');
      return;
    }

    // Upload mode: send the real raster file as multipart form data to the
    // backend inspection endpoint (the authoritative contract).
    if (!selectedFile) {
      setError('No raster file selected');
      return;
    }
    if (submissionLock.current) return;
    submissionLock.current = true;
    setIsLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('band_index', '1');
      if (rasterType) {
        formData.append('user_confirmed_type', rasterType);
      }
      const data: any = await apiPostForm('/inspect', formData);
      setInspection(data);
      // rasterData/rasterMeta were populated by handleSelectFile (readRasterFile)
      setStage('inspect_missing');
    } catch (err: any) {
      setError(err.message || 'Inspection failed');
    } finally {
      setIsLoading(false);
      submissionLock.current = false;
    }
  }, [rasterData, rasterMeta, rasterType, selectedFile, uploadMode]);

  const handleChooseMethod = useCallback((m: InterpolationMethod) => {
    setMethod(m);
    const compat = compatibleMethods.find((cm) => cm.id === m);
    if (compat) {
      setConfig({ ...compat.default_config });
    }
    setStage('choose_method');
  }, [compatibleMethods]);

  const handleConfigure = useCallback(() => {
    setStage('configure');
  }, []);

  const handleValidate = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      if (method.startsWith('temporal_')) {
        // The backend /validate endpoint does not support temporal methods.
        // Record a client-side validation summary and proceed to execution.
        const validation: ValidationResult = {
          method,
          config,
          validation_strategy: 'spatial_block' as ValidationStrategy,
          validation_sample_count: 0,
          n_excluded_from_validation: 0,
          mae: null,
          rmse: null,
          bias: null,
          median_absolute_error: null,
          mape: null,
          r_squared: null,
          reconstruction_percentage: 0,
          remaining_nodata_percentage: 0,
          excluded_percentage: 0,
          processing_duration_ms: 0,
          warnings: [
            'Cross-validation is not available for temporal methods; proceeding to execution.',
          ],
          limitations: ['Temporal cross-validation is not supported by the backend.'],
          validation_predicted: [],
          validation_observed: [],
          validation_residuals: [],
          validation_coords: [],
        };
        setValidation(validation);
        setStage('validate');
        return;
      }
      if (!rasterData) return;
      const data: any = await apiPost('/interpolate/validate', {
        data: rasterData,
        method,
        raster_type: rasterType,
        ...config,
      });
      setValidation(data);
      setStage('validate');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [rasterData, method, rasterType, config]);

  const handleExecute = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      if (method.startsWith('temporal_')) {
        const assembled = assembleTemporalCube(temporalObservations);
        if (!assembled) {
          setError(
            'Temporal cube could not be assembled. Add temporal observations with matching dimensions and CRS, unique timestamps, and valid data.'
          );
          return;
        }
        const data: any = await apiPost('/interpolate', {
          data: assembled.cube,
          method,
          band_index: 1,
          raster_type: 'temporal',
          times: assembled.times,
          max_gap_steps: config.max_gap_steps,
          allow_extrapolation: config.allow_extrapolation,
        });
        setResult(data);
        setStage('execute');
        return;
      }
      if (!rasterData) return;
      const data: any = await apiPost('/interpolate', {
        data: rasterData,
        method,
        raster_type: rasterType,
        max_distance_pixels: config.max_distance_pixels,
        geographic_crs: config.geographic_crs,
        pixel_size_x: rasterMeta?.transform?.a ?? undefined,
        pixel_size_y: rasterMeta?.transform?.e ? Math.abs(rasterMeta.transform.e) : undefined,
        neighborhood: config.neighborhood,
        edge_handling: config.edge_handling,
        clip_to_observed_range: config.clip_to_observed_range,
        window_size: config.window_size,
        min_valid_neighbors: config.min_valid_neighbors,
        neighborhood_shape: config.neighborhood_shape,
        power: config.power,
        search_radius_pixels: config.search_radius_pixels,
        max_neighbors: config.max_neighbors,
        min_neighbors: config.min_neighbors,
        variogram_model: config.variogram_model,
        variogram_nugget: config.variogram_nugget,
        variogram_sill: config.variogram_sill,
        variogram_range: config.variogram_range,
        max_gap_steps: config.max_gap_steps,
        allow_extrapolation: config.allow_extrapolation,
      });
      setResult(data);
      setStage('execute');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [
    rasterData,
    rasterMeta,
    method,
    rasterType,
    config,
    assembleTemporalCube,
  ]);

  const handleAddTemporalObservation = useCallback(
    async (file: File, timestamp: string) => {
      const err = validateFile(file);
      if (err) {
        setUploadError(err);
        return;
      }
      const time = toDecimalYear(timestamp);
      if (!Number.isFinite(time)) {
        setUploadError('Invalid timestamp. Use a year (e.g. 2020.5) or ISO date.');
        return;
      }
      if (temporalObservations.some((o) => o.time === time)) {
        setUploadError('Duplicate timestamp. Each temporal observation must have a unique time.');
        return;
      }
      setIsLoading(true);
      setUploadError(null);
      try {
        const raster = await readRasterFile(file);
        for (const o of temporalObservations) {
          if (o.width !== raster.width || o.height !== raster.height) {
            setUploadError('All temporal rasters must have identical dimensions.');
            return;
          }
          if (o.crs !== null && raster.crs !== null && o.crs !== raster.crs) {
            setUploadError('All temporal rasters must share the same CRS.');
            return;
          }
        }
        const obs: TemporalObservation = {
          id: crypto.randomUUID(),
          file,
          timestamp,
          time,
          data: raster.data,
          width: raster.width,
          height: raster.height,
          crs: raster.crs,
          nodata: raster.nodata,
        };
        setTemporalObservations((prev) => [...prev, obs]);
        setUploadError(null);
        // Expose the assembled cube/times so downstream stages can inspect them.
        const assembled = assembleTemporalCube(temporalObservations);
        if (assembled) {
          setTemporalCube(assembled.cube);
          setTemporalTimes(assembled.times);
        }
      } catch (e: any) {
        setUploadError(e.message || 'Failed to read temporal raster');
      } finally {
        setIsLoading(false);
      }
    },
    [temporalObservations, assembleTemporalCube]
  );

  const handleRemoveTemporalObservation = useCallback((id: string) => {
    setTemporalObservations((prev) => prev.filter((o) => o.id !== id));
    const filtered = temporalObservations.filter((o) => o.id !== id);
    const assembled = assembleTemporalCube(filtered);
    if (assembled) {
      setTemporalCube(assembled.cube);
      setTemporalTimes(assembled.times);
    } else {
      setTemporalCube(null);
      setTemporalTimes(null);
    }
  }, [temporalObservations]);

  const handleClearTemporalObservations = useCallback(() => {
    setTemporalObservations([]);
    setTemporalCube(null);
    setTemporalTimes(null);
    setUploadError(null);
  }, []);

  const handleRegenerate = useCallback(() => {
    setUploadModeState('demo');
    setSelectedFile(null);
    setSelectedFileError(null);
    setTemporalObservations([]);
    setTemporalCube(null);
    setTemporalTimes(null);
    setInspection(null);
    setResult(null);
    setError(null);
    setUploadError(null);
    const synth = generateSyntheticRaster({
      width: 64,
      height: 64,
      type: 'continuous',
      missingPct: 0.3,
      seed: 123,
    });
    setRasterData(
      synth.data.map((row) => row.map((v) => (Number.isNaN(v) ? null : v)))
    );
    setRasterMeta({
      width: synth.width,
      height: synth.height,
      crs: synth.crs,
      transform: synth.transform,
      nodata: synth.nodata,
    });
    setStage('select_dataset');
  }, []);

  const setUploadMode = useCallback((mode: 'demo' | 'upload') => {
    setUploadModeState(mode);
    setSelectedFile(null);
    setSelectedFileError(null);
    setRasterData(null);
    setRasterMeta(null);
    setTemporalObservations([]);
    setTemporalCube(null);
    setTemporalTimes(null);
    setInspection(null);
    setResult(null);
    setError(null);
    setUploadError(null);
    setStage('select_dataset');
  }, []);

  const stageOrder: InterpolationStage[] = [
    'select_dataset',
    'inspect_missing',
    'choose_method',
    'configure',
    'validate',
    'execute',
    'compare_results',
    'export',
  ];

  const stageLabels: Record<InterpolationStage, string> = {
    select_dataset: '1. Select Dataset',
    inspect_missing: '2. Inspect Missing Data',
    choose_method: '3. Choose Interpolation',
    configure: '4. Configure Parameters',
    validate: '5. Cross-Validation',
    execute: '6. Execute Interpolation',
    compare_results: '7. Compare Results',
    export: '8. Export',
  };

  const currentIndex = stageOrder.indexOf(stage);

  const currentMethodInfo = useMemo(() => {
    return compatibleMethods.find((cm) => cm.id === method);
  }, [method, compatibleMethods]);

  return {
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
    uploadMode,
    setUploadMode,
    selectedFile,
    selectedFileError,
    handleSelectFile,
    handleRemoveFile,
    handleInspect,
    handleRegenerate,
    temporalObservations,
    temporalCube,
    temporalTimes,
    uploadError,
    setUploadError,
    handleAddTemporalObservation,
    handleRemoveTemporalObservation,
    handleClearTemporalObservations,
    handleChooseMethod,
    handleConfigure,
    handleValidate,
    handleExecute,
    currentMethodInfo,
    currentIndex,
    stageOrder,
    stageLabels,
  };
}
