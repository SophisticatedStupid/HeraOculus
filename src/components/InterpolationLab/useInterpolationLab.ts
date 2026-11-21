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
} from '../../types/earthScience';
import { apiGet, apiPost } from './api';
import { generateSyntheticRaster } from './synthetic';

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
  handleInspect: () => Promise<void>;
  handleChooseMethod: (m: InterpolationMethod) => void;
  handleConfigure: () => void;
  handleValidate: () => Promise<void>;
  handleExecute: () => Promise<void>;
  handleRegenerate: () => void;
  currentMethodInfo: CompatibleMethod | undefined;
  currentIndex: number;
  stageOrder: InterpolationStage[];
  stageLabels: Record<InterpolationStage, string>;
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

  useEffect(() => {
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
  }, []);

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

  const handleInspect = useCallback(async () => {
    if (!rasterData) return;
    setIsLoading(true);
    setError(null);
    try {
      const data: any = await apiPost('/inspect', {
        data: rasterData,
        band_index: 1,
        user_confirmed_type: rasterType,
      });
      setInspection(data);
      setStage('inspect_missing');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [rasterData, rasterType]);

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
    if (!rasterData) return;
    setIsLoading(true);
    setError(null);
    try {
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
    if (!rasterData) return;
    setIsLoading(true);
    setError(null);
    try {
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
  }, [rasterData, method, rasterType, config, rasterMeta]);

  const handleRegenerate = useCallback(() => {
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
    setInspection(null);
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
    handleInspect,
    handleChooseMethod,
    handleConfigure,
    handleValidate,
    handleExecute,
    handleRegenerate,
    currentMethodInfo,
    currentIndex,
    stageOrder,
    stageLabels,
  };
}
