import React, { useCallback, useEffect, useState } from 'react';
import {
  Cpu,
  ShieldCheck,
  RefreshCw,
  Server,
  TrendingUp,
  Grid,
  Mountain,
  Flame,
  Wind,
  CheckCircle2,
  AlertTriangle,
  SlidersHorizontal,
} from 'lucide-react';
import {
  AuthStatusResponse,
  getBackendBaseUrl,
  HealthResponse,
  phase2Api,
  Phase2SeriesHandoff,
  setBackendBaseUrl,
} from '../../services/phase2Api';
import { ForecastMcmcPanel } from './ForecastMcmcPanel';
import { SpatialFdrPanel } from './SpatialFdrPanel';
import { RasterChangePanel } from './RasterChangePanel';
import { FirmsWildfirePanel } from './FirmsWildfirePanel';
import { AtmosphereNetcdfPanel } from './AtmosphereNetcdfPanel';

export type Phase2ModuleId = 'forecast_mcmc' | 'spatial_fdr' | 'rasters' | 'firms' | 'atmosphere';

interface Phase2LabProps {
  initialHandoff?: Phase2SeriesHandoff | null;
  onClearHandoff?: () => void;
}

export const Phase2Lab: React.FC<Phase2LabProps> = ({ initialHandoff, onClearHandoff }) => {
  const [activeTab, setActiveTab] = useState<Phase2ModuleId>('forecast_mcmc');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthChecking, setHealthChecking] = useState(false);
  const [authStatus, setAuthStatus] = useState<AuthStatusResponse | null>(null);

  const [localHandoff, setLocalHandoff] = useState<Phase2SeriesHandoff | null>(initialHandoff ?? null);
  const [externalCube, setExternalCube] = useState<{
    label: string;
    cube: (number | null)[][][];
  } | null>(null);

  // Backend settings drawer state (no credential inputs: credentials are server-managed)
  const [showAuthDrawer, setShowAuthDrawer] = useState(false);
  const [backendUrlInput, setBackendUrlInput] = useState(getBackendBaseUrl());
  const [backendNotice, setBackendNotice] = useState<string | null>(null);

  useEffect(() => {
    if (initialHandoff) {
      setLocalHandoff(initialHandoff);
      setActiveTab('forecast_mcmc');
    }
  }, [initialHandoff]);

  const checkHealth = useCallback(async () => {
    setHealthChecking(true);
    try {
      const h = await phase2Api.getHealth();
      setHealth(h);
      setAuthStatus(h.auth);
    } catch {
      setHealth(null);
      setAuthStatus(null);
    } finally {
      setHealthChecking(false);
    }
  }, []);

  const refreshAuthStatus = useCallback(async () => {
    setHealthChecking(true);
    try {
      const status = await phase2Api.getAuthStatus();
      setAuthStatus(status);
    } catch {
      setAuthStatus(null);
    } finally {
      setHealthChecking(false);
    }
  }, []);

  useEffect(() => {
    void checkHealth();
  }, [checkHealth]);

  const handleSaveBackendUrl = () => {
    setBackendBaseUrl(backendUrlInput);
    setBackendNotice(
      backendUrlInput.trim()
        ? `Backend target updated to ${backendUrlInput.trim()}`
        : 'Backend target reset to Vite /api proxy (default localhost:8000)'
    );
    void checkHealth();
  };

  const credentialSourceLabel = (source: string, configured: boolean): string => {
    if (!configured) {
      return 'Not configured (keyless fallback)';
    }
    return source === 'environment' ? 'Configured via server environment' : source;
  };

  const modules: Array<{ id: Phase2ModuleId; label: string; icon: React.FC<{ className?: string }> }> = [
    { id: 'forecast_mcmc', label: 'SARIMA, Bayesian MCMC & Extremes', icon: TrendingUp },
    { id: 'spatial_fdr', label: 'Gridded Spatial FDR (Wilks)', icon: Grid },
    { id: 'rasters', label: 'DEM, SAR & LULC Rasters', icon: Mountain },
    { id: 'firms', label: 'FIRMS & EONET Wildfires', icon: Flame },
    { id: 'atmosphere', label: 'Vertical, Trace Gases & NetCDF4', icon: Wind },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-5">
      {/* Header Card */}
      <div className="rounded-2xl bg-ink-900 border border-ink-700 p-5 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-gold-400/15 border border-gold-400/30 text-gold-400 text-[10px] font-bold">
                <Cpu className="w-3.5 h-3.5" />
                Python Compute Engine
              </span>
              {health ? (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-gold-400/15 border border-gold-400/30 text-silver-300 text-[10px] font-bold">
                  <CheckCircle2 className="w-3 h-3" />
                  Python FastAPI Online (v{health.version})
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-gold-400/15 border border-gold-400/40 text-gold-300 text-[10px] font-bold">
                  <AlertTriangle className="w-3 h-3" />
                  Backend Offline (Start uvicorn on :8000)
                </span>
              )}
            </div>
            <h1 className="text-xl sm:text-2xl font-black text-silver-100 tracking-tight">
              HeraOculus Earth Compute &amp; Geospatial Modeling Lab
            </h1>
            <p className="text-xs text-silver-500 max-w-3xl">
              Server-side Python scientific workflows powered by Statsmodels SARIMAX, Adaptive MCMC posterior
              sampling, Wilks Spatial FDR, Xarray NetCDF4/HDF5 extraction, Rasterio DEM/SAR/LULC analysis,
              and live NASA FIRMS / EONET / ASF / CMR feeds.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <button
              onClick={() => void checkHealth()}
              disabled={healthChecking}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-ink-950/90 hover:bg-ink-800 border border-ink-700 text-xs font-semibold text-silver-100 cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${healthChecking ? 'animate-spin' : ''}`} />
              Ping API
            </button>
            <button
              onClick={() => setShowAuthDrawer((v) => !v)}
              className={`inline-flex items-center gap-1.5 px-3 py-2 rounded-xl border text-xs font-semibold cursor-pointer transition ${
                showAuthDrawer
                  ? 'bg-ink-950/60 border-gold-400/40 text-gold-400'
                  : 'bg-ink-950/90 hover:bg-ink-800 border-ink-700 text-silver-100'
              }`}
            >
              <SlidersHorizontal className="w-3.5 h-3.5" />
              Backend &amp; NASA Auth
            </button>
          </div>
        </div>

        {/* Collapsible Backend URL & credential status drawer (no credential inputs: server-managed) */}
        {showAuthDrawer && (
          <div className="rounded-xl bg-ink-950/90 border border-ink-700 p-4 space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex items-center gap-1.5 text-xs font-bold text-silver-100">
                  <Server className="w-4 h-4 text-gold-400" />
                  FastAPI Backend URL (Optional Override)
                </div>
                <p className="text-[11px] text-silver-500">
                  Leave empty in local dev to route seamlessly through Vite&#39;s proxy (<code>/api &rarr; 127.0.0.1:8000</code>),
                  or specify a hosted backend URL (e.g. <code>https://your-fastapi-host.run.app</code>).
                </p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={backendUrlInput}
                    onChange={(e) => setBackendUrlInput(e.target.value)}
                    placeholder="Empty = Vite /api proxy (default localhost:8000)"
                    className="flex-1 px-3 py-1.5 rounded-lg bg-ink-900 border border-ink-700 text-xs text-silver-100"
                  />
                  <button
                    onClick={handleSaveBackendUrl}
                    className="px-3 py-1.5 rounded-lg bg-gold-400 hover:bg-gold-400 text-xs font-bold text-silver-100 cursor-pointer"
                  >
                    Apply URL
                  </button>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-silver-100">
                    <ShieldCheck className="w-4 h-4 text-gold-300" />
                    NASA Credential Status
                  </div>
                  <button
                    onClick={() => void refreshAuthStatus()}
                    disabled={healthChecking}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-ink-900 hover:bg-ink-800 border border-ink-700 text-[10px] font-semibold text-silver-300 cursor-pointer"
                  >
                    <RefreshCw className={`w-3 h-3 ${healthChecking ? 'animate-spin' : ''}`} />
                    Refresh
                  </button>
                </div>
                <p className="text-[11px] text-silver-500">
                  Credentials are configured server-side via environment variables (EARTHDATA_TOKEN,
                  EARTHDATA_USERNAME / EARTHDATA_PASSWORD, NASA_FIRMS_MAP_KEY). They cannot be set from the
                  browser. This panel reflects the deployed backend status only.
                </p>
                <div className="flex items-center gap-2 text-[10px] text-silver-500">
                  <span>
                    EDL: <span className="text-silver-300">{credentialSourceLabel(authStatus?.earthdata_source ?? 'none', authStatus?.earthdata_configured ?? false)}</span>
                  </span>
                  <span>&middot;</span>
                  <span>
                    FIRMS: <span className="text-silver-300">{credentialSourceLabel(authStatus?.firms_source ?? 'none', authStatus?.firms_configured ?? false)}</span>
                  </span>
                </div>
                {authStatus?.keyless_feeds_available?.length ? (
                  <div className="space-y-0.5 pt-1">
                    {authStatus.keyless_feeds_available.map((feed, i) => (
                      <div key={i} className="text-[10px] text-silver-500">
                        • {feed}
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        )}

        {/* 5 Module Tabs */}
        <div className="flex flex-wrap gap-2 pt-1 border-t border-ink-700/80">
          {modules.map((mod) => {
            const Icon = mod.icon;
            const active = activeTab === mod.id;
            return (
              <button
                key={mod.id}
                onClick={() => setActiveTab(mod.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold border transition cursor-pointer ${
                  active
                    ? 'bg-gold-400/15 border-gold-400/40 text-gold-400'
                    : 'bg-ink-950/70 border-ink-700 text-silver-500 hover:text-silver-100 hover:border-ink-700'
                }`}
              >
                <Icon className="w-4 h-4" />
                {mod.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Phase 2 Workspace */}
      {activeTab === 'forecast_mcmc' && (
        <ForecastMcmcPanel
          handoffSeries={localHandoff}
          onClearHandoff={() => {
            setLocalHandoff(null);
            onClearHandoff?.();
          }}
        />
      )}

      {activeTab === 'spatial_fdr' && (
        <SpatialFdrPanel
          externalCube={externalCube}
          onClearExternalCube={() => setExternalCube(null)}
        />
      )}

      {activeTab === 'rasters' && <RasterChangePanel />}

      {activeTab === 'firms' && <FirmsWildfirePanel />}

      {activeTab === 'atmosphere' && (
        <AtmosphereNetcdfPanel
          onSendSeries={(series) => {
            setLocalHandoff(series);
            setActiveTab('forecast_mcmc');
          }}
          onSendCube={(cubePayload) => {
            setExternalCube(cubePayload);
            setActiveTab('spatial_fdr');
          }}
        />
      )}
    </div>
  );
};
