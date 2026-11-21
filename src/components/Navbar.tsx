import React, { useEffect, useRef, useState } from 'react';
import {
  Globe2,
  Search,
  GitCompare,
  BookOpen,
  UploadCloud,
  FileText,
  FlaskConical,
  Cpu,
  Layout,
  Loader2,
  Satellite,
  RefreshCw,
  Image,
  Upload,
  Menu,
  X,
  type LucideIcon,
} from 'lucide-react';

export type AppMode = 'explorer' | 'detective' | 'contrast' | 'encyclopedia' | 'upload' | 'lab' | 'phase2' | 'interpolation' | 'imagery';
export type DataFeedSource = 'live' | 'calibrated' | 'csv' | 'satellite' | 'reanalysis';

const DATA_FEED_META: Record<
  string,
  { shortLabel: string; label: string; colorClass: string; pulse: boolean; icon: LucideIcon }
> = {
  live: {
    shortLabel: 'NASA POWER',
    label: 'NASA POWER reanalysis',
    colorClass: 'text-gold-400',
    pulse: true,
    icon: RefreshCw,
  },
  reanalysis: {
    shortLabel: 'Reanalysis',
    label: 'NASA model / reanalysis',
    colorClass: 'text-gold-400',
    pulse: true,
    icon: RefreshCw,
  },
  satellite: {
    shortLabel: 'Sat obs',
    label: 'Satellite observation (MODIS/NOAA/USGS)',
    colorClass: 'text-silver-500',
    pulse: true,
    icon: Satellite,
  },
  csv: {
    shortLabel: 'Custom CSV',
    label: 'User upload (CSV/JSON)',
    colorClass: 'text-gold-300',
    pulse: false,
    icon: Upload,
  },
  calibrated: {
    shortLabel: 'Calibrated',
    label: 'Calibrated demo model',
    colorClass: 'text-silver-600',
    pulse: false,
    icon: FlaskConical,
  },
};

interface NavbarProps {
  currentMode: AppMode;
  onSelectMode: (mode: AppMode) => void;
  onOpenExport: () => void;
  dataFeedSource: DataFeedSource;
  selectedRegionName: string;
  isLoadingFeed?: boolean;
}

const navItems: { mode: AppMode; label: string; Icon: LucideIcon }[] = [
  { mode: 'explorer', label: 'Trend explorer', Icon: Globe2 },
  { mode: 'detective', label: 'Data detective', Icon: Search },
  { mode: 'contrast', label: 'Opposite trends lab', Icon: GitCompare },
  { mode: 'encyclopedia', label: 'Encyclopedia', Icon: BookOpen },
  { mode: 'upload', label: 'Custom data', Icon: UploadCloud },
  { mode: 'lab', label: 'Statistical lab', Icon: FlaskConical },
  { mode: 'phase2', label: 'Earth compute lab', Icon: Cpu },
  { mode: 'interpolation', label: 'Raster Interpolation', Icon: Layout },
  { mode: 'imagery', label: 'Imagery Search', Icon: Image },
];

function HeraOculusLogo() {
  return (
    <svg
      width="32"
      height="32"
      viewBox="0 0 100 100"
      aria-hidden
      className="shrink-0"
    >
      <circle cx="50" cy="50" r="42" fill="none" stroke="#D4AF37" strokeWidth="5" />
      <circle cx="50" cy="50" r="18" fill="none" stroke="#F2F2F2" strokeWidth="4" />
      <circle cx="50" cy="50" r="6" fill="#D4AF37" />
    </svg>
  );
}

export const Navbar: React.FC<NavbarProps> = ({
  currentMode,
  onSelectMode,
  onOpenExport,
  dataFeedSource,
  selectedRegionName,
  isLoadingFeed = false,
}) => {
  const feedMeta = DATA_FEED_META[dataFeedSource];
  const FeedIcon = feedMeta.icon;
  const [drawerOpen, setDrawerOpen] = useState(false);
  const drawerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDrawerOpen(false);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [drawerOpen]);

  const handleSelect = (mode: AppMode) => {
    onSelectMode(mode);
    setDrawerOpen(false);
  };

  return (
    <header className="sticky top-0 z-50 bg-ink-950/95 backdrop-blur-md border-b border-ink-700 px-4 lg:px-8 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        {/* Brand */}
        <button
          type="button"
          className="flex items-center space-x-3 text-left"
          onClick={() => handleSelect('explorer')}
        >
          <HeraOculusLogo />
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-display text-xl font-semibold leading-none text-silver-100">
                HeraOculus
              </span>
              <span className="px-1.5 py-0.5 text-[10px] font-medium bg-ink-800 text-silver-300 rounded border border-ink-700">
                HeraOculus
              </span>
            </div>
            <p className="text-xs text-silver-500 font-medium">
              Earth science change detective & significance engine
            </p>
          </div>
        </button>

        {/* Data feed status (desktop) */}
        <div className="hidden md:flex items-center space-x-3">
          <div className="flex items-center space-x-2 px-3 py-1.5 text-xs">
            {isLoadingFeed ? (
              <>
                <Loader2 className="w-3.5 h-3.5 text-gold-300 animate-spin" />
                <div>
                  <span className="text-silver-500 text-[10px] block leading-none">Data feed</span>
                  <span className="font-semibold text-gold-300">Fetching live data…</span>
                </div>
              </>
            ) : (
              <>
                <FeedIcon
                  className={`w-3.5 h-3.5 ${feedMeta.colorClass} ${feedMeta.pulse ? 'motion-safe:animate-pulse' : ''}`}
                />
                <div>
                  <span className="text-silver-500 text-[10px] block leading-none">Data feed</span>
                  <span className={`font-semibold ${feedMeta.colorClass}`}>{feedMeta.label}</span>
                </div>
              </>
            )}
          </div>

          <button
            type="button"
            onClick={onOpenExport}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-silver-100 border border-ink-700 hover:bg-ink-800 transition-colors"
            title={`Export dossier for ${selectedRegionName}`}
          >
            <FileText className="w-3.5 h-3.5 text-gold-400" />
            <span>Export report</span>
          </button>
        </div>

        {/* Mobile menu button */}
        <button
          type="button"
          className="md:hidden inline-flex items-center justify-center w-9 h-9 rounded-lg border border-ink-700 text-silver-100 hover:bg-ink-800"
          aria-label="Open navigation menu"
          aria-expanded={drawerOpen}
          aria-controls="mobile-nav-drawer"
          onClick={() => setDrawerOpen((v) => !v)}
        >
          {drawerOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
        </button>
      </div>

      {/* Desktop nav */}
      <nav className="hidden md:flex max-w-7xl mx-auto mt-3 items-center gap-1 overflow-x-auto">
        {navItems.map(({ mode, label, Icon }) => {
          const active = currentMode === mode;
          return (
            <button
              key={mode}
              type="button"
              onClick={() => handleSelect(mode)}
              className={`relative flex items-center space-x-2 px-3 py-1.5 text-xs font-medium whitespace-nowrap transition-colors ${
                active ? 'text-gold-400' : 'text-silver-500 hover:text-silver-100'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{label}</span>
              {active && (
                <span className="absolute left-2 right-2 -bottom-0.5 h-0.5 bg-gold-400 rounded-full" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Mobile drawer */}
      {drawerOpen && (
        <div
          id="mobile-nav-drawer"
          ref={drawerRef}
          className="md:hidden mt-3 rounded-xl border border-ink-700 bg-ink-900 p-2 space-y-1"
          role="dialog"
          aria-modal="true"
          aria-label="Navigation menu"
        >
          {navItems.map(({ mode, label, Icon }) => {
            const active = currentMode === mode;
            return (
              <button
                key={mode}
                type="button"
                onClick={() => handleSelect(mode)}
                className={`w-full flex items-center space-x-2 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                  active
                    ? 'text-gold-400 bg-ink-800/60 border-l-2 border-gold-400'
                    : 'text-silver-500 hover:text-silver-100 hover:bg-ink-800/60'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{label}</span>
              </button>
            );
          })}
          <div className="pt-2 mt-2 border-t border-ink-700 flex items-center justify-between px-3">
            <div className="flex items-center space-x-2 text-xs">
              {isLoadingFeed ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 text-gold-300 animate-spin" />
                  <span className="text-gold-300">Fetching live data…</span>
                </>
              ) : (
                <>
                  <FeedIcon className={`w-3.5 h-3.5 ${feedMeta.colorClass}`} />
                  <span className={feedMeta.colorClass}>{feedMeta.label}</span>
                </>
              )}
            </div>
            <button
              type="button"
              onClick={() => {
                onOpenExport();
                setDrawerOpen(false);
              }}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-silver-100 border border-ink-700 hover:bg-ink-800"
            >
              <FileText className="w-3.5 h-3.5 text-gold-400" />
              <span>Export report</span>
            </button>
          </div>
        </div>
      )}
    </header>
  );
};
