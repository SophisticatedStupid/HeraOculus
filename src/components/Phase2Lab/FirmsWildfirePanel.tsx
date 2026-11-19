import React, { useEffect, useState } from 'react';
import { Bar, Scatter } from 'react-chartjs-2';
import { Flame, Play, Loader2, MapPin } from 'lucide-react';
import { FirmsQueryResponse, phase2Api } from '../../services/phase2Api';
import { MetricTile, Phase2ErrorAlert, ProvenanceBanner, CHART_AXIS_TICK, CHART_AXIS_LABEL, CHART_GRID, CHART_GRID_SOFT, CHART_SERIES, CHART_LEGEND } from './shared';

interface BboxPreset {
  label: string;
  west: number;
  south: number;
  east: number;
  north: number;
}

const FIRE_BBOX_PRESETS: BboxPreset[] = [
  { label: 'California & Western Sierra', west: -125, south: 32, east: -114, north: 43 },
  { label: 'Amazon Deforestation Arc', west: -68, south: -14, east: -48, north: -2 },
  { label: 'Canadian Boreal Shield', west: -124, south: 52, east: -92, north: 64 },
  { label: 'Mediterranean Basin', west: -8, south: 35, east: 28, north: 45 },
  { label: 'Southeast Australia', west: 140, south: -39, east: 154, north: -26 },
];

export const FirmsWildfirePanel: React.FC = () => {
  const [presetIdx, setPresetIdx] = useState(0);
  const [bbox, setBbox] = useState<BboxPreset>(FIRE_BBOX_PRESETS[0]);
  const [source, setSource] = useState<'VIIRS_SNPP_NRT' | 'VIIRS_NOAA20_NRT' | 'MODIS_NRT'>(
    'VIIRS_SNPP_NRT'
  );
  const [dayRange, setDayRange] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [res, setRes] = useState<FirmsQueryResponse | null>(null);

  const runQuery = async (targetBox: BboxPreset = bbox) => {
    setLoading(true);
    setError(null);
    try {
      const out = await phase2Api.queryFirmsFires({
        west: targetBox.west,
        south: targetBox.south,
        east: targetBox.east,
        north: targetBox.north,
        source,
        day_range: dayRange,
        region_label: targetBox.label,
        allow_eonet_fallback: true,
      });
      setRes(out);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void runQuery(bbox);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handlePresetChange = (idx: number) => {
    setPresetIdx(idx);
    const chosen = FIRE_BBOX_PRESETS[idx];
    setBbox(chosen);
    void runQuery(chosen);
  };

  return (
    <div className="space-y-4">
      <div className="bg-ink-900/90 border border-ink-700 rounded-2xl p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3 items-end">
        <label className="text-[11px] text-silver-300 space-y-1 lg:col-span-2">
          <span className="block text-silver-500 font-semibold">Fire Regime / Region Preset</span>
          <select
            value={presetIdx}
            onChange={(e) => handlePresetChange(Number(e.target.value))}
            className="w-full px-2.5 py-1.5 rounded-lg bg-ink-950 border border-ink-700 text-xs text-silver-100"
          >
            {FIRE_BBOX_PRESETS.map((p, i) => (
              <option key={p.label} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>

        <label className="text-[11px] text-silver-300 space-y-1">
          <span className="block text-silver-500 font-semibold">FIRMS Instrument</span>
          <select
            value={source}
            onChange={(e) => setSource(e.target.value as typeof source)}
            className="w-full px-2.5 py-1.5 rounded-lg bg-ink-950 border border-ink-700 text-xs text-silver-100"
          >
            <option value="VIIRS_SNPP_NRT">VIIRS S-NPP (375m)</option>
            <option value="VIIRS_NOAA20_NRT">VIIRS NOAA-20 (375m)</option>
            <option value="MODIS_NRT">MODIS Terra/Aqua (1km)</option>
          </select>
        </label>

        <label className="text-[11px] text-silver-300 space-y-1">
          <span className="block text-silver-500 font-semibold">Lookback (Days)</span>
          <input
            type="number"
            min={1}
            max={10}
            value={dayRange}
            onChange={(e) => setDayRange(Math.max(1, Math.min(10, Number(e.target.value) || 5)))}
            className="w-full px-2.5 py-1.5 rounded-lg bg-ink-950 border border-ink-700 text-xs text-silver-100"
          />
        </label>

        <div className="text-[11px] text-silver-300 space-y-1">
          <span className="block text-silver-500 font-semibold">BBox [W, S, E, N]</span>
          <div className="grid grid-cols-4 gap-1 text-[10px]">
            {(['west', 'south', 'east', 'north'] as const).map((k) => (
              <input
                key={k}
                type="number"
                value={bbox[k]}
                onChange={(e) => setBbox((p) => ({ ...p, label: 'Custom BBox', [k]: Number(e.target.value) || 0 }))}
                className="px-1 py-1.5 rounded bg-ink-950 border border-ink-700 text-center text-silver-100"
              />
            ))}
          </div>
        </div>

        <button
          onClick={() => void runQuery(bbox)}
          disabled={loading}
          className="flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl bg-gold-400 hover:bg-gold-400 disabled:opacity-50 text-xs font-bold text-silver-100 cursor-pointer"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          Fetch Active Fires
        </button>
      </div>

      <Phase2ErrorAlert error={error} onRetry={() => void runQuery(bbox)} />

      {res && (
        <>
          <ProvenanceBanner provenance={res.provenance} />

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <MetricTile
              label="Active Detections"
              value={res.summary.total_detections}
              sub={`High-conf: ${res.summary.high_confidence_count}`}
              accent="text-gold-300"
            />
            <MetricTile
              label="Total FRP"
              value={res.summary.total_frp_mw !== null ? `${res.summary.total_frp_mw.toLocaleString()} MW` : 'Event Track'}
              accent="text-gold-300"
            />
            <MetricTile
              label="Mean FRP / Hotspot"
              value={res.summary.mean_frp_mw !== null ? `${res.summary.mean_frp_mw} MW` : 'N/A (EONET)'}
            />
            <MetricTile
              label="95th Pct / Max FRP"
              value={res.summary.max_frp_mw !== null ? `${res.summary.p95_frp_mw ?? '—'} / ${res.summary.max_frp_mw} MW` : '—'}
            />
            <MetricTile
              label="Source Feed"
              value={res.source}
              sub={res.region_label}
              accent="text-gold-400"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-ink-900/90 border border-ink-700 rounded-2xl p-4">
              <div className="text-xs font-bold text-silver-100 mb-2 flex items-center gap-1.5">
                <Flame className="w-4 h-4 text-gold-400" />
                Daily Thermal Activity &amp; Fire Radiative Power
              </div>
              <div className="h-64">
                <Bar
                  data={{
                    labels: res.daily_series.map((d) => d.date),
                    datasets: [
                      {
                        label:
                          res.summary.total_frp_mw !== null
                            ? 'Total Daily FRP (MW)'
                            : 'Wildfire Event Count',
                        data: res.daily_series.map((d) =>
                          d.total_frp_mw !== null ? d.total_frp_mw : d.count
                        ),
                        backgroundColor: 'rgba(212, 175, 55, 0.72)',
                        borderColor: CHART_SERIES.gold400,
                        borderWidth: 1,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: CHART_AXIS_LABEL, font: { size: 11 } } } },
                    scales: {
                      x: { ticks: { color: CHART_AXIS_TICK }, grid: { display: false } },
                      y: { ticks: { color: CHART_AXIS_TICK }, grid: { color: CHART_GRID } },
                    },
                  }}
                />
              </div>
            </div>

            <div className="bg-ink-900/90 border border-ink-700 rounded-2xl p-4">
              <div className="text-xs font-bold text-silver-100 mb-2 flex items-center gap-1.5">
                <MapPin className="w-4 h-4 text-gold-300" />
                Thermal Footprint Coordinate Scatter (Lon &times; Lat)
              </div>
              <div className="h-64">
                <Scatter
                  data={{
                    datasets: [
                      {
                        label: 'Detected Thermal Hotspots',
                        data: res.hotspots.map((h) => ({ x: h.longitude, y: h.latitude })),
                        backgroundColor: res.hotspots.map((h) =>
                          (h.frp_mw ?? 0) > 80 ? CHART_SERIES.gold400 : CHART_SERIES.gold300
                        ),
                        pointRadius: 4,
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: CHART_AXIS_LABEL, font: { size: 11 } } } },
                    scales: {
                      x: {
                        title: { display: true, text: 'Longitude (°E)', color: CHART_AXIS_TICK },
                        ticks: { color: CHART_AXIS_TICK },
                        grid: { color: CHART_GRID_SOFT },
                      },
                      y: {
                        title: { display: true, text: 'Latitude (°N)', color: CHART_AXIS_TICK },
                        ticks: { color: CHART_AXIS_TICK },
                        grid: { color: CHART_GRID_SOFT },
                      },
                    },
                  }}
                />
              </div>
            </div>
          </div>

          <div className="bg-ink-900/90 border border-ink-700 rounded-2xl p-4 max-h-72 overflow-auto">
            <div className="text-xs font-bold text-silver-100 mb-2">
              Detected Fire Hotspots ({res.hotspots.length} shown)
            </div>
            <table className="w-full text-[11px] text-left">
              <thead>
                <tr className="text-silver-500 border-b border-ink-700">
                  <th className="py-1.5 pr-3">Event / Detection</th>
                  <th className="py-1.5 pr-3">Date / UTC</th>
                  <th className="py-1.5 pr-3">Coords (Lat, Lon)</th>
                  <th className="py-1.5 pr-3">FRP (MW)</th>
                  <th className="py-1.5 pr-3">Confidence</th>
                  <th className="py-1.5">Platform</th>
                </tr>
              </thead>
              <tbody>
                {res.hotspots.slice(0, 60).map((h, i) => (
                  <tr key={i} className="border-b border-ink-700/40 text-silver-300">
                    <td className="py-1.5 pr-3 font-sans truncate max-w-[210px]">{h.title || h.satellite}</td>
                    <td className="py-1.5 pr-3">{h.acq_date} {h.acq_time}</td>
                    <td className="py-1.5 pr-3">
                      {h.latitude.toFixed(2)}, {h.longitude.toFixed(2)}
                    </td>
                    <td className="py-1.5 pr-3 text-gold-300 font-bold">
                      {h.frp_mw !== null ? h.frp_mw.toFixed(1) : '—'}
                    </td>
                    <td className="py-1.5 pr-3">{h.confidence}</td>
                    <td className="py-1.5">{h.satellite}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
};
