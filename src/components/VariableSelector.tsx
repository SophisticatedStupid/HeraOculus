import React, { useState } from 'react';
import { EarthVariable, VariableId } from '../types/earthScience';
import { EARTH_VARIABLES } from '../services/nasaData';
import {
  Thermometer,
  CloudRain,
  Trees,
  Sprout,
  Droplets,
  Wind,
  Sun,
  ShieldAlert,
  Satellite,
  Cpu,
  Sparkles,
  Gauge,
  Flame,
  Snowflake,
  Waves,
  Activity,
  Layers,
  Check,
} from 'lucide-react';

interface VariableSelectorProps {
  selectedVariableId: VariableId;
  onSelectVariable: (variable: EarthVariable) => void;
  recommendedVariableId?: VariableId;
}

export const VariableSelector: React.FC<VariableSelectorProps> = ({
  selectedVariableId,
  onSelectVariable,
  recommendedVariableId,
}) => {
  const [activeCategory, setActiveCategory] = useState<string>('All');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const categories = [
    { label: 'All', count: EARTH_VARIABLES.length },
    { label: 'Atmosphere', count: EARTH_VARIABLES.filter((v) => v.category === 'Atmosphere').length },
    { label: 'Land', count: EARTH_VARIABLES.filter((v) => v.category === 'Land').length },
    { label: 'Hydrosphere', count: EARTH_VARIABLES.filter((v) => v.category === 'Hydrosphere').length },
    { label: 'Biosphere', count: EARTH_VARIABLES.filter((v) => v.category === 'Biosphere').length },
    { label: 'Cryosphere', count: EARTH_VARIABLES.filter((v) => v.category === 'Cryosphere').length },
    { label: 'Ocean', count: EARTH_VARIABLES.filter((v) => v.category === 'Ocean').length },
  ];

  const filteredVariables = EARTH_VARIABLES.filter((v) => {
    const matchesCategory = activeCategory === 'All' || v.category === activeCategory;
    const matchesSearch =
      v.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      v.shortName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      v.missions.some((m) => m.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesCategory && matchesSearch;
  });

  const getVariableIcon = (id: VariableId) => {
    switch (id) {
      case 'temp_air':
        return <Thermometer className="w-4 h-4 text-gold-400" />;
      case 'precip':
        return <CloudRain className="w-4 h-4 text-silver-500" />;
      case 'ndvi':
        return <Trees className="w-4 h-4 text-silver-300" />;
      case 'soil_moist':
        return <Sprout className="w-4 h-4 text-gold-300" />;
      case 'evapo':
        return <Droplets className="w-4 h-4 text-silver-500" />;
      case 'aod':
        return <Wind className="w-4 h-4 text-silver-600" />;
      case 'lst_day':
        return <Sun className="w-4 h-4 text-gold-400" />;
      case 'ozone':
        return <ShieldAlert className="w-4 h-4 text-gold-400" />;
      case 'pressure':
        return <Gauge className="w-4 h-4 text-silver-300" />;
      case 'humidity':
        return <Droplets className="w-4 h-4 text-silver-300" />;
      case 'specific_humidity':
        return <CloudRain className="w-4 h-4 text-silver-500" />;
      case 'wind_speed':
        return <Wind className="w-4 h-4 text-silver-300" />;
      case 'radiation_sw':
        return <Sun className="w-4 h-4 text-gold-300" />;
      case 'radiation_lw':
        return <Flame className="w-4 h-4 text-gold-400" />;
      case 'albedo':
        return <Sparkles className="w-4 h-4 text-silver-300" />;
      case 'snow_depth':
        return <Snowflake className="w-4 h-4 text-silver-300" />;
      case 'snow_precip':
        return <CloudRain className="w-4 h-4 text-silver-500" />;
      case 'ozone_total':
        return <ShieldAlert className="w-4 h-4 text-gold-400" />;
      case 'soil_moist_top':
        return <Sprout className="w-4 h-4 text-gold-300" />;
      case 'soil_moist_prof':
        return <Layers className="w-4 h-4 text-gold-300" />;
      case 'sea_level':
        return <Waves className="w-4 h-4 text-silver-500" />;
      case 'sst':
        return <Thermometer className="w-4 h-4 text-gold-400" />;
      case 'streamflow':
        return <Activity className="w-4 h-4 text-silver-500" />;
      case 'evi':
        return <Trees className="w-4 h-4 text-silver-300" />;
      default:
        return <Satellite className="w-4 h-4 text-silver-600" />;
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="flex items-center space-x-1.5 overflow-x-auto w-full sm:w-auto p-1 bg-ink-900 rounded-xl border border-ink-700">
          {categories.map((cat) => (
            <button
              key={cat.label}
              type="button"
              onClick={() => setActiveCategory(cat.label)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors whitespace-nowrap flex items-center space-x-1.5 ${
                activeCategory === cat.label
                  ? 'bg-gold-400 text-ink-950'
                  : 'text-silver-500 hover:text-silver-100 hover:bg-ink-800/60'
              }`}
            >
              <span>{cat.label}</span>
              <span
                className={`text-[10px] px-1.5 rounded-full tabular-nums ${
                  activeCategory === cat.label
                    ? 'bg-ink-950/20 text-ink-950'
                    : 'bg-ink-800 text-silver-500'
                }`}
              >
                {cat.count}
              </span>
            </button>
          ))}
        </div>

        <input
          type="text"
          placeholder="Filter by variable or mission…"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full sm:w-64 px-3 py-1.5 bg-ink-900 border border-ink-700 rounded-xl text-xs text-silver-100 placeholder-silver-500 focus:outline-none focus:ring-1 focus:ring-gold-400 focus:border-gold-400 transition-colors"
        />
      </div>

      <div className="flex items-center justify-between px-1 text-xs text-silver-500">
        <span>
          Showing{' '}
          <strong className="text-gold-400 tabular-nums">{filteredVariables.length}</strong> of{' '}
          <strong className="tabular-nums">{EARTH_VARIABLES.length}</strong> Earth system variables
        </span>
        {activeCategory !== 'All' && (
          <button
            type="button"
            onClick={() => setActiveCategory('All')}
            className="text-[11px] text-gold-400 hover:underline"
          >
            Show all 24 variables
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {filteredVariables.map((variable) => {
          const isSelected = selectedVariableId === variable.id;
          const isRecommended = recommendedVariableId === variable.id;

          return (
            <div
              key={variable.id}
              role="button"
              tabIndex={0}
              onClick={() => onSelectVariable(variable)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelectVariable(variable);
                }
              }}
              className={`relative group p-3.5 rounded-xl border cursor-pointer transition-colors duration-200 flex flex-col justify-between ${
                isSelected
                  ? 'bg-ink-900 border-gold-400 ring-1 ring-gold-400/50'
                  : 'bg-ink-950 hover:bg-ink-900 border-ink-700 hover:border-ink-700'
              }`}
            >
              {isSelected && (
                <div className="absolute -top-2.5 left-2 flex items-center space-x-1 px-2 py-0.5 rounded-full bg-gold-400 text-ink-950 text-[10px] font-semibold ring-2 ring-ink-950 z-10">
                  <Check className="w-2.5 h-2.5 stroke-[3]" />
                  <span>Active</span>
                </div>
              )}

              {isRecommended && (
                <div className="absolute -top-2 right-2 flex items-center space-x-1 px-2 py-0.5 rounded-full bg-gold-400 text-[10px] font-semibold text-ink-950 z-10">
                  <Sparkles className="w-2.5 h-2.5" />
                  <span>Key indicator</span>
                </div>
              )}

              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <div className="p-1.5 rounded-lg bg-ink-900 border border-ink-700">
                      {getVariableIcon(variable.id)}
                    </div>
                    <span className="text-[11px] font-medium text-silver-500">{variable.category}</span>
                  </div>
                  <span className="text-[11px] tabular-nums px-1.5 py-0.5 rounded bg-ink-800 text-gold-400 border border-ink-700">
                    {variable.unit}
                  </span>
                </div>

                <h4 className="text-sm font-semibold text-silver-100 group-hover:text-gold-400 transition-colors">
                  {variable.shortName}
                </h4>

                <p className="text-xs text-silver-500 mt-1 leading-relaxed">
                  {variable.studentExplanation}
                </p>
              </div>

              <div className="mt-3 pt-2.5 border-t border-ink-700">
                <div className="flex items-start space-x-1.5 text-[11px] text-silver-500">
                  {variable.measurementType === 'Satellite Instrument' ? (
                    <Satellite className="w-3.5 h-3.5 text-gold-400 shrink-0 mt-0.5" />
                  ) : (
                    <Cpu className="w-3.5 h-3.5 text-gold-400 shrink-0 mt-0.5" />
                  )}
                  <span>
                    <span className="text-silver-300 font-semibold">
                      {variable.measurementType === 'Satellite Instrument' ? 'Mission:' : 'Model:'}
                    </span>{' '}
                    {variable.missions[0]}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
