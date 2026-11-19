import React from 'react';

interface TabsProps {
  tabs: { id: string; label: string }[];
  active: string;
  onChange: (id: string) => void;
}

export const Tabs: React.FC<TabsProps> = ({ tabs, active, onChange }) => (
  <div className="flex flex-wrap gap-1 rounded-xl border border-ink-700 bg-ink-900 p-1">
    {tabs.map((t) => (
      <button
        key={t.id}
        type="button"
        onClick={() => onChange(t.id)}
        className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none ${
          active === t.id
            ? 'bg-gold-400 text-ink-950'
            : 'text-silver-500 hover:text-silver-100 hover:bg-ink-800/60'
        }`}
      >
        {t.label}
      </button>
    ))}
  </div>
);

export default Tabs;
