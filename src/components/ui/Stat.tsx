import React from 'react';

interface StatProps {
  label: string;
  value: string | number;
  className?: string;
}

export const Stat: React.FC<StatProps> = ({ label, value, className = '' }) => (
  <div className={`space-y-0.5 ${className}`}>
    <span className="block text-[10px] font-medium text-silver-500">{label}</span>
    <span className="block tabular-nums text-sm font-semibold text-silver-100">{value}</span>
  </div>
);

export default Stat;
