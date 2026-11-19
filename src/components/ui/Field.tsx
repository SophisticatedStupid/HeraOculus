import React from 'react';

interface FieldProps {
  label: string;
  help?: string;
  children: React.ReactNode;
  className?: string;
}

export const Field: React.FC<FieldProps> = ({ label, help, children, className = '' }) => (
  <label className={`block space-y-1 ${className}`}>
    <span className="text-xs font-medium text-silver-500">{label}</span>
    {children}
    {help && <span className="block text-[10px] text-silver-500">{help}</span>}
  </label>
);

export default Field;
