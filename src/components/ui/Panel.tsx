import React from 'react';

interface PanelProps {
  title?: string;
  children: React.ReactNode;
  className?: string;
}

export const Panel: React.FC<PanelProps> = ({ title, children, className = '' }) => (
  <section
    className={`rounded-2xl border border-ink-700 bg-ink-900 p-5 ${className}`}
  >
    {title && (
      <h3 className="mb-3 font-display text-base font-semibold text-silver-100">{title}</h3>
    )}
    {children}
  </section>
);

export default Panel;
