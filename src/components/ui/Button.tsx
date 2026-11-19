import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'quiet';
  children: React.ReactNode;
}

const variants: Record<NonNullable<ButtonProps['variant']>, string> = {
  primary: 'bg-gold-400 text-ink-950 hover:bg-gold-400/90 font-semibold',
  secondary: 'border border-ink-700 bg-ink-900 text-silver-100 hover:bg-ink-800',
  quiet: 'text-silver-500 hover:text-silver-100 hover:bg-ink-800/60',
};

export const Button: React.FC<ButtonProps> = ({ variant = 'secondary', className = '', children, ...rest }) => (
  <button
    {...rest}
    className={`inline-flex items-center gap-2 rounded-xl px-3.5 py-2 text-xs transition-colors focus-visible:outline-none disabled:opacity-50 disabled:cursor-not-allowed ${variants[variant]} ${className}`}
  >
    {children}
  </button>
);

export default Button;
