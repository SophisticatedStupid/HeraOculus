import React, { useEffect, useRef, useState } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface IrisGaugeProps {
  pValue: number;
  direction: 'increasing' | 'decreasing' | 'none';
  size?: number;
}

function openness(p: number): number {
  if (p >= 0.1) return 0.15;
  if (p >= 0.05) return 0.35;
  if (p >= 0.01) return 0.62;
  if (p >= 0.001) return 0.82;
  return 1;
}

function isSignificant(p: number): boolean {
  return p < 0.05;
}

function labelFor(p: number, direction: IrisGaugeProps['direction']): string {
  const sig = p < 0.001 ? 'extremely significant' : p < 0.01 ? 'highly significant' : p < 0.05 ? 'significant' : 'not significant';
  const dir = direction === 'increasing' ? 'increasing' : direction === 'decreasing' ? 'decreasing' : 'no';
  return `Statistically ${sig} ${dir} trend, p = ${p < 0.001 ? '<0.001' : p.toFixed(3)}`;
}

export const IrisGauge: React.FC<IrisGaugeProps> = ({ pValue, direction, size = 96 }) => {
  const [animated, setAnimated] = useState(false);
  const prevRef = useRef(pValue);
  const open = openness(pValue);
  const sig = isSignificant(pValue);

  useEffect(() => {
    if (prevRef.current !== pValue) {
      setAnimated(true);
      const t = setTimeout(() => setAnimated(false), 300);
      prevRef.current = pValue;
      return () => clearTimeout(t);
    }
  }, [pValue]);

  const cx = size / 2;
  const cy = size / 2;
  const outerR = size / 2 - 2;
  const innerR = outerR * (1 - open * 0.72);
  const irisR = outerR * 0.3;

  const strokeColor = sig ? '#D4AF37' : '#5E5E5E';
  const innerColor = sig ? '#F0D58A' : '#5E5E5E';

  const pLabel = pValue < 0.001 ? '<0.001' : pValue.toFixed(3);

  const Arrow = direction === 'increasing' ? TrendingUp : direction === 'decreasing' ? TrendingDown : Minus;

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label={labelFor(pValue, direction)}
      className="shrink-0"
    >
      <circle cx={cx} cy={cy} r={outerR} fill="none" stroke="#1E1E1E" strokeWidth={2} />
      <circle
        cx={cx}
        cy={cy}
        r={innerR}
        fill="none"
        stroke={strokeColor}
        strokeWidth={2.5}
        style={{ transition: animated ? 'r 300ms ease-out' : 'none' }}
      />
      <circle cx={cx} cy={cy} r={irisR} fill={innerColor} opacity={sig ? 0.85 : 0.5} />
      <foreignObject x={cx - 8} y={cy - size * 0.18} width="16" height="16">
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: strokeColor,
          }}
        >
          <Arrow size={14} />
        </div>
      </foreignObject>
      <text
        x={cx}
        y={cy + size * 0.22}
        textAnchor="middle"
        fontSize={size * 0.16}
        fontWeight={600}
        fill="#F2F2F2"
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {pLabel}
      </text>
    </svg>
  );
};

export default IrisGauge;
