import React, { useState } from 'react';
import { TrendAnalysisResult } from '../types/earthScience';
import { HelpCircle, X, ShieldCheck } from 'lucide-react';
import { significanceAccent, tokens } from '../theme/tokens';

interface SignificanceBadgeProps {
  stats: TrendAnalysisResult;
  /** Compact inline variant vs full gold-400 gauge */
  variant?: 'gauge' | 'compact';
}

function levelToFill(level: string): number {
  switch (level) {
    case 'Extremely Significant':
      return 1;
    case 'Highly Significant':
      return 0.82;
    case 'Significant':
      return 0.65;
    case 'Marginally Significant':
      return 0.38;
    default:
      return 0.12;
  }
}

function sentenceLevel(level: string): string {
  return level;
}

export const SignificanceBadge: React.FC<SignificanceBadgeProps> = ({
  stats,
  variant = 'gauge',
}) => {
  const [showExplainer, setShowExplainer] = useState(false);
  const fill = levelToFill(stats.significanceLevel);
  const isSignificant =
    stats.significanceLevel !== 'Not Significant' &&
    stats.significanceLevel !== 'Marginally Significant';
  const accent = significanceAccent(stats.significanceLevel);
  const ringColor = isSignificant ? tokens.gold400 : tokens.silver600;

  const size = 88;
  const stroke = 7;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const dashOffset = c * (1 - fill);

  const pLabel =
    stats.mkPValue < 0.001 ? 'p < 0.001' : `p = ${stats.mkPValue.toFixed(3)}`;

  const openExplainer = () => setShowExplainer(true);

  if (variant === 'compact') {
    return (
      <>
        <button
          type="button"
          onClick={openExplainer}
          className={`inline-flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-left transition-colors focus-visible:outline-none ${
            isSignificant
              ? 'border-gold-400/40 bg-ink-900 text-gold-400'
              : 'border-ink-700 bg-ink-900 text-silver-600'
          }`}
          title="How significance is calculated"
        >
          <span className="tabular-nums text-sm font-semibold">{stats.significanceStars || 'ns'}</span>
          <span className="text-xs text-silver-300">{sentenceLevel(stats.significanceLevel)}</span>
          <span className="border-l border-ink-700 pl-2 tabular-nums text-xs text-silver-500">
            {pLabel}
          </span>
          <HelpCircle className="h-3.5 w-3.5 opacity-60" />
        </button>
        {showExplainer && (
          <ExplainerModal onClose={() => setShowExplainer(false)} />
        )}
      </>
    );
  }

  return (
    <>
      <div
        className={`flex flex-wrap items-center gap-4 ${
          isSignificant ? 'iris-gauge-significant' : ''
        }`}
      >
        <button
          type="button"
          onClick={openExplainer}
          className="group relative flex items-center gap-4 rounded-2xl border border-ink-700 bg-ink-900 p-3 text-left transition-colors hover:border-gold-400/40 focus-visible:outline-none"
          title="How significance is calculated"
          aria-label={`Significance: ${stats.significanceLevel}, ${pLabel}`}
        >
          <div className="relative shrink-0" style={{ width: size, height: size }}>
            <svg width={size} height={size} className="-rotate-90" aria-hidden>
              <circle
                cx={size / 2}
                cy={size / 2}
                r={r}
                fill="none"
                stroke={tokens.ink700}
                strokeWidth={stroke}
              />
              <circle
                className="iris-gauge-ring iris-gauge-glow"
                cx={size / 2}
                cy={size / 2}
                r={r}
                fill="none"
                stroke={ringColor}
                strokeWidth={stroke}
                strokeLinecap="round"
                strokeDasharray={c}
                strokeDashoffset={dashOffset}
                opacity={isSignificant ? 1 : 0.55}
              />
              {/* Inner gold-400 pupil */}
              <circle
                cx={size / 2}
                cy={size / 2}
                r={r * 0.38}
                fill="none"
                stroke={isSignificant ? tokens.gold400 : tokens.silver600}
                strokeWidth={2}
                opacity={0.7}
                className="iris-gauge-glow"
              />
              <circle
                cx={size / 2}
                cy={size / 2}
                r={r * 0.14}
                fill={accent}
              />
            </svg>
            <span className="absolute inset-0 flex items-center justify-center pt-0.5">
              <span
                className={`font-display text-lg font-semibold tabular-nums ${
                  isSignificant ? 'text-gold-400' : 'text-silver-600'
                }`}
              >
                {stats.significanceStars || 'ns'}
              </span>
            </span>
          </div>

          <div className="min-w-0 pr-1">
            <p className="font-display text-base font-semibold text-silver-100">
              {sentenceLevel(stats.significanceLevel)}
            </p>
            <p className="mt-0.5 tabular-nums text-sm text-silver-300">{pLabel}</p>
            <p className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-silver-500">
              <span className="tabular-nums">
                MK Z{' '}
                <strong className="font-medium text-silver-300">
                  {stats.mkZ > 0 ? '+' : ''}
                  {stats.mkZ}
                </strong>
              </span>
              <span className="tabular-nums">
                τ{' '}
                <strong className="font-medium text-silver-300">
                  {stats.mkTau > 0 ? '+' : ''}
                  {stats.mkTau}
                </strong>
              </span>
            </p>
            <span className="mt-2 inline-flex items-center gap-1 text-[11px] text-silver-500 group-hover:text-gold-400">
              <HelpCircle className="h-3 w-3" />
              Why this matters
            </span>
          </div>
        </button>
      </div>

      {showExplainer && <ExplainerModal onClose={() => setShowExplainer(false)} />}
    </>
  );
};

function ExplainerModal({ onClose }: { onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/80 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="sig-explainer-title"
    >
      <div className="relative w-full max-w-lg rounded-2xl border border-ink-700 bg-ink-900 p-6 text-silver-100 shadow-xl">
        <button
          type="button"
          onClick={onClose}
          className="absolute right-4 top-4 rounded-lg p-1.5 text-silver-500 transition-colors hover:bg-ink-800 hover:text-silver-100 focus-visible:outline-none"
          aria-label="Close"
        >
          <X className="h-5 w-5" />
        </button>

        <div className="mb-4 flex items-center gap-3">
          <div className="rounded-xl border border-gold-400/30 bg-ink-800 p-2.5 text-gold-400">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div>
            <h3 id="sig-explainer-title" className="font-display text-lg font-semibold text-silver-100">
              Statistical significance in Earth science
            </h3>
            <p className="text-xs text-silver-500">
              How scientists tell real change from weather noise
            </p>
          </div>
        </div>

        <div className="space-y-3.5 text-xs leading-relaxed text-silver-300">
          <div className="rounded-xl border border-ink-700 bg-ink-950/80 p-3">
            <span className="mb-1 block font-semibold text-silver-100">1. The null hypothesis (H₀)</span>
            <p>
              Scientists start by assuming there is no real climate trend — any rise or fall is
              accidental weather noise.
            </p>
          </div>

          <div className="rounded-xl border border-ink-700 bg-ink-950/80 p-3">
            <span className="mb-1 block font-semibold text-silver-100">2. The p-value</span>
            <p>
              The p-value asks: what is the chance of seeing this pattern by coincidence?
              <br />
              • <strong className="text-silver-100">p &lt; 0.05 (*)</strong>: standard threshold
              <br />
              • <strong className="text-silver-100">p &lt; 0.01 (**)</strong>: high confidence
              <br />
              • <strong className="text-silver-100">p &lt; 0.001 (***)</strong>: extreme certainty
            </p>
          </div>

          <div className="rounded-xl border border-ink-700 bg-ink-950/80 p-3">
            <span className="mb-1 block font-semibold text-silver-100">3. Why Mann-Kendall?</span>
            <p>
              Linear regression can be skewed by a single storm or eruption. Mann-Kendall only
              checks whether values consistently rank higher or lower over time, so it resists
              outliers and non-normal climate distributions.
            </p>
          </div>
        </div>

        <div className="mt-5 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl bg-gold-400 px-4 py-2 text-xs font-semibold text-ink-950 transition-colors hover:bg-gold-400/90 focus-visible:outline-none"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
}
