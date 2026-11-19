import React from 'react';
import { DiagnosticResults } from '../../types/earthScience';
import { AlertCircle, CheckCircle, ShieldAlert, BarChart, Info } from 'lucide-react';

interface DiagnosticsPanelProps {
  diagnostics: DiagnosticResults;
  variableName: string;
  outlierMethod?: 'iqr' | 'zscore' | 'grubbs';
  onOutlierMethodChange?: (method: 'iqr' | 'zscore' | 'grubbs') => void;
}

export const DiagnosticsPanel: React.FC<DiagnosticsPanelProps> = ({
  diagnostics,
  variableName,
  outlierMethod = 'iqr',
  onOutlierMethodChange,
}) => {
  const { normalityTest, outliers, missingnessProfile, autocorrelation, heteroskedasticity, vif } = diagnostics;

  return (
    <div className="space-y-6 bg-ink-900 border border-ink-700 rounded-2xl p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-700 pb-3">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="w-5 h-5 text-gold-300" />
          <h3 className="text-sm font-bold text-silver-100">
            Statistical Assumptions & Diagnostic Suite ({variableName})
          </h3>
        </div>
        {onOutlierMethodChange ? (
          <div className="flex items-center space-x-2 text-xs">
            <span className="text-silver-500">Outlier Criterion:</span>
            <select
              value={outlierMethod}
              onChange={e => onOutlierMethodChange(e.target.value as 'iqr' | 'zscore' | 'grubbs')}
              className="bg-ink-950 border border-ink-700 rounded-lg px-2 py-1 text-xs text-gold-400"
            >
              <option value="iqr">1.5x IQR Fence</option>
              <option value="zscore">Z-Score (|z| &gt; 3)</option>
              <option value="grubbs">Grubbs Maximum Deviate</option>
            </select>
          </div>
        ) : (
          <span className="text-xs text-silver-500">
            Rigorous Scientific Validation
          </span>
        )}
      </div>

      {/* 4 Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* 1. Normality */}
        <div className="bg-ink-900/80 border border-ink-700 rounded-xl p-3.5 space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-silver-500">
              Normality
            </span>
            {normalityTest.isNormal ? (
              <CheckCircle className="w-4 h-4 text-silver-500" />
            ) : (
              <AlertCircle className="w-4 h-4 text-gold-300" />
            )}
          </div>
          <div className="text-sm font-bold text-silver-100">
            {normalityTest.isNormal ? 'Gaussian Normal' : 'Non-Normal'}
          </div>
          <div className="text-[10px] text-silver-500">
            p = {normalityTest.pValue.toFixed(4)} ({normalityTest.method})
          </div>
        </div>

        {/* 2. Outliers */}
        <div className="bg-ink-900/80 border border-ink-700 rounded-xl p-3.5 space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-silver-500">
              Outliers
            </span>
            {outliers.length === 0 ? (
              <CheckCircle className="w-4 h-4 text-silver-500" />
            ) : (
              <AlertCircle className="w-4 h-4 text-gold-400" />
            )}
          </div>
          <div className="text-sm font-bold text-silver-100">
            {outliers.length} Outliers Detected
          </div>
          <div className="text-[10px] text-silver-500">
            Method: {outliers[0]?.method || '1.5x IQR Rule'}
          </div>
        </div>

        {/* 3. Autocorrelation */}
        <div className="bg-ink-900/80 border border-ink-700 rounded-xl p-3.5 space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-silver-500">
              Autocorrelation (DW &amp; LB)
            </span>
            <BarChart className="w-4 h-4 text-gold-400" />
          </div>
          <div className="text-sm font-bold text-silver-100">
            DW: {autocorrelation.durbinWatson.toFixed(2)} | r₁: {autocorrelation.lag1.toFixed(2)}
          </div>
          <div className="text-[10px] text-silver-500">
            Ljung-Box Q = {autocorrelation.ljungBoxQ.toFixed(2)} (p = {autocorrelation.ljungBoxP.toFixed(4)})
          </div>
        </div>

        {/* 4. Missing Data */}
        <div className="bg-ink-900/80 border border-ink-700 rounded-xl p-3.5 space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-silver-500">
              Data Completeness
            </span>
            {missingnessProfile.missing === 0 ? (
              <CheckCircle className="w-4 h-4 text-silver-500" />
            ) : (
              <AlertCircle className="w-4 h-4 text-gold-300" />
            )}
          </div>
          <div className="text-sm font-bold text-silver-100">
            {missingnessProfile.missing === 0 ? '100% Complete' : `${missingnessProfile.percent.toFixed(1)}% Missing`}
          </div>
          <div className="text-[10px] text-silver-500">
            {missingnessProfile.missing} missing of {missingnessProfile.total}
          </div>
        </div>
      </div>

      {/* Outlier Table if Outliers Detected */}
      {outliers.length > 0 && (
        <div className="space-y-2 bg-ink-900/60 border border-ink-700 rounded-xl p-4">
          <h4 className="text-xs font-bold text-gold-300">
            Detected Anomalies &amp; Outliers ({outliers.length} total • showing up to 12)
          </h4>
          <div className="flex flex-wrap gap-2">
            {outliers.slice(0, 12).map((o, idx) => (
              <span
                key={idx}
                className="px-2.5 py-1 rounded-lg bg-ink-950 border border-gold-400/40/30 text-[11px] text-silver-100"
              >
                row #{o.index}: <strong className="text-gold-300">{o.value.toFixed(3)}</strong>
                {o.zScore !== undefined ? ` (${o.zScore.toFixed(2)})` : ''}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Multicollinearity VIF Table if present */}
      {vif && vif.length > 0 && (
        <div className="space-y-2 bg-ink-900/60 border border-ink-700 rounded-xl p-4">
          <h4 className="text-xs font-bold text-silver-100 flex items-center space-x-1.5">
            <Info className="w-3.5 h-3.5 text-gold-400" />
            <span>Multicollinearity Analysis (Variance Inflation Factor - VIF)</span>
          </h4>
          <p className="text-[11px] text-silver-500">
            VIF &gt; 5 indicates moderate collinearity; VIF &gt; 10 signals severe multicollinearity requiring predictor removal.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
            {vif.map((item, idx) => (
              <div
                key={idx}
                className="bg-ink-950/80 border border-ink-700 rounded-lg p-2.5 text-xs"
              >
                <div className="text-silver-500 truncate">{item.variable}</div>
                <div className={`text-sm font-bold ${
                  item.vif > 10 ? 'text-gold-400' : item.vif > 5 ? 'text-gold-300' : 'text-silver-500'
                }`}>
                  VIF = {item.vif.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Heteroskedasticity if present */}
      {heteroskedasticity && (
        <div className="bg-ink-900/60 border border-ink-700 rounded-xl p-4 flex items-center justify-between text-xs">
          <div>
            <div className="font-semibold text-silver-100">
              Breusch-Pagan Heteroskedasticity Test
            </div>
            <p className="text-silver-500 text-[11px] mt-0.5">
              Tests whether residual variance depends systematically on predictor magnitudes.
            </p>
          </div>
          <div className="text-right">
            <span className={`px-2 py-1 rounded text-xs font-bold ${
              heteroskedasticity.isHomoskedastic
                ? 'bg-gold-400/10 text-silver-300 border border-gold-400/40/30'
                : 'bg-gold-400/10 text-gold-300 border border-gold-400/40/30'
            }`}>
              {heteroskedasticity.isHomoskedastic ? 'Homoskedastic (Equal Variance)' : 'Heteroskedastic'}
            </span>
            <div className="text-[10px] text-silver-300 mt-1">
              p = {heteroskedasticity.bpPValue.toFixed(4)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
