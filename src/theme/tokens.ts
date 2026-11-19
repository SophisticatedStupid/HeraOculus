export const tokens = {
  gold300: '#F0D58A',
  gold400: '#D4AF37',
  gold500: '#B8922A',
  gold700: '#7A5F1A',
  silver100: '#F2F2F2',
  silver300: '#C0C0C0',
  silver500: '#8A8A8A',
  silver600: '#5E5E5E',
  ink950: '#000000',
  ink900: '#0A0A0A',
  ink800: '#141414',
  ink700: '#1E1E1E',
  line: '#2E2E2E',
} as const;

export type TokenName = keyof typeof tokens;

export const chartSeries = [
  tokens.gold400,
  tokens.silver300,
  tokens.silver500,
  tokens.gold300,
  tokens.silver600,
] as const;

export function trendColor(direction: 'Increasing' | 'Decreasing' | 'No Trend' | string): string {
  // Direction is never shown by hue; use gold for significant signal.
  return tokens.gold400;
}

export function significanceAccent(level: string): string {
  if (level === 'Not Significant' || level === 'Marginally Significant') return tokens.silver600;
  return tokens.gold400;
}
