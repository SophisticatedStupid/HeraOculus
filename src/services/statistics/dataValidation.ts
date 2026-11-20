import { TimeSeriesPoint, VariableId } from '../../types/earthScience';

export type DataState = 'loading' | 'ready' | 'unavailable' | 'error' | 'empty';

export interface DataValidationResult {
  state: DataState;
  validPoints: TimeSeriesPoint[];
  totalPoints: number;
  validCount: number;
  missingCount: number;
  messages: string[];
  minRequired: number;
  method: string;
}

function countValid(points: TimeSeriesPoint[], useAnomaly: boolean): number {
  let count = 0;
  for (const p of points) {
    const raw = p.raw;
    if (raw !== null && isFinite(raw)) {
      if (useAnomaly && p.anomaly !== undefined && !isFinite(p.anomaly)) continue;
      count++;
    }
  }
  return count;
}

function validateTimestamps(points: TimeSeriesPoint[]): string[] {
  const messages: string[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (!/^\d{4}-\d{2}$/.test(p.date)) {
      messages.push(`Invalid date format at index ${i}: "${p.date}" (expected YYYY-MM)`);
    }
    if (seen.has(p.date)) {
      messages.push(`Duplicate timestamp: ${p.date}`);
    }
    seen.add(p.date);
  }
  return messages;
}

function validateValues(points: TimeSeriesPoint[]): string[] {
  const messages: string[] = [];
  let nullCount = 0;
  let nanCount = 0;
  let infCount = 0;
  for (const p of points) {
    if (p.raw === null) {
      nullCount++;
    } else if (!isFinite(p.raw)) {
      if (isNaN(p.raw)) nanCount++;
      else infCount++;
    }
  }
  if (nullCount > 0) messages.push(`${nullCount} null observations`);
  if (nanCount > 0) messages.push(`${nanCount} NaN observations`);
  if (infCount > 0) messages.push(`${infCount} Infinity observations`);
  return messages;
}

function checkMonotonicTimestamps(points: TimeSeriesPoint[]): string[] {
  const messages: string[] = [];
  for (let i = 1; i < points.length; i++) {
    if (points[i].date <= points[i - 1].date) {
      messages.push(`Non-monotonic timestamps: ${points[i - 1].date} >= ${points[i].date}`);
      break;
    }
  }
  return messages;
}

export function getMinRequiredObservations(method: string, useAnomaly = true): number {
  switch (method) {
    case 'mann-kendall':
      return useAnomaly ? 4 : 4;
    case 'theil-sen':
      return 2;
    case 'ols-regression':
      return 3;
    case 'seasonal-mk':
      return 12;
    case 'anova':
      return 3;
    default:
      return 3;
  }
}

export function validateTimeSeriesForAnalysis(
  points: TimeSeriesPoint[],
  variableId: VariableId,
  useAnomaly = true
): DataValidationResult {
  const method = 'trend-analysis';
  const minRequired = getMinRequiredObservations(method, useAnomaly);
  
  const timestampMessages = validateTimestamps(points);
  const valueMessages = validateValues(points);
  const monotonicMessages = checkMonotonicTimestamps(points);
  
  const allMessages = [...timestampMessages, ...valueMessages, ...monotonicMessages];
  
  const totalPoints = points.length;
  const validCount = countValid(points, useAnomaly);
  const missingCount = totalPoints - validCount;
  
  let state: DataState;
  if (totalPoints === 0) {
    state = 'empty';
  } else if (validCount < minRequired) {
    state = 'unavailable';
  } else if (allMessages.length > 0) {
    state = 'error';
  } else {
    state = 'ready';
  }
  
  const validPoints = points.filter(p => {
    const raw = p.raw;
    if (raw === null || !isFinite(raw)) return false;
    if (useAnomaly && p.anomaly !== undefined && !isFinite(p.anomaly)) return false;
    return true;
  });
  
  return {
    state,
    validPoints,
    totalPoints,
    validCount,
    missingCount,
    messages: allMessages,
    minRequired,
    method,
  };
}

export function validateTimeSeriesForSeasonalMK(
  points: TimeSeriesPoint[],
  seasonLength = 12
): DataValidationResult {
  const method = 'seasonal-mann-kendall';
  const minRequired = seasonLength * 2;
  
  const timestampMessages = validateTimestamps(points);
  const valueMessages = validateValues(points);
  
  const allMessages = [...timestampMessages, ...valueMessages];
  
  const totalPoints = points.length;
  const validCount = points.filter(p => p.raw !== null && isFinite(p.raw)).length;
  const missingCount = totalPoints - validCount;
  
  let state: DataState;
  if (totalPoints === 0) {
    state = 'empty';
  } else if (validCount < minRequired) {
    state = 'unavailable';
  } else if (allMessages.length > 0) {
    state = 'error';
  } else {
    state = 'ready';
  }
  
  const validPoints = points.filter(p => p.raw !== null && isFinite(p.raw));
  
  return {
    state,
    validPoints,
    totalPoints,
    validCount,
    missingCount,
    messages: allMessages,
    minRequired,
    method,
  };
}

export function validateTimeSeriesForTheilSen(
  points: TimeSeriesPoint[],
  variableId: VariableId,
  useAnomaly = true
): DataValidationResult {
  return validateTimeSeriesForAnalysis(points, variableId, useAnomaly);
}

export function validateTimeSeriesForOLS(
  points: TimeSeriesPoint[],
  variableId: VariableId,
  useAnomaly = true
): DataValidationResult {
  return validateTimeSeriesForAnalysis(points, variableId, useAnomaly);
}

export function validateTimeSeriesForANOVA(
  points: TimeSeriesPoint[],
  groups: number
): DataValidationResult {
  const method = 'anova';
  const minRequired = groups * 2;
  
  const timestampMessages = validateTimestamps(points);
  const valueMessages = validateValues(points);
  
  const allMessages = [...timestampMessages, ...valueMessages];
  
  const totalPoints = points.length;
  const validCount = points.filter(p => p.raw !== null && isFinite(p.raw)).length;
  const missingCount = totalPoints - validCount;
  
  let state: DataState;
  if (totalPoints === 0) {
    state = 'empty';
  } else if (validCount < minRequired) {
    state = 'unavailable';
  } else if (allMessages.length > 0) {
    state = 'error';
  } else {
    state = 'ready';
  }
  
  const validPoints = points.filter(p => p.raw !== null && isFinite(p.raw));
  
  return {
    state,
    validPoints,
    totalPoints,
    validCount,
    missingCount,
    messages: allMessages,
    minRequired,
    method,
  };
}