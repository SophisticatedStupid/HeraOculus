/**
 * GEOIRIS Modular Statistics Engine
 * Exports all statistical modules and maintains 100% backward compatibility with original statistics.ts
 */

export * from './distributions';
export * from './descriptive';
export * from './inferential';
export * from './regression';
export * from './timeseries';
export * from './diagnostics';
export * from './fdr';
export * from './anova';
export * from './twoWayAnova';

// ── Backward compatibility exports for existing codebase ─────────────

import { TimeSeriesPoint, TrendAnalysisResult, VariableId } from '../../types/earthScience';
import { normalPValue, tCDF, tQuantile } from './distributions';
import { computeTieGroups, tieSign, MANN_KENDALL_TIE_TOLERANCE } from './timeseries';

/**
 * Calculate Deseasonalized Anomalies and 12-Month Moving Averages
 * Explicitly preserves null/missing (e.g. -999.0 polar night) time slots.
 */
export function processTimeSeries(rawPoints: { date: string; value: number | null }[]): TimeSeriesPoint[] {
  if (rawPoints.length === 0) return [];

  const points: TimeSeriesPoint[] = rawPoints.map(p => {
    const parts = p.date.split('-');
    const year = parseInt(parts[0], 10);
    const month = parseInt(parts[1], 10);
    const isValid = p.value !== null && typeof p.value === 'number' && !isNaN(p.value) && isFinite(p.value);
    return {
      date: p.date,
      year,
      month,
      raw: isValid ? p.value : null,
      isMissing: !isValid,
    };
  });

  const monthlySums: { [m: number]: { sum: number; count: number } } = {};
  for (let m = 1; m <= 12; m++) {
    monthlySums[m] = { sum: 0, count: 0 };
  }

  for (const p of points) {
    if (p.raw !== null && monthlySums[p.month]) {
      monthlySums[p.month].sum += p.raw;
      monthlySums[p.month].count += 1;
    }
  }

  const monthlyMeans: { [m: number]: number } = {};
  for (let m = 1; m <= 12; m++) {
    monthlyMeans[m] = monthlySums[m].count > 0 ? monthlySums[m].sum / monthlySums[m].count : 0;
  }

  for (const p of points) {
    if (p.raw === null) {
      p.anomaly = undefined;
      continue;
    }
    const baseline = monthlyMeans[p.month] || 0;
    p.anomaly = +(p.raw - baseline).toFixed(3);
  }

  const windowSize = 12;
  const halfWindow = Math.floor(windowSize / 2);

  for (let i = 0; i < points.length; i++) {
    if (points[i].raw === null) {
      points[i].movingAvg = undefined;
      continue;
    }
    const start = Math.max(0, i - halfWindow);
    const end = Math.min(points.length, i + halfWindow);
    let sum = 0;
    let count = 0;
    for (let j = start; j < end; j++) {
      const val = points[j].raw;
      if (val !== null) {
        sum += val;
        count++;
      }
    }
    points[i].movingAvg = count > 0 ? +(sum / count).toFixed(3) : undefined;
  }

  return points;
}

function getValidPointEntries(
  points: TimeSeriesPoint[],
  useAnomaly: boolean
): { t: number; y: number; raw: number }[] {
  const valid: { t: number; y: number; raw: number }[] = [];
  for (const p of points) {
    if (p.raw === null || !isFinite(p.raw)) continue;
    const val = useAnomaly && p.anomaly !== undefined ? p.anomaly : p.raw;
    if (!isFinite(val)) continue;
    valid.push({
      t: p.year + (p.month - 1) / 12,
      y: val,
      raw: p.raw,
    });
  }
  return valid;
}

/**
 * Compute Theil-Sen Robust Slope Estimator
 */
export function calculateTheilSenSlope(points: TimeSeriesPoint[], useAnomaly = true): number {
  const valid = getValidPointEntries(points, useAnomaly);
  const n = valid.length;
  if (n < 2) return 0;

  const slopes: number[] = [];

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dt = valid[j].t - valid[i].t;
      if (dt > 0.001) {
        slopes.push((valid[j].y - valid[i].y) / dt);
      }
    }
  }

  if (slopes.length === 0) return 0;
  slopes.sort((a, b) => a - b);
  const mid = Math.floor(slopes.length / 2);
  return slopes.length % 2 === 0 ? (slopes[mid - 1] + slopes[mid]) / 2 : slopes[mid];
}

/**
 * Perform Mann-Kendall Non-Parametric Trend Test
 *
 * Tie handling: a single canonical tie grouping (computeTieGroups) is used for
 * both the S statistic and the tie-corrected variance, so ties are defined
 * identically everywhere. No arbitrary rounding is applied to the measurements;
 * ties are determined by the relative tolerance (default MANN_KENDALL_TIE_TOLERANCE).
 *
 * @param points      Time series points.
 * @param useAnomaly  Whether to use anomaly values.
 * @param tieTolerance Optional relative tie tolerance (default MANN_KENDALL_TIE_TOLERANCE).
 */
export function calculateMannKendall(
  points: TimeSeriesPoint[],
  useAnomaly = true,
  tieTolerance: number = MANN_KENDALL_TIE_TOLERANCE
): {
  s: number;
  varS: number;
  z: number;
  tau: number;
  pValue: number;
} {
  const values = getValidPointEntries(points, useAnomaly).map(e => e.y);
  const n = values.length;
  if (n < 4) {
    return { s: 0, varS: 1, z: 0, tau: 0, pValue: 1 };
  }

  const { groupIds, groupSizes } = computeTieGroups(values, tieTolerance);

  let s = 0;
  for (let k = 0; k < n - 1; k++) {
    const gk = groupIds[k];
    for (let j = k + 1; j < n; j++) {
      s += tieSign(values[j], values[k], groupIds[j], gk);
    }
  }

  let tieTerm = 0;
  for (const size of groupSizes) {
    if (size > 1) {
      tieTerm += size * (size - 1) * (2 * size + 5);
    }
  }

  // Tie-corrected variance; clamped to be non-negative.
  const varS = Math.max(0, (n * (n - 1) * (2 * n + 5) - tieTerm) / 18);

  let z = 0;
  const sqrtVar = Math.sqrt(varS);
  if (s > 0 && sqrtVar > 0) {
    z = (s - 1) / sqrtVar; // continuity correction
  } else if (s < 0 && sqrtVar > 0) {
    z = (s + 1) / sqrtVar; // continuity correction
  }

  const tau = (2 * s) / (n * (n - 1));
  const pValue = normalPValue(z);

  return { s, varS, z, tau, pValue };
}

/**
 * Complete Statistical Trend Analysis
 */
export function analyzeTrend(
  points: TimeSeriesPoint[],
  variableId: VariableId,
  variableName: string,
  unit: string,
  regionName: string,
  coordinates: [number, number],
  useAnomaly = true
): TrendAnalysisResult {
  const totalSlots = points.length;
  if (totalSlots === 0) {
    throw new Error('Cannot analyze empty time series');
  }

  const valid = getValidPointEntries(points, useAnomaly);
  const n = valid.length;
  const missingCount = totalSlots - n;
  if (n < 3) {
    throw new Error('Insufficient valid observations in time series');
  }

  const times = valid.map(e => e.t);
  const values = valid.map(e => e.y);

  const meanT = times.reduce((a, b) => a + b, 0) / n;
  const meanY = values.reduce((a, b) => a + b, 0) / n;

  let ssXX = 0;
  let ssYY = 0;
  let ssXY = 0;

  for (let i = 0; i < n; i++) {
    const dt = times[i] - meanT;
    const dy = values[i] - meanY;
    ssXX += dt * dt;
    ssYY += dy * dy;
    ssXY += dt * dy;
  }

  const olsSlope = ssXX !== 0 ? ssXY / ssXX : 0;
  const olsIntercept = meanY - olsSlope * meanT;

  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    const pred = olsSlope * times[i] + olsIntercept;
    const res = values[i] - pred;
    ssRes += res * res;
  }

  const rSquared = ssYY !== 0 ? Math.max(0, 1 - ssRes / ssYY) : 0;
  const df = Math.max(1, n - 2);
  const sErrSq = ssRes / df;
  const standardError = Math.sqrt(sErrSq);
  const slopeSE = ssXX > 0 ? Math.sqrt(sErrSq / ssXX) : 0;

  const tStatistic = slopeSE > 0 ? olsSlope / slopeSE : 0;
  const tPValue = 2 * (1 - tCDF(Math.abs(tStatistic), df));

  const sensSlopePerYear = calculateTheilSenSlope(points, useAnomaly);
  const sensSlopePerDecade = sensSlopePerYear * 10;

  const mk = calculateMannKendall(points, useAnomaly);

  const rawTCrit = tQuantile(0.975, df);
  const tCritical = isFinite(rawTCrit) ? rawTCrit : 1.96;
  for (let i = 0; i < totalSlots; i++) {
    const t = points[i].year + (points[i].month - 1) / 12;
    const trendVal = olsSlope * t + olsIntercept;
    points[i].trend = +trendVal.toFixed(3);

    const marginOfError =
      ssXX > 0
        ? tCritical * standardError * Math.sqrt(1 / n + Math.pow(t - meanT, 2) / ssXX)
        : 0;
    points[i].ciUpper = +(trendVal + marginOfError).toFixed(3);
    points[i].ciLower = +(trendVal - marginOfError).toFixed(3);
  }

  const slopePerYear = olsSlope;
  const slopePerDecade = olsSlope * 10;
  const totalDurationYears = times[n - 1] - times[0];
  const totalChange = slopePerYear * totalDurationYears;

  const rawMean = valid.reduce((acc, p) => acc + p.raw, 0) / n;
  const percentChange = rawMean !== 0 ? (totalChange / Math.abs(rawMean)) * 100 : 0;

  let direction: 'Increasing' | 'Decreasing' | 'Stable' = 'Stable';
  if (Math.abs(olsSlope) > 0.0001 && mk.pValue < 0.1) {
    direction = olsSlope > 0 ? 'Increasing' : 'Decreasing';
  }

  const p = mk.pValue;
  let significanceLevel: 'Extremely Significant' | 'Highly Significant' | 'Significant' | 'Marginally Significant' | 'Not Significant';
  let significanceStars: '***' | '**' | '*' | '†' | 'n.s.';
  let significanceColor: string;

  if (p < 0.001) {
    significanceLevel = 'Extremely Significant';
    significanceStars = '***';
    significanceColor = '#10B981';
  } else if (p < 0.01) {
    significanceLevel = 'Highly Significant';
    significanceStars = '**';
    significanceColor = '#06B6D4';
  } else if (p < 0.05) {
    significanceLevel = 'Significant';
    significanceStars = '*';
    significanceColor = '#3B82F6';
  } else if (p < 0.10) {
    significanceLevel = 'Marginally Significant';
    significanceStars = '†';
    significanceColor = '#F59E0B';
  } else {
    significanceLevel = 'Not Significant';
    significanceStars = 'n.s.';
    significanceColor = '#94A3B8';
  }

  const dirText = direction === 'Increasing' ? 'an upward (increasing)' : direction === 'Decreasing' ? 'a downward (decreasing)' : 'a neutral (stable)';
  const rateFmt = Math.abs(slopePerDecade).toFixed(2);
  const pFmt = p < 0.001 ? '< 0.001' : `= ${p.toFixed(3)}`;

  let plainEnglishVerdict = '';
  if (significanceLevel === 'Extremely Significant' || significanceLevel === 'Highly Significant' || significanceLevel === 'Significant') {
    plainEnglishVerdict = `A statistically detectable monotonic ${dirText} trend was detected in ${variableName} for ${regionName}. Over the analysis period (${Math.round(totalDurationYears)} years), the estimated change is ${totalChange > 0 ? '+' : ''}${totalChange.toFixed(2)} ${unit} (rate of ${totalChange > 0 ? '+' : '-'}${rateFmt} ${unit}/decade, p ${pFmt}). The estimated slope has an approximate standard error of ${standardError.toFixed(3)} ${unit}/year; treat the magnitude as an estimate with uncertainty, not an exact value. Statistical significance indicates the observed pattern is unlikely under a null hypothesis of no trend; it does not establish causality or identify a specific physical mechanism. Natural variability cannot be ruled out as a partial contributor.`;
  } else if (significanceLevel === 'Marginally Significant') {
    plainEnglishVerdict = `A weak or emerging ${dirText} trend is visible (${totalChange > 0 ? '+' : '-'}${rateFmt} ${unit}/decade, p = ${p.toFixed(3)}), but it falls slightly short of conventional 95% certainty (p < 0.05). The evidence is suggestive but not conclusive; additional observational years would help distinguish a directional signal from natural decadal variability.`;
  } else {
    plainEnglishVerdict = `No statistically significant monotonic trend was detected in ${variableName} for ${regionName} (p = ${p.toFixed(3)}). This does not prove the absence of change; it indicates the data do not provide sufficient evidence to conclude a directional trend is present beyond random variation. Observed variations may reflect natural seasonal cycles and inter-annual oscillations, which cannot be ruled out.`;
  }

  const scientificImplication = generateScientificImplication(variableId, direction, significanceLevel, slopePerDecade, unit, regionName);

  return {
    variableId,
    variableName,
    unit,
    regionName,
    coordinates,
    startDate: points[0].date,
    endDate: points[n - 1].date,
    sampleCount: n,
    missingCount,
    slopePerYear: +slopePerYear.toFixed(4),
    slopePerDecade: +slopePerDecade.toFixed(3),
    sensSlopePerDecade: +sensSlopePerDecade.toFixed(3),
    totalChange: +totalChange.toFixed(3),
    percentChange: +percentChange.toFixed(2),
    direction,
    rSquared: +rSquared.toFixed(3),
    olsSlope: +olsSlope.toFixed(5),
    olsIntercept: +olsIntercept.toFixed(3),
    standardError: +standardError.toFixed(3),
    tStatistic: +tStatistic.toFixed(3),
    tPValue: +tPValue.toFixed(4),
    mkS: mk.s,
    mkVarS: +mk.varS.toFixed(1),
    mkZ: +mk.z.toFixed(3),
    mkTau: +mk.tau.toFixed(3),
    mkPValue: +mk.pValue.toFixed(4),
    significanceLevel,
    significanceStars,
    significanceColor,
    plainEnglishVerdict,
    scientificImplication,
  };
}

function generateScientificImplication(
  variableId: VariableId,
  direction: 'Increasing' | 'Decreasing' | 'Stable',
  significance: string,
  ratePerDecade: number,
  unit: string,
  regionName: string
): string {
  if (significance === 'Not Significant') {
    return `In ${regionName}, no statistically detectable directional trend was identified for this variable. This is not evidence of a static or unchanging system; it means the data do not show a clear monotonic pattern beyond natural variability. Physical processes may still be active, but the current record does not resolve a sustained directional change.`;
  }

  const magnitude = Math.abs(ratePerDecade).toFixed(2);
  const dirWord = direction === 'Increasing' ? 'an increasing' : 'a decreasing';

  switch (variableId) {
    case 'temp_air':
    case 'lst_day':
      return direction === 'Increasing'
        ? `The data show ${dirWord} thermal trend (~+${magnitude} ${unit}/decade) in ${regionName}. This is consistent with — but not proof of — regional warming associated with radiative forcing, land-surface feedback, or circulation shifts. Attributing this to any specific mechanism requires controlled attribution studies beyond a monotonic trend test.`
        : `The data show ${dirWord} thermal trend (~${magnitude} ${unit}/decade cooling) in ${regionName}. Possible contributors include regional radiative anomalies, cloud-cover feedbacks, or aerosol effects; establishing causation requires further analysis.`;

    case 'precip':
      return direction === 'Increasing'
        ? `An increasing precipitation trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}. This may be consistent with changes in regional moisture transport or circulation, but a monotonic trend alone does not identify the physical driver.`
        : `A decreasing precipitation trend (~${magnitude} ${unit}/decade) was detected in ${regionName}. This could relate to shifting moisture transport, circulation changes, or other factors; attributing the cause requires additional analysis.`;

    case 'ndvi':
      return direction === 'Decreasing'
        ? `A declining NDVI trend (~${magnitude}/decade) was detected in ${regionName}, consistent with reduced vegetation greenness. Possible explanations include land-cover change, water stress, or disturbance; a trend test alone does not distinguish among them.`
        : `An increasing NDVI trend (~+${magnitude}/decade) was detected in ${regionName}, consistent with increasing vegetation greenness. This may reflect land management, climate, or recovery from disturbance; attribution requires additional evidence.`;

    case 'soil_moist':
      return direction === 'Decreasing'
        ? `A decreasing soil-moisture trend (~${magnitude} ${unit}/decade) was detected in ${regionName}. This may indicate drying conditions, but could also reflect changes in measurement or land use; the trend alone does not establish the cause.`
        : `An increasing soil-moisture trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}, consistent with wetter conditions. Multiple physical pathways could produce this; the trend does not by itself identify the mechanism.`;

    case 'evapo':
      return direction === 'Decreasing'
        ? `A decreasing evapotranspiration trend (~${magnitude} ${unit}/decade) was detected in ${regionName}. This is consistent with reduced surface water availability or vegetation changes, but causation cannot be inferred from the trend alone.`
        : `An increasing evapotranspiration trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}. This may be consistent with increased evaporative demand or land-surface changes; the trend alone does not establish the cause.`;

    case 'aod':
      return direction === 'Increasing'
        ? `An increasing aerosol optical depth trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}. This may reflect changes in emissions, dust, or biomass burning, but a monotonic trend does not identify the source; source apportionment requires additional analysis.`
        : `A decreasing aerosol optical depth trend (~${magnitude} ${unit}/decade) was detected in ${regionName}, consistent with lower column aerosol loading. This may reflect emissions changes or meteorological flushing; the trend alone does not distinguish among these.`;

    case 'ozone':
      return direction === 'Decreasing'
        ? `A decreasing ozone trend (~${magnitude} ${unit}/decade) was detected in ${regionName}. This could relate to stratospheric or tropospheric chemistry changes; attributing it to a specific mechanism requires additional analysis.`
        : `An increasing ozone trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}. This may reflect changes in precursor emissions or transport; the trend alone does not establish the cause.`;

    case 'pressure':
      return direction === 'Decreasing'
        ? `A decreasing pressure trend (~${magnitude} ${unit}/decade) was detected in ${regionName}, consistent with large-scale circulation shifts. Possible contributors include changes in storm tracks or thermal structure; the trend alone does not identify the mechanism.`
        : `An increasing pressure trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}, consistent with circulation changes. Attribution to specific dynamics requires additional analysis beyond the monotonic trend.`;

    case 'humidity':
      return direction === 'Decreasing'
        ? `A decreasing relative-humidity trend (~${magnitude} ${unit}/decade) was detected in ${regionName}. This is consistent with increased vapor-pressure deficit, but the trend alone does not establish causation.`
        : `An increasing relative-humidity trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}. This may reflect changes in moisture supply or temperature; the trend alone does not identify the mechanism.`;

    case 'specific_humidity':
      return direction === 'Increasing'
        ? `An increasing specific-humidity trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}. This is consistent with thermodynamic moisture scaling under warming (Clausius-Clapeyron ~7% per °C), but a trend alone does not prove that warming is the cause.`
        : `A decreasing specific-humidity trend (~${magnitude} ${unit}/decade) was detected in ${regionName}. This could reflect changes in moisture transport or source regions; the trend alone does not establish the cause.`;

    case 'wind_speed':
      return direction === 'Increasing'
        ? `An increasing wind-speed trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}. This may be consistent with circulation changes; attribution to specific drivers requires additional analysis.`
        : `A decreasing wind-speed trend (~${magnitude} ${unit}/decade) was detected in ${regionName} (sometimes termed "stilling"). Possible contributors include circulation changes or surface roughness; the trend alone does not distinguish among them.`;

    case 'radiation_sw':
      return direction === 'Increasing'
        ? `An increasing surface shortwave radiation trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}, consistent with reduced cloud or aerosol attenuation. Possible explanations include cloud-cover or aerosol changes; the trend alone does not identify the cause.`
        : `A decreasing surface shortwave radiation trend (~${magnitude} ${unit}/decade) was detected in ${regionName}, consistent with increased attenuation. Possible explanations include cloud or aerosol changes; attribution requires additional analysis.`;

    case 'radiation_lw':
      return direction === 'Increasing'
        ? `An increasing downward longwave radiation trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}. This is consistent with — but not proof of — changes in greenhouse-gas concentration, cloud, or water-vapor feedbacks. Attribution requires additional analysis.`
        : `A decreasing downward longwave radiation trend (~${magnitude} ${unit}/decade) was detected in ${regionName}. This may reflect changes in cloud or moisture; the trend alone does not establish the cause.`;

    case 'albedo':
      return direction === 'Decreasing'
        ? `A decreasing albedo trend (~${magnitude}/decade) was detected in ${regionName}, consistent with surface darkening. Possible contributors include snow/ice retreat or land-cover change; the trend alone does not identify the mechanism.`
        : `An increasing albedo trend (~+${magnitude}/decade) was detected in ${regionName}, consistent with brighter surface conditions. Possible contributors include snow/ice expansion or land-cover change; attribution requires additional analysis.`;

    case 'snow_depth':
      return direction === 'Decreasing'
        ? `A decreasing snow-depth trend (~${magnitude} ${unit}/decade) was detected in ${regionName}, consistent with reduced snowpack. Possible contributors include warming or precipitation changes; the trend alone does not distinguish among them.`
        : `An increasing snow-depth trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}, consistent with increased snowpack. Possible contributors include precipitation or temperature changes; attribution requires additional analysis.`;

    case 'snow_precip':
      return direction === 'Decreasing'
        ? `A decreasing snowfall trend (~${magnitude} ${unit}/decade) was detected in ${regionName}. This may be consistent with a shift from snow to rain as temperatures rise, but the trend alone does not establish that mechanism.`
        : `An increasing snowfall trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}. Possible contributors include temperature or circulation changes; the trend alone does not establish the cause.`;

    case 'ozone_total':
      return direction === 'Increasing'
        ? `An increasing total-column ozone trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}. This is consistent with — but not proof of — ozone recovery, which may relate to declining halogen loading under the Montreal Protocol. Attribution requires additional analysis.`
        : `A decreasing total-column ozone trend (~${magnitude} ${unit}/decade) was detected in ${regionName}. Possible contributors include chemical or dynamical changes; the trend alone does not establish the cause.`;

    case 'soil_moist_top':
    case 'soil_moist_prof':
      return direction === 'Decreasing'
        ? `A decreasing soil-wetness trend (~${magnitude}/decade) was detected in ${regionName}, consistent with drying. Possible contributors include precipitation or evapotranspiration changes; the trend alone does not establish the cause.`
        : `An increasing soil-wetness trend (~+${magnitude}/decade) was detected in ${regionName}, consistent with wetting. Possible contributors include precipitation or evapotranspiration changes; attribution requires additional analysis.`;

    case 'sea_level':
      return direction === 'Increasing'
        ? `An increasing sea-level trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}. This is consistent with — but not proof of — contributions from thermal expansion and land-ice melt. Attribution to specific sources requires additional analysis.`
        : `A decreasing relative sea-level trend (~${magnitude} ${unit}/decade) was detected in ${regionName}. This could reflect local land motion (e.g., isostatic rebound) outpacing ocean-volume rise; the trend alone does not establish the cause.`;

    case 'sst':
      return direction === 'Increasing'
        ? `An increasing SST trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}, consistent with ocean warming. This may be associated with thermal stress on marine ecosystems, but the trend alone does not establish causation or attribute the warming to a specific driver.`
        : `A decreasing SST trend (~${magnitude} ${unit}/decade) was detected in ${regionName}. Possible contributors include upwelling changes or freshwater stratification; the trend alone does not establish the cause.`;

    case 'streamflow':
      return direction === 'Decreasing'
        ? `A decreasing streamflow trend (~${magnitude} ${unit}/decade) was detected in ${regionName}, consistent with reduced discharge. Possible contributors include precipitation, temperature, or water-management changes; the trend alone does not distinguish among them.`
        : `An increasing streamflow trend (~+${magnitude} ${unit}/decade) was detected in ${regionName}, consistent with increased discharge. Possible contributors include precipitation or land-use changes; attribution requires additional analysis.`;

    case 'evi':
      return direction === 'Decreasing'
        ? `A declining EVI trend (~${magnitude}/decade) was detected in ${regionName}, consistent with reduced canopy greenness. Possible explanations include vegetation stress, land-cover change, or disturbance; the trend alone does not distinguish among them.`
        : `An increasing EVI trend (~+${magnitude}/decade) was detected in ${regionName}, consistent with increased canopy greenness. Possible contributors include land management, climate, or recovery; attribution requires additional analysis.`;

    default:
      return `A ${dirWord} trend was detected in ${regionName} (~${direction === 'Increasing' ? '+' : '-'}${magnitude} ${unit}/decade). The physical driver(s) cannot be determined from the trend test alone; attribution requires additional analysis.`;
  }
}
