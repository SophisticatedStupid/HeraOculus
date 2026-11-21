import { describe, it, expect } from 'vitest';

// Mock ImageData for Node environment
globalThis.ImageData = class ImageData {
  width: number;
  height: number;
  data: Uint8ClampedArray;
  constructor(w: number, h: number) {
    this.width = w;
    this.height = h;
    this.data = new Uint8ClampedArray(w * h * 4);
  }
} as any;

import {
  generateSyntheticRaster,
  arrayToImageData,
  hexToRgb,
  catColor,
  viridis,
} from '../src/components/InterpolationLab';
import { InterpolationRasterType } from '../src/types/earthScience';

describe('generateSyntheticRaster', () => {
  it('generates a continuous raster with correct dimensions', () => {
    const r = generateSyntheticRaster({ width: 32, height: 16, type: 'continuous', missingPct: 0.0, seed: 42 });
    expect(r.width).toBe(32);
    expect(r.height).toBe(16);
    expect(r.data.length).toBe(16);
    expect(r.data[0].length).toBe(32);
    expect(r.crs).toBe('EPSG:32610');
  });

  it('generates a categorical raster with integer class values', () => {
    const r = generateSyntheticRaster({ width: 10, height: 10, type: 'categorical', missingPct: 0.0, seed: 42 });
    for (const row of r.data) {
      for (const v of row) {
        expect(Number.isInteger(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(5);
      }
    }
  });

  it('introduces missing values', () => {
    const r = generateSyntheticRaster({ width: 100, height: 100, type: 'continuous', missingPct: 0.2, seed: 42 });
    let missing = 0;
    let total = 0;
    for (const row of r.data) {
      for (const v of row) {
        total++;
        if (Number.isNaN(v)) missing++;
      }
    }
    // ~20% base missing + clustering => should be > 20%
    expect(missing / total).toBeGreaterThan(0.15);
    expect(missing / total).toBeLessThan(0.8);
  });

  it('is deterministic with same seed', () => {
    const r1 = generateSyntheticRaster({ width: 10, height: 10, seed: 42 });
    const r2 = generateSyntheticRaster({ width: 10, height: 10, seed: 42 });
    expect(r1.data).toEqual(r2.data);
  });

  it('produces different data with different seeds', () => {
    const r1 = generateSyntheticRaster({ width: 10, height: 10, seed: 1 });
    const r2 = generateSyntheticRaster({ width: 10, height: 10, seed: 2 });
    let different = false;
    for (let i = 0; i < 10; i++) {
      for (let j = 0; j < 10; j++) {
        if (r1.data[i][j] !== r2.data[i][j]) { different = true; break; }
      }
      if (different) break;
    }
    expect(different).toBe(true);
  });
});

describe('arrayToImageData', () => {
  it('creates ImageData with correct dimensions', () => {
    const data: number[][] = [[1, 2], [3, 4]];
    const img = arrayToImageData(data);
    expect(img.width).toBe(2);
    expect(img.height).toBe(2);
    expect(img.data.length).toBe(16); // 2*2*4
  });

  it('renders NoData as nodataColor', () => {
    const data: number[][] = [[1, NaN], [3, 4]];
    const img = arrayToImageData(data, { nodataColor: '#ff0000' });
    const idx = (0 * 2 + 1) * 4; // row 0, col 1
    expect(img.data[idx]).toBe(255);   // R
    expect(img.data[idx + 1]).toBe(0); // G
    expect(img.data[idx + 2]).toBe(0); // B
    expect(img.data[idx + 3]).toBe(255); // A
  });

  it('normalizes values to colormap range', () => {
    const data: number[][] = [[0, 50], [100, 150]];
    const img = arrayToImageData(data, { colormap: 'gray' });
    // Min value 0 -> darkest, max 150 -> brightest
    const idxMin = (0 * 2 + 0) * 4; // value 0
    const idxMax = (1 * 2 + 1) * 4; // value 150
    expect(img.data[idxMax]).toBeGreaterThan(img.data[idxMin]);
  });

  it('uses categorical colormap', () => {
    const data: number[][] = [[0, 1], [2, 3]];
    const img = arrayToImageData(data, { colormap: 'categorical' });
    // Just verify it doesn't crash and produces valid RGBA
    expect(img.data.length).toBe(16);
    for (let i = 0; i < img.data.length; i += 4) {
      expect(img.data[i + 3]).toBe(255); // alpha always 255
    }
  });
});

describe('hexToRgb', () => {
  it('converts hex to RGB', () => {
    expect(hexToRgb('#ff0000')).toEqual([255, 0, 0]);
    expect(hexToRgb('#00ff00')).toEqual([0, 255, 0]);
    expect(hexToRgb('#0000ff')).toEqual([0, 0, 255]);
    expect(hexToRgb('#ffffff')).toEqual([255, 255, 255]);
    expect(hexToRgb('#000000')).toEqual([0, 0, 0]);
  });

  it('handles hex without # prefix', () => {
    expect(hexToRgb('ff0000')).toEqual([255, 0, 0]);
  });
});

describe('catColor', () => {
  it('returns cyclic colors', () => {
    expect(catColor(0)).toEqual([255, 107, 107]);
    expect(catColor(5)).toEqual(catColor(0)); // cycles back
    expect(catColor(10)).toEqual(catColor(0));
  });
});

describe('viridis', () => {
  it('returns values in [0, 255]', () => {
    for (let t = 0; t <= 1; t += 0.1) {
      const [r, g, b] = viridis(t);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(255);
      expect(g).toBeGreaterThanOrEqual(0);
      expect(g).toBeLessThanOrEqual(255);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThanOrEqual(255);
    }
  });

  it('is monotonically increasing in brightness', () => {
    const colors = [0, 0.25, 0.5, 0.75, 1].map(viridis);
    const brightness = colors.map(([r, g, b]) => r + g + b);
    for (let i = 1; i < brightness.length; i++) {
      expect(brightness[i]).toBeGreaterThanOrEqual(brightness[i - 1]);
    }
  });
});
