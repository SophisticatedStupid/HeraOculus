import { InterpolationRasterType } from '../../types/earthScience';

export function generateSyntheticRaster({
  width = 64,
  height = 64,
  type = 'continuous',
  missingPct = 0.15,
  seed = 42,
}: {
  width?: number;
  height?: number;
  type?: InterpolationRasterType;
  missingPct?: number;
  seed?: number;
}) {
  const rng = (() => {
    let s = seed;
    return () => {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    };
  })();

  const data: number[][] = [];
  for (let r = 0; r < height; r++) {
    const row: number[] = [];
    for (let c = 0; c < width; c++) {
      let val: number;
      if (type === 'categorical') {
        val = Math.floor(rng() * 5);
      } else {
        const grad =
          Math.sin((r / height) * Math.PI * 2) * 10 +
          Math.cos((c / width) * Math.PI * 2) * 10;
        val = 50 + grad + rng() * 5;
      }
      row.push(val);
    }
    data.push(row);
  }

  const missingMask: boolean[][] = [];
  for (let r = 0; r < height; r++) {
    missingMask.push(new Array(width).fill(false));
  }

  const nMissing = Math.floor(width * height * missingPct);
  const indices: [number, number][] = [];
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      indices.push([r, c]);
    }
  }
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.min(i, Math.floor(rng() * (i + 1)));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  for (let i = 0; i < nMissing && i < indices.length; i++) {
    const [r, c] = indices[i];
    missingMask[r][c] = true;
    if (rng() < 0.7) {
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const nr = r + dr,
            nc = c + dc;
          if (
            nr >= 0 &&
            nr < height &&
            nc >= 0 &&
            nc < width &&
            !missingMask[nr][nc] &&
            rng() < 0.5
          ) {
            missingMask[nr][nc] = true;
          }
        }
      }
    }
  }

  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      if (missingMask[r][c]) data[r][c] = NaN;
    }
  }

  return {
    data,
    width,
    height,
    transform: { a: 10, b: 0, c: 500000, d: 0, e: -10, f: 5000000 },
    crs: 'EPSG:32610',
    nodata: -9999,
  };
}
