export function arrayToImageData(
  data: number[][],
  options: {
    colormap?: 'viridis' | 'gray' | 'categorical';
    nodataColor?: string;
    vmin?: number;
    vmax?: number;
  } = {}
): ImageData {
  const { colormap = 'viridis', nodataColor = '#1a1a1a', vmin, vmax } = options;
  const h = data.length;
  const w = data[0]?.length ?? 0;
  const imgData = new ImageData(w, h);

  const vals: number[] = [];
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      const v = data[r][c];
      if (v !== null && !Number.isNaN(v)) vals.push(v);
    }

  const minV = vmin ?? (vals.length > 0 ? Math.min(...vals) : 0);
  const maxV = vmax ?? (vals.length > 0 ? Math.max(...vals) : 1);
  const range = maxV - minV || 1;

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const idx = (r * w + c) * 4;
      const v = data[r][c];
      if (v === null || Number.isNaN(v)) {
        const [rr, gg, bb] = hexToRgb(nodataColor);
        imgData.data[idx] = rr;
        imgData.data[idx + 1] = gg;
        imgData.data[idx + 2] = bb;
        imgData.data[idx + 3] = 255;
      } else {
        const t = Math.max(0, Math.min(1, (v - minV) / range));
        const [rr, gg, bb] =
          colormap === 'categorical'
            ? catColor(Math.floor(v))
            : colormap === 'gray'
              ? [t * 255, t * 255, t * 255]
              : viridis(t);
        imgData.data[idx] = rr;
        imgData.data[idx + 1] = gg;
        imgData.data[idx + 2] = bb;
        imgData.data[idx + 3] = 255;
      }
    }
  }
  return imgData;
}

export function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!m) return [0, 0, 0];
  return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)];
}

export function catColor(c: number): [number, number, number] {
  const colors: [number, number, number][] = [
    [255, 107, 107],
    [78, 205, 196],
    [69, 183, 209],
    [156, 122, 245],
    [255, 176, 78],
  ];
  return colors[c % colors.length];
}

export function viridis(t: number): [number, number, number] {
  const r = 68 + Math.round(187 * Math.pow(t, 0.7));
  const g = 18 + Math.round(150 * Math.pow(t, 0.8));
  const b = 114 - Math.round(70 * Math.pow(t, 0.6));
  return [
    Math.min(255, Math.max(0, r)),
    Math.min(255, Math.max(0, g)),
    Math.min(255, Math.max(0, b)),
  ];
}
