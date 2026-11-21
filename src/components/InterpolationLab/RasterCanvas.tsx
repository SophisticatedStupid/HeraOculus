import React, { useEffect, useRef } from 'react';
import { arrayToImageData } from './rendering';

function RasterCanvas({
  data,
  title,
  colormap = 'viridis',
}: {
  data: (number | null)[][];
  title: string;
  colormap?: 'viridis' | 'gray' | 'categorical';
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.length) return;
    const h = data.length;
    const w = data[0]?.length ?? 0;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const flat: number[][] = data.map((row) =>
      row.map((v) => (v === null ? NaN : v))
    );
    const imgData = arrayToImageData(flat, { colormap });
    ctx.putImageData(imgData, 0, 0);
  }, [data, colormap]);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="text-xs font-semibold text-silver-400">{title}</div>
      <canvas
        ref={canvasRef}
        className="border border-ink-700 bg-ink-950 block max-w-full max-h-[300px]"
        style={{ imageRendering: 'pixelated' }}
      />
    </div>
  );
}

export default RasterCanvas;
