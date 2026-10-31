// The histogram: the brightness graph editors look at (kelvin/editor/histogram.py).
// Left is black, right is white; a tall bar means lots of pixels at that brightness.
//   - everything bunched on the left  -> the photo is too dark
//   - a spike stuck against the right -> bright parts are "blown out"
//   - nothing near either edge        -> the photo looks flat

import { luminance } from "./editor.js";

const BINS = 64;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// pixels: Float32Array of r,g,b,r,g,b... from 0 to 1. Returns 4 lists of
// BINS counts (brightness, red, green, blue), scaled so the tallest bar is 1.
function counts(pixels) {
  const bins = [0, 1, 2, 3].map(() => new Float32Array(BINS));
  const bin = (v) => Math.min(BINS - 1, Math.floor(v * BINS));
  for (let i = 0; i < pixels.length; i += 3) {
    const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
    bins[0][bin(luminance(r, g, b))]++;
    bins[1][bin(r)]++;
    bins[2][bin(g)]++;
    bins[3][bin(b)]++;
  }
  return bins.map((c) => {
    const tallest = Math.max(1, ...c);
    return c.map((v) => v / tallest);
  });
}

// Draws on the dark panel colour, not white, so the graph is never the
// brightest thing on screen (that would throw off your eye for the photo).
export function drawHistogram(canvas, edited, original) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);

  const legend = 20; // room for the "Before / After" key at the top
  const labels = 18; // and for "Black  Mid  White" at the bottom
  const plotHeight = height - labels;
  const x = (i) => ((i + 0.5) / BINS) * width;
  const y = (v) => plotHeight - v * (plotHeight - legend);
  const line = (values) => {
    ctx.beginPath();
    values.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  };

  const [lum, ...channels] = counts(edited);
  ["--hist-red", "--hist-green", "--hist-blue"].forEach((colour, c) => {
    line(channels[c]);
    ctx.lineTo(x(BINS - 1), plotHeight);
    ctx.lineTo(x(0), plotHeight);
    ctx.globalAlpha = 0.2;
    ctx.fillStyle = css(colour);
    ctx.fill();
    ctx.globalAlpha = 1;
  });

  if (original) {
    line(counts(original)[0]);
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = css("--muted");
    ctx.stroke();
    ctx.setLineDash([]);
  }
  line(lum);
  ctx.lineWidth = 1.6;
  ctx.strokeStyle = css("--text");
  ctx.stroke();

  ctx.font = `11px ${css("--font-ui")}`;
  ctx.fillStyle = css("--muted");
  ctx.textBaseline = "bottom";
  [["Black", 0, "left"], ["Mid", width / 2, "center"], ["White", width, "right"]].forEach(([text, at, align]) => {
    ctx.textAlign = align;
    ctx.fillText(text, at, height);
  });

  // Legend, top left
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 1.6;
  let at = 2;
  for (const [label, colour, dashed] of [["Before", "--muted", true], ["After", "--text", false]]) {
    if (label === "Before" && !original) continue;
    ctx.setLineDash(dashed ? [4, 3] : []);
    ctx.strokeStyle = css(colour);
    ctx.beginPath();
    ctx.moveTo(at, 8);
    ctx.lineTo(at + 18, 8);
    ctx.stroke();
    ctx.fillStyle = css("--text-soft");
    ctx.fillText(label, at + 24, 8);
    at += 24 + ctx.measureText(label).width + 16;
  }
  ctx.setLineDash([]);
}
