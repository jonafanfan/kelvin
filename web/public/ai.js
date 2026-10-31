// The AI version of the Auto button, in the browser: the same as
// kelvin/auto/ai.py, but the AI runs here instead of in PyTorch.
//
// slider_net.onnx is the trained AI, saved by `python -m kelvin.model.export`.
// ONNX Runtime Web (in vendor/) runs it on the visitor's own device, so photos
// never leave their computer or phone.

import { SLIDER_NAMES, neutralParams } from "./editor.js";

const PHOTO_SIDE = 256; // the AI always looks at a 256 x 256 copy, like in training
const MIN_CHANGE = 0.05; // skip changes too small to notice, like the rules do

// slider -> [what it did when raised, what it did when lowered]
export const EXPLANATIONS = {
  temp: ["Made the light warmer (more orange)", "Made the light cooler (more blue)"],
  tint: ["Shifted the colours towards pink", "Shifted the colours towards green"],
  exposure: ["Brightened the whole photo", "Darkened the whole photo"],
  contrast: ["Added contrast", "Softened the contrast"],
  highlights: ["Brightened the bright parts", "Toned down the bright parts"],
  shadows: ["Lifted the dark parts", "Deepened the dark parts"],
  whites: ["Pushed the brightest point up", "Pulled the brightest point down"],
  blacks: ["Lifted the darkest point", "Pushed the darkest point down"],
  vibrance: ["Boosted the duller colours", "Muted the duller colours"],
  saturation: ["Made every colour stronger", "Made every colour greyer"],
};

let ort;
let session;

// Load the runtime and the AI once, the first time Auto is pressed (or earlier,
// in the background, so the first press is quick).
export async function loadAI() {
  if (!session) {
    ort = await import("./vendor/ort.wasm.bundle.min.mjs");
    // The runtime's engine, compiled to WebAssembly (fast code browsers can run)
    ort.env.wasm.wasmPaths = { wasm: new URL("./vendor/ort-wasm-simd-threaded.wasm", import.meta.url).href };
    ort.env.wasm.numThreads = 1; // the AI is small; extra threads need special server settings
    session = await ort.InferenceSession.create("./slider_net.onnx");
  }
  return session;
}

// image: anything the browser can draw (an ImageBitmap, a canvas...).
// Returns { params, reasons } like Suggestion in kelvin/auto/rules.py.
export async function aiSuggest(image) {
  const ai = await loadAI();
  const input = new ort.Tensor("float32", toAIInput(image), [1, 3, PHOTO_SIDE, PHOTO_SIDE]);
  const { sliders } = await ai.run({ photo: input });

  const params = neutralParams();
  const reasons = {};
  SLIDER_NAMES.forEach((name, i) => {
    const value = sliders.data[i];
    if (Math.abs(value) >= MIN_CHANGE) {
      params[name] = Math.round(value * 100) / 100; // the sliders move in 0.01 steps
      const [up, down] = EXPLANATIONS[name];
      reasons[name] = `${value > 0 ? up : down}.`;
    }
  });
  return { params, reasons };
}

// Squash the photo to 256 x 256 and lay it out the way PyTorch expects:
// all the red values, then all the green, then all the blue, from 0 to 1.
function toAIInput(image) {
  const canvas = new OffscreenCanvas(PHOTO_SIDE, PHOTO_SIDE);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(image, 0, 0, PHOTO_SIDE, PHOTO_SIDE);
  const rgba = ctx.getImageData(0, 0, PHOTO_SIDE, PHOTO_SIDE).data;
  const n = PHOTO_SIDE * PHOTO_SIDE;
  const planes = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    planes[i] = rgba[4 * i] / 255;
    planes[n + i] = rgba[4 * i + 1] / 255;
    planes[2 * n + i] = rgba[4 * i + 2] / 255;
  }
  return planes;
}
