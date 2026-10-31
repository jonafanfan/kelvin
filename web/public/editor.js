// The photo editor, in JavaScript: a line-by-line copy of kelvin/editor/ops.py.
//
// The Python version (PyTorch) is what the AI trained with. This copy runs the
// same maths in the browser, so the app needs no server. tests/test_web.py
// checks that the two give the same result on the same photo.
//
// A photo here is a Float32Array of red, green, blue, red, green, blue...
// with values from 0 (black) to 1 (full brightness).
//
// Rule for every step: a slider value of 0 must leave the photo unchanged.

// The sliders, in the order the edits are applied (kelvin/editor/params.py).
export const SLIDERS = [
  { name: "temp", label: "Warmth", help: "Makes the photo more orange (+) or more blue (-)." },
  { name: "tint", label: "Tint", help: "Makes the photo more pink (+) or more green (-)." },
  { name: "exposure", label: "Exposure", help: "Brightens or darkens the whole photo, like letting in more or less light." },
  { name: "contrast", label: "Contrast", help: "Pushes darks darker and brights brighter (+), or flattens them (-)." },
  { name: "highlights", label: "Highlights", help: "Changes only the bright parts. Turn down to recover a blown-out sky." },
  { name: "shadows", label: "Shadows", help: "Changes only the dark parts. Turn up to lift shadows and reveal detail." },
  { name: "whites", label: "Whites", help: "Moves the very brightest point of the photo." },
  { name: "blacks", label: "Blacks", help: "Moves the very darkest point of the photo." },
  { name: "vibrance", label: "Vibrance", help: "Boosts dull colours more than bright ones." },
  { name: "saturation", label: "Saturation", help: "Makes every colour stronger (+) or greyer (-)." },
];
export const SLIDER_NAMES = SLIDERS.map((s) => s.name);

// How strongly each slider acts at its maximum (+1 or -1), as in ops.py.
const EXPOSURE_STOPS = 2.0;
const WARMTH_STOPS = 0.4;
const TINT_STOPS = 0.3;
const TONE_STRENGTH = 0.15;
const LEVELS_STRENGTH = 0.15;

export function neutralParams() {
  return Object.fromEntries(SLIDER_NAMES.map((name) => [name, 0]));
}

// How bright a colour looks to a human. Green counts most, blue least.
export function luminance(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// Photo files store brightness on a curved scale (sRGB); real light adds up
// on a straight ("linear") one. See srgb_to_linear in ops.py.
function srgbToLinear(v) {
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(v) {
  return v <= 0.0031308 ? v * 12.92 : 1.055 * Math.max(v, 1e-6) ** (1 / 2.4) - 0.055;
}

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

// Apply every slider, in the same order as apply_edits in ops.py, and return
// a new edited photo. `params` maps slider name -> value from -1 to 1.
export function applyEdits(pixels, params) {
  const p = { ...neutralParams(), ...params };
  const out = new Float32Array(pixels.length);

  // Everything that's the same for every pixel, worked out once.
  const light = 2 ** (EXPOSURE_STOPS * p.exposure);
  const gainR = light * 2 ** (WARMTH_STOPS * p.temp);
  const gainG = light * 2 ** (-TINT_STOPS * p.tint);
  const gainB = light * 2 ** (-WARMTH_STOPS * p.temp);
  const contrast = 2 * p.contrast;
  const blackPoint = -LEVELS_STRENGTH * p.blacks;
  const whitePoint = 1 - LEVELS_STRENGTH * p.whites;
  const range = whitePoint - blackPoint;
  const saturation = 1 + p.saturation;

  for (let i = 0; i < pixels.length; i += 3) {
    // 1. Light: white balance and exposure, on linear light.
    let r = clamp01(linearToSrgb(srgbToLinear(pixels[i]) * gainR));
    let g = clamp01(linearToSrgb(srgbToLinear(pixels[i + 1]) * gainG));
    let b = clamp01(linearToSrgb(srgbToLinear(pixels[i + 2]) * gainB));

    // 2. Tone. Contrast: an S-shaped curve around middle grey.
    r += contrast * (r - 0.5) * r * (1 - r);
    g += contrast * (g - 0.5) * g * (1 - g);
    b += contrast * (b - 0.5) * b * (1 - b);

    // Highlights and shadows: masks that pick out the bright or dark parts.
    const lum = luminance(r, g, b);
    const shadowMask = 6.75 * lum * (1 - lum) ** 2;
    const highlightMask = 6.75 * lum ** 2 * (1 - lum);
    const change = TONE_STRENGTH * (p.shadows * shadowMask + p.highlights * highlightMask);
    r += change;
    g += change;
    b += change;

    // Whites and blacks: stretch or squash the range, like the "levels" tool.
    r = (r - blackPoint) / range;
    g = (g - blackPoint) / range;
    b = (b - blackPoint) / range;

    // 3. Colour: move each pixel's colour away from (or towards) grey.
    const grey = luminance(r, g, b);
    const colourfulness = Math.max(r, g, b) - Math.min(r, g, b);
    const boost = saturation * (1 + p.vibrance * (1 - colourfulness));
    out[i] = clamp01(grey + (r - grey) * boost);
    out[i + 1] = clamp01(grey + (g - grey) * boost);
    out[i + 2] = clamp01(grey + (b - grey) * boost);
  }
  return out;
}
