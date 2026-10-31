"""The rule-based "Auto" button: hand-written rules that set the sliders.

This is step 2 on the roadmap. No AI yet: each rule looks at one measurement
of the photo (mostly from its histogram) and sets one or two sliders, the way
a careful beginner following an editing tutorial would. It's also the score
the AI has to beat in step 3: if a trained model can't do better than these
rules, it isn't worth having.

How it works
------------
The rules run in the same order the editor applies the sliders (see
params.SLIDERS). After each rule we apply everything suggested so far and
measure the result, so later rules see the photo as it now looks. For example,
the contrast rule measures the photo *after* exposure has been fixed.

The golden rule is "first, do no harm": each rule only acts when something is
clearly off, and never moves a slider all the way. A photo that already looks
fine should come back almost untouched.

Every rule also writes a one-line reason, so the app can say why it moved
each slider.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from kelvin.editor import apply_edits, luminance, neutral_params
from kelvin.editor import ops

# The rules only need a rough look at the photo, so they measure a small copy.
MEASURE_SIDE = 256

# Ignore changes smaller than this: a slider at 0.03 does nothing you'd notice,
# and an explanation for it would just be noise.
MIN_CHANGE = 0.05

# What a "typical" well-exposed photo measures. Hand-picked starting points;
# step 3 will show how far off they are from what the experts do.
MIDTONE_RANGE = (0.30, 0.55)  # median brightness: inside this range, leave exposure alone
TARGET_SPREAD = 0.17  # standard deviation of brightness: lower looks flat
TARGET_COLOURFULNESS = 0.10  # average colourfulness: lower looks dull
CLIP_LOW, CLIP_HIGH = 0.04, 0.96  # "nearly black" / "nearly white"
GREYISH = 0.35  # how far a pixel's channels can differ (relative) and still count as "nearly grey"

# The furthest each rule will move its slider(s). Rules are guesses; a wrong
# guess at full strength ruins a photo, a wrong guess at half strength doesn't.
MAX_WHITE_BALANCE = 0.4
MAX_EXPOSURE = 0.75  # 1.5 stops: a night photo is supposed to be dark
MAX_CONTRAST = 0.5
MAX_TONE = 0.6  # highlights / shadows
MAX_LEVELS = 0.5  # whites / blacks
MAX_VIBRANCE = 0.3


@dataclass
class Suggestion:
    params: dict  # slider name -> value from -1 to 1 (every slider is present)
    reasons: dict = field(default_factory=dict)  # slider name -> one-line reason


def suggest(img: torch.Tensor) -> Suggestion:
    """Look at a photo (3, H, W) with values from 0 to 1 and suggest sliders."""
    img = _shrink(img)
    s = Suggestion(neutral_params())
    for rule in RULES:
        current = apply_edits(img, s.params)
        for name, (value, reason) in rule(current).items():
            value = round(max(-1.0, min(1.0, value)), 2)
            if abs(value) >= MIN_CHANGE:
                s.params[name] = value
                s.reasons[name] = reason
    return s


# ---------------------------------------------------------------------------
# The rules, in the order they run. Each takes the photo as it looks so far
# and returns {slider name: (value, reason)}.
# ---------------------------------------------------------------------------

def white_balance(img):
    """Find the things in the photo that are *nearly* grey (walls, clouds,
    roads, shirts). If they all lean orange or blue (or green or pink), that's
    probably the light, not the objects, so we lean it back. Only halfway,
    because sometimes the light really is orange (a sunset) and fully
    "correcting" it would ruin the photo.

    Strongly coloured pixels are ignored: a big green lawn says nothing about
    the light. So are blown-out and very dark pixels, whose colour is unreliable.
    If there's too little near-grey in the photo, we leave white balance alone."""
    top, bottom = img.amax(dim=0), img.amin(dim=0)
    near_grey = (top < CLIP_HIGH) & (top > 0.1) & ((top - bottom) / top.clamp_min(1e-4) < GREYISH)
    if near_grey.float().mean() < 0.05:
        return {}
    r, g, b = ops.srgb_to_linear(img)[:, near_grey].mean(dim=1).clamp_min(1e-4)
    # The editor's warmth multiplies red by 2^(0.4 t) and blue by 2^(-0.4 t),
    # so this value of t would make red and blue equal.
    temp = _cap(-0.5 * torch.log2(r / b).item() / (2 * ops.WARMTH_STOPS), MAX_WHITE_BALANCE)
    # Tint multiplies green by 2^(-0.3 n); aim green at the middle of red and blue.
    tint = _cap(0.5 * torch.log2(g / (r * b).sqrt()).item() / ops.TINT_STOPS, MAX_WHITE_BALANCE)

    out = {}
    if temp:
        cast = "orange" if temp < 0 else "blue"
        out["temp"] = (temp, f"Grey things in the photo looked {cast}, a sign the light was {cast}. This evens it out partway, to keep the mood.")
    if tint:
        cast = "green" if tint > 0 else "pink"
        out["tint"] = (tint, f"Grey things in the photo leaned {cast}, so this nudges them back partway.")
    return out


def exposure(img):
    """If the middle brightness (the median) is clearly too dark or too bright,
    move it to the nearest edge of MIDTONE_RANGE. Inside the range, it's a
    matter of taste, so we leave it. Exposure doubles the light per stop, so
    the gap is measured in stops."""
    median = _brightness(img).median().item()
    low, high = MIDTONE_RANGE
    if low <= median <= high:
        return {}
    target = low if median < low else high
    stops = torch.log2(ops.srgb_to_linear(torch.tensor(target)) / ops.srgb_to_linear(torch.tensor(max(median, 1e-3)))).item()
    value = _cap(stops / ops.EXPOSURE_STOPS, MAX_EXPOSURE)
    word = "dark" if value > 0 else "bright"
    return {"exposure": (value, f"The photo was {word} overall (middle brightness {median:.0%}; most good photos sit between {low:.0%} and {high:.0%}).")}


def contrast(img):
    """If the brightness values are bunched together, the photo looks flat.
    Near middle grey, contrast +a stretches differences by about (1 + a/2).
    Only ever adds contrast: a contrasty photo is usually contrasty on purpose."""
    spread = _brightness(img).std().item()
    if spread >= TARGET_SPREAD:
        return {}
    value = _cap(2 * (TARGET_SPREAD / max(spread, 1e-3) - 1), MAX_CONTRAST)
    return {"contrast": (value, "The photo looked flat: its brightest and darkest parts were too close together.")}


def highlights_shadows(img):
    """If a lot of the photo is nearly white, pull the bright parts down to
    bring back detail. If a lot is nearly black, lift the dark parts."""
    lum = _brightness(img)
    bright = (lum > 0.9).float().mean().item()
    dark = (lum < 0.1).float().mean().item()
    out = {}
    if bright > 0.08:
        out["highlights"] = (-_cap(2 * bright, MAX_TONE), f"{bright:.0%} of the photo was very bright, so this recovers detail there.")
    if dark > 0.08:
        out["shadows"] = (_cap(2 * dark, MAX_TONE), f"{dark:.0%} of the photo was very dark, so this lifts detail out of the shadows.")
    return out


def whites_blacks(img):
    """Make the brightest pixels reach (nearly) white and the darkest reach
    (nearly) black, so the photo uses the whole range. Like the "levels" tool:
    we find the near-brightest point (ignoring a few stray pixels) and stretch
    it upwards."""
    brightest = torch.quantile(img.amax(dim=0).flatten(), 0.995).item()
    darkest = torch.quantile(img.amin(dim=0).flatten(), 0.005).item()
    out = {}
    # The editor puts the white point at 1 - 0.15 w, so this w puts it at `brightest`.
    if brightest < 0.9:
        whites = _cap((CLIP_HIGH - brightest) / ops.LEVELS_STRENGTH, MAX_LEVELS)
        out["whites"] = (whites, f"Nothing in the photo reached white (brightest point {brightest:.0%}), which looks dull.")
    # ...and the black point at -0.15 b.
    if darkest > 0.08:
        blacks = -_cap((darkest - CLIP_LOW) / ops.LEVELS_STRENGTH, MAX_LEVELS)
        out["blacks"] = (blacks, f"Nothing in the photo reached black (darkest point {darkest:.0%}), which looks washed out.")
    return out


def colour(img):
    """If colours are dull on average, boost them with vibrance (it goes easy
    on colours that are already strong, and on skin). Saturation is left alone:
    it's the blunter tool."""
    colourfulness = (img.amax(dim=0) - img.amin(dim=0)).mean().item()
    if colourfulness >= TARGET_COLOURFULNESS:
        return {}
    value = _cap(2 * (TARGET_COLOURFULNESS - colourfulness) / TARGET_COLOURFULNESS, MAX_VIBRANCE)
    return {"vibrance": (value, "Colours were on the dull side, so this boosts the weaker ones.")}


RULES = [white_balance, exposure, contrast, highlights_shadows, whites_blacks, colour]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cap(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _shrink(img: torch.Tensor) -> torch.Tensor:
    longest = max(img.shape[-2:])
    if longest <= MEASURE_SIDE:
        return img
    scale = MEASURE_SIDE / longest
    return F.interpolate(img[None], scale_factor=scale, mode="area")[0]


def _brightness(img: torch.Tensor) -> torch.Tensor:
    return luminance(img[None])[0, 0]
