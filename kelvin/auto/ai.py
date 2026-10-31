"""The AI version of the Auto button (step 4 on the roadmap).

The trained AI from kelvin/model chooses the sliders. It's better at that than
the rules in rules.py (step 3.3 measured it), but it can't say *why*: its
"reasoning" is spread across 2.5 million weights that no person can read.
Getting an AI to explain itself is a whole field of research.

So we explain the honest way instead: for every slider the AI moved, we
measure what that slider did to *this* photo, for example "middle brightness
22% -> 34%". The sliders are applied one at a time in the editor's order, and
we measure just before and just after each one, so every number belongs to
that slider alone.
"""

import torch

from kelvin.editor import SLIDER_NAMES, apply_edits, luminance, neutral_params

from .rules import MIN_CHANGE, Suggestion, _shrink


def ai_suggest(img: torch.Tensor, model) -> Suggestion:
    """Let the trained AI look at a photo (3, H, W), values 0 to 1, and set the sliders."""
    img = _shrink(img)
    values = model(img[None])[0]
    s = Suggestion(neutral_params())
    # Like the rules, skip changes too small to notice: they'd only add noise
    # to the explanations. Round to the sliders' 0.01 steps.
    for name, value in zip(SLIDER_NAMES, values.tolist()):
        if abs(value) >= MIN_CHANGE:
            s.params[name] = round(value, 2)

    so_far = neutral_params()
    for name in SLIDER_NAMES:
        if s.params[name] == 0:
            continue
        before = apply_edits(img, so_far)
        so_far[name] = s.params[name]
        s.reasons[name] = explain(name, s.params[name], before, apply_edits(img, so_far))
    return s


def explain(name: str, value: float, before: torch.Tensor, after: torch.Tensor) -> str:
    """One line saying what a slider did, measured on the photo."""
    up, down, measure = EXPLANATIONS[name]
    what = up if value > 0 else down
    if measure is None:
        return f"{what}."
    label, how = measure
    return f"{what}: {label} {how(before):.0%} → {how(after):.0%}."


# ---------------------------------------------------------------------------
# Measurements. Each takes a photo (3, H, W) and returns a number from 0 to 1.
# ---------------------------------------------------------------------------

def _brightness(img):
    return luminance(img[None])[0, 0].flatten()


def middle(img):
    return _brightness(img).median().item()


def gap(img):
    """How far apart the lighter and darker halves of the photo are: the
    brightness three-quarters of the way up minus a quarter of the way up."""
    q = torch.quantile(_brightness(img), torch.tensor([0.25, 0.75]))
    return (q[1] - q[0]).item()


def bright_parts(img):
    """The brightness of the brightest tenth of the photo."""
    return torch.quantile(_brightness(img), 0.9).item()


def dark_parts(img):
    """The brightness of the darkest tenth of the photo."""
    return torch.quantile(_brightness(img), 0.1).item()


def brightest(img):
    # Ignore the top 0.5% of pixels: a few stray ones shouldn't count.
    return torch.quantile(img.amax(dim=0).flatten(), 0.995).item()


def darkest(img):
    return torch.quantile(img.amin(dim=0).flatten(), 0.005).item()


def colourfulness(img):
    """On average, how far apart a pixel's red, green and blue are (0 for grey)."""
    return (img.amax(dim=0) - img.amin(dim=0)).mean().item()


# slider -> (what it did when raised, what it did when lowered, (measurement name, measurement) or None)
# Warmth and tint have no number: a "colour of the light" percentage would mean nothing to most people.
EXPLANATIONS = {
    "temp": ("Made the light warmer (more orange)", "Made the light cooler (more blue)", None),
    "tint": ("Shifted the colours towards pink", "Shifted the colours towards green", None),
    "exposure": ("Brightened the whole photo", "Darkened the whole photo", ("middle brightness", middle)),
    "contrast": ("Added contrast", "Softened the contrast", ("gap between light and dark parts", gap)),
    "highlights": ("Brightened the bright parts", "Toned down the bright parts", ("bright parts", bright_parts)),
    "shadows": ("Lifted the dark parts", "Deepened the dark parts", ("dark parts", dark_parts)),
    "whites": ("Pushed the brightest point up", "Pulled the brightest point down", ("brightest point", brightest)),
    "blacks": ("Lifted the darkest point", "Pushed the darkest point down", ("darkest point", darkest)),
    "vibrance": ("Boosted the duller colours", "Muted the duller colours", ("colourfulness", colourfulness)),
    "saturation": ("Made every colour stronger", "Made every colour greyer", ("colourfulness", colourfulness)),
}
