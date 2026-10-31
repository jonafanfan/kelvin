"""The AI version of the Auto button (step 4 on the roadmap).

The trained AI from kelvin/model chooses the sliders. It's better at that than
the rules in rules.py (step 3.3 measured it), but it can't say *why*: its
"reasoning" is spread across 2.5 million weights that no person can read.
So each line just says, in plain words, what that slider did to the photo.
"""

import torch

from kelvin.editor import SLIDER_NAMES, neutral_params

from .rules import MIN_CHANGE, Suggestion, _shrink


def ai_suggest(img: torch.Tensor, model) -> Suggestion:
    """Let the trained AI look at a photo (3, H, W), values 0 to 1, and set the sliders."""
    values = model(_shrink(img)[None])[0]
    s = Suggestion(neutral_params())
    # Like the rules, skip changes too small to notice: they'd only add noise
    # to the explanations. Round to the sliders' 0.01 steps.
    for name, value in zip(SLIDER_NAMES, values.tolist()):
        if abs(value) >= MIN_CHANGE:
            s.params[name] = round(value, 2)
            up, down = EXPLANATIONS[name]
            s.reasons[name] = f"{up if value > 0 else down}."
    return s


# slider -> (what it did when raised, what it did when lowered)
EXPLANATIONS = {
    "temp": ("Made the light warmer (more orange)", "Made the light cooler (more blue)"),
    "tint": ("Shifted the colours towards pink", "Shifted the colours towards green"),
    "exposure": ("Brightened the whole photo", "Darkened the whole photo"),
    "contrast": ("Added contrast", "Softened the contrast"),
    "highlights": ("Brightened the bright parts", "Toned down the bright parts"),
    "shadows": ("Lifted the dark parts", "Deepened the dark parts"),
    "whites": ("Pushed the brightest point up", "Pulled the brightest point down"),
    "blacks": ("Lifted the darkest point", "Pushed the darkest point down"),
    "vibrance": ("Boosted the duller colours", "Muted the duller colours"),
    "saturation": ("Made every colour stronger", "Made every colour greyer"),
}
