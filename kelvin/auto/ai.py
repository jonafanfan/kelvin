"""The AI version of the Auto button (step 4 on the roadmap).

The trained AI from kelvin/model chooses the sliders. It's better at that than
the rules in rules.py (step 3.3 measured it), but it can't say *why*: its
"reasoning" is spread across 2.5 million weights that no person can read.
So each line says, in plain words, what that slider did to the photo, with a
small caption underneath giving the usual reason an editor makes that change.
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
            line, caption = EXPLANATIONS[name][0 if value > 0 else 1]
            s.reasons[name] = f"{line}."
            s.captions[name] = f"{caption}."
    return s


# slider -> ((what it did when raised, usual reason), (what it did when lowered, usual reason))
# The captions say what usually calls for each change: the AI can't tell us
# its own reason, so we don't pretend it can.
EXPLANATIONS = {
    "temp": (("Made the light warmer (more orange)", "Usually because the scene looks too blue or cold"),
             ("Made the light cooler (more blue)", "Usually because the scene looks too orange or yellow")),
    "tint": (("Shifted the colours towards pink", "Usually because the scene has a green cast"),
             ("Shifted the colours towards green", "Usually because the scene has a pink or purple cast")),
    "exposure": (("Brightened the whole photo", "Usually because the photo looks too dark"),
                 ("Darkened the whole photo", "Usually because the photo looks too bright")),
    "contrast": (("Added contrast", "Usually because the photo looks flat or hazy"),
                 ("Softened the contrast", "Usually because the photo looks too harsh")),
    "highlights": (("Brightened the bright parts", "Usually to make skies and lights pop"),
                   ("Toned down the bright parts", "Usually to bring back detail in skies and bright areas")),
    "shadows": (("Lifted the dark parts", "Usually to show detail hiding in the shadows"),
                ("Deepened the dark parts", "Usually to add depth and mood")),
    "whites": (("Pushed the brightest point up", "Usually because the photo looks dull or grey"),
               ("Pulled the brightest point down", "Usually to stop the brightest parts turning pure white")),
    "blacks": (("Lifted the darkest point", "Usually for a softer, faded look"),
               ("Pushed the darkest point down", "Usually because the photo looks washed out")),
    "vibrance": (("Boosted the duller colours", "Usually because the colours look dull"),
                 ("Muted the duller colours", "Usually for a calmer, more natural look")),
    "saturation": (("Made every colour stronger", "Usually because the whole photo looks washed out"),
                   ("Made every colour greyer", "Usually because colours look too loud or unnatural")),
}
