"""The list of sliders Kelvin knows about.

Every slider goes from -1 to +1 and 0 always means "leave the photo alone".
Keeping every slider on the same scale makes life easier later: the AI only
has to learn to output numbers between -1 and +1, whatever the slider is.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Slider:
    name: str  # the key used in code, e.g. "exposure"
    label: str  # what the user sees, e.g. "Exposure"
    help: str  # a plain-English explanation shown in the app


# The order here is the order the edits are applied in (see ops.apply_edits),
# and roughly the order a human editor works in: fix the colour of the light,
# then overall brightness, then contrast, then the bright/dark parts, then colour.
SLIDERS = [
    Slider("temp", "Warmth", "Makes the photo more orange (+) or more blue (-)."),
    Slider("tint", "Tint", "Makes the photo more pink (+) or more green (-)."),
    Slider("exposure", "Exposure", "Brightens or darkens the whole photo, like letting in more or less light."),
    Slider("contrast", "Contrast", "Pushes darks darker and brights brighter (+), or flattens them (-)."),
    Slider("highlights", "Highlights", "Changes only the bright parts. Turn down to recover a blown-out sky."),
    Slider("shadows", "Shadows", "Changes only the dark parts. Turn up to lift shadows and reveal detail."),
    Slider("whites", "Whites", "Moves the very brightest point of the photo."),
    Slider("blacks", "Blacks", "Moves the very darkest point of the photo."),
    Slider("vibrance", "Vibrance", "Boosts dull colours more than bright ones."),
    Slider("saturation", "Saturation", "Makes every colour stronger (+) or greyer (-)."),
]

SLIDER_NAMES = [s.name for s in SLIDERS]


def neutral_params() -> dict:
    """All sliders at zero, which leaves the photo unchanged."""
    return {name: 0.0 for name in SLIDER_NAMES}


def as_params(values) -> dict:
    """A (batch, 10) tensor of slider values, one row per photo, in SLIDERS
    order -> {"temp": (batch,), "tint": (batch,), ...}, ready for apply_edits.
    This is the format the AI produces: 10 numbers per photo."""
    return {name: values[:, i] for i, name in enumerate(SLIDER_NAMES)}
