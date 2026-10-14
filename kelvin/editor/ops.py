"""The photo editor itself: one small function per slider.

How images are stored here
--------------------------
A photo is a PyTorch "tensor" (a grid of numbers) shaped (batch, 3, height, width):
  - batch  = how many photos at once (the app uses 1; training uses many)
  - 3      = the red, green and blue channels
  - values go from 0.0 (black) to 1.0 (full brightness)

Why PyTorch and not a normal image library?
-------------------------------------------
Later, the AI learns by asking "if I nudge this slider a tiny bit, does the photo
get closer to the expert's edit?". PyTorch can answer that question for every
slider automatically (this is called a "gradient"), but only if the editing is
written in PyTorch. So the editor is written in PyTorch once, and both the app
and the AI training use this same file. That way what you see in the app is
exactly what the AI learned to do.

Rule for every function: a slider value of 0 must leave the photo unchanged.
"""

import torch

from .params import SLIDER_NAMES

# How strongly each slider acts at its maximum (+1 or -1). These are hand-picked
# so that +1/-1 is a big but not absurd change. Tweak them freely.
EXPOSURE_STOPS = 2.0  # +1 = 2 "stops" brighter = 4x the light
WARMTH_STOPS = 0.4
TINT_STOPS = 0.3
TONE_STRENGTH = 0.15  # highlights / shadows
LEVELS_STRENGTH = 0.15  # whites / blacks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_batch(img: torch.Tensor) -> torch.Tensor:
    """Accept a single photo (3, H, W) as well as a batch (B, 3, H, W)."""
    return img.unsqueeze(0) if img.dim() == 3 else img


def _amount(value, img: torch.Tensor) -> torch.Tensor:
    """Turn a slider value (a number, or one number per photo in the batch)
    into a shape that can be multiplied with the image."""
    return torch.as_tensor(value, dtype=img.dtype, device=img.device).reshape(-1, 1, 1, 1)


def luminance(img: torch.Tensor) -> torch.Tensor:
    """How bright each pixel looks to a human, ignoring colour.
    Green counts most and blue least, because that's how our eyes work."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def srgb_to_linear(img: torch.Tensor) -> torch.Tensor:
    """Photo files store brightness on a curved scale (called sRGB) that matches
    how we see. Real light adds up on a straight ("linear") scale. Exposure and
    white balance are about real light, so we convert before doing them."""
    return torch.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img: torch.Tensor) -> torch.Tensor:
    # clamp_min avoids a divide-by-zero inside PyTorch's gradient maths at exactly 0
    curved = 1.055 * img.clamp_min(1e-6) ** (1 / 2.4) - 0.055
    return torch.where(img <= 0.0031308, img * 12.92, curved)


# ---------------------------------------------------------------------------
# One function per slider
# ---------------------------------------------------------------------------

def white_balance(linear: torch.Tensor, temp, tint) -> torch.Tensor:
    """Warmth turns red up and blue down (or the reverse). Tint turns green down
    (looks pinker) or up (looks greener). Works on linear light."""
    t = _amount(temp, linear)
    n = _amount(tint, linear)
    red = 2 ** (WARMTH_STOPS * t)
    green = 2 ** (-TINT_STOPS * n)
    blue = 2 ** (-WARMTH_STOPS * t)
    gains = torch.cat([red, green, blue], dim=1)
    return linear * gains


def exposure(linear: torch.Tensor, amount) -> torch.Tensor:
    """Multiply all the light, like a camera letting in more of it."""
    return linear * 2 ** (EXPOSURE_STOPS * _amount(amount, linear))


def contrast(img: torch.Tensor, amount) -> torch.Tensor:
    """An S-shaped curve: pixels darker than middle grey get darker, brighter
    ones get brighter. Pure black (0) and pure white (1) never move."""
    a = _amount(amount, img)
    return img + 2 * a * (img - 0.5) * img * (1 - img)


def highlights_shadows(img: torch.Tensor, highlights, shadows) -> torch.Tensor:
    """Brighten or darken only the dark parts (shadows) or only the bright parts
    (highlights). A "mask" decides how much each pixel is affected:
      - the shadow mask is strongest for dark-ish pixels (about 1/3 brightness)
      - the highlight mask is strongest for bright-ish pixels (about 2/3)
    Both masks are 0 at pure black and pure white, so those don't move."""
    lum = luminance(img)
    shadow_mask = 6.75 * lum * (1 - lum) ** 2  # peaks at 1.0 when lum = 1/3
    highlight_mask = 6.75 * lum**2 * (1 - lum)  # peaks at 1.0 when lum = 2/3
    change = TONE_STRENGTH * (_amount(shadows, img) * shadow_mask + _amount(highlights, img) * highlight_mask)
    return img + change


def whites_blacks(img: torch.Tensor, whites, blacks) -> torch.Tensor:
    """Stretch or squash the range between the darkest and brightest point
    (like the "levels" tool). Whites + makes near-white pixels hit full white;
    blacks + lifts the darkest pixels into grey (a faded film look)."""
    black_point = -LEVELS_STRENGTH * _amount(blacks, img)
    white_point = 1 - LEVELS_STRENGTH * _amount(whites, img)
    return (img - black_point) / (white_point - black_point)


def vibrance_saturation(img: torch.Tensor, vibrance, saturation) -> torch.Tensor:
    """Move each pixel's colour away from (or towards) grey.
    Saturation does this equally for every pixel. Vibrance does it more for
    pixels that are currently dull, so already-strong colours (and skin)
    don't go neon."""
    grey = luminance(img)
    current_colourfulness = img.max(dim=1, keepdim=True).values - img.min(dim=1, keepdim=True).values
    boost = (1 + _amount(saturation, img)) * (1 + _amount(vibrance, img) * (1 - current_colourfulness))
    return grey + (img - grey) * boost


# ---------------------------------------------------------------------------
# Put it all together
# ---------------------------------------------------------------------------

def apply_edits(img: torch.Tensor, params: dict) -> torch.Tensor:
    """Apply every slider in a fixed order and return the edited photo.

    `params` maps slider name -> value between -1 and 1. Missing sliders count
    as 0. Values can be plain numbers, or one value per photo when editing a
    batch (that's what the AI will do during training).
    """
    unknown = set(params) - set(SLIDER_NAMES)
    if unknown:
        raise ValueError(f"Unknown sliders: {sorted(unknown)}")
    p = {name: params.get(name, 0.0) for name in SLIDER_NAMES}

    single = img.dim() == 3
    x = _as_batch(img)

    # 1. Light-based edits, done on linear light (see srgb_to_linear).
    x = srgb_to_linear(x)
    x = white_balance(x, p["temp"], p["tint"])
    x = exposure(x, p["exposure"])
    x = linear_to_srgb(x)
    # Anything pushed brighter than pure white is cut off here. The steps below
    # assume values between 0 and 1. (Real editors keep extra headroom so
    # highlights can be pulled back down; that's a possible future improvement.)
    x = x.clamp(0, 1)

    # 2. Tone edits: decide how bright or dark each part of the photo is.
    x = contrast(x, p["contrast"])
    x = highlights_shadows(x, p["highlights"], p["shadows"])
    x = whites_blacks(x, p["whites"], p["blacks"])

    # 3. Colour edits.
    x = vibrance_saturation(x, p["vibrance"], p["saturation"])

    x = x.clamp(0, 1)
    return x[0] if single else x
