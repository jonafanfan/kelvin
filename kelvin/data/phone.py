"""Make a FiveK "before" photo look like it came out of a phone camera.

Why this exists
---------------
The AI learned from FiveK's "before" photos: flat, dull versions of RAW
files that nobody has edited yet. But people open phone photos, which the
phone has already edited: it fixed the brightness and colour, added
contrast and colour, and sharpened. The AI had never seen a photo like
that, so it edited them as if they were flat, and they got edited twice
(too much contrast, sunsets brightened out of their mood). In ML this is
called "distribution shift": the photos it's used on differ from the ones
it learned on.

The fix is to show it phone-style photos while it trains. We can't get
phone photos with expert edits, so we make them:
  1. Do part of the expert's edit already. Phones get brightness and colour
     roughly right, so we blend the before photo part of the way towards
     Expert C's version.
  2. Add the phone's own "punch": a bit of extra contrast and colour, a
     lifted shadow, and sharpening, more than the expert would use.
The answer stays Expert C's edit. So for these photos the AI learns that
little (or even the opposite) needs doing, and it has to look at the photo
to tell which kind it got. That's "data augmentation" again, like the
mirror flips in fivek.py, but this one teaches something new.
"""

import random

import torch
import torch.nn.functional as F

from kelvin.editor import apply_edits

# How far the phone gets towards the expert's edit by itself (0 = not at all, 1 = all the way)
HEAD_START = (0.3, 0.8)
# The phone's extra punch, as slider values: each one picked at random in its range
PUNCH = {
    "contrast": (0.0, 0.4),
    "shadows": (0.0, 0.4),
    "highlights": (-0.4, 0.0),
    "vibrance": (0.0, 0.4),
    "saturation": (0.0, 0.2),
}
SHARPEN = (0.0, 1.0)


def sharpen(photo: torch.Tensor, amount: float) -> torch.Tensor:
    """"Unsharp mask", the classic sharpening trick: find the fine detail
    (the photo minus a blurred copy of it) and add extra of it back."""
    blur = torch.tensor([1.0, 2.0, 1.0]) / 4
    kernel = (blur[:, None] * blur[None, :]).expand(3, 1, 3, 3)
    blurred = F.conv2d(F.pad(photo[None], (1, 1, 1, 1), mode="replicate"), kernel, groups=3)[0]
    return (photo + amount * (photo - blurred)).clamp(0, 1)


def phone_look(before: torch.Tensor, after: torch.Tensor, rng: random.Random = random) -> torch.Tensor:
    """(3, H, W) before and Expert C's after -> a phone-style version of before."""
    head_start = rng.uniform(*HEAD_START)
    photo = before + head_start * (after - before)
    punch = {name: rng.uniform(*span) for name, span in PUNCH.items()}
    photo = apply_edits(photo, punch)
    return sharpen(photo, rng.uniform(*SHARPEN))
