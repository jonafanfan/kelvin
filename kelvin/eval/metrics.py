"""How close is an edited photo to the expert's? Two ways to put a number on it.

PSNR ("peak signal-to-noise ratio"), in decibels, higher is better
------------------------------------------------------------------
Take every pixel, measure how far its value is from the expert's, square
those gaps and average them (that average is the "mean squared error").
PSNR turns that into a friendlier scale: every +10 dB means the error
shrank 10x. Roughly: 20 dB is clearly different, 25 dB is close, 30+ dB is
hard to tell apart. It's the score most photo-editing papers report.

Delta E (written ΔE), lower is better
--------------------------------------
PSNR treats all errors alike, but our eyes don't: we notice a shift in skin
tone far more than the same-sized change in a dark corner. ΔE measures colour
difference in "Lab", a colour space designed so equal distances look equally
different to people. A ΔE of about 2 is just noticeable side by side; 10 is
obvious. We report the average over all pixels.

Both functions take photos shaped (batch, 3, H, W) with values from 0 to 1,
and return one score per photo.
"""

import torch

from kelvin.editor.ops import srgb_to_linear

# Linear sRGB -> XYZ (how the eye's sensors respond), for daylight-white (D65)
_RGB_TO_XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])  # XYZ of pure white


def psnr(edited: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = ((edited - target) ** 2).mean(dim=(1, 2, 3))
    return 10 * torch.log10(1 / mse.clamp_min(1e-10))  # clamp: identical photos would divide by 0


def srgb_to_lab(img: torch.Tensor) -> torch.Tensor:
    """sRGB (0 to 1) -> Lab: L = lightness 0..100, a = green(-) to red(+),
    b = blue(-) to yellow(+)."""
    xyz = torch.einsum("ij,bjhw->bihw", _RGB_TO_XYZ.to(img), srgb_to_linear(img))
    t = xyz / _WHITE.to(img).reshape(1, 3, 1, 1)
    # The eye responds to light on a curve (roughly a cube root), with a
    # straight bit near black.
    f = torch.where(t > (6 / 29) ** 3, t.clamp_min(1e-12) ** (1 / 3), t / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return torch.stack([L, a, b], dim=1)


def delta_e(edited: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gap = srgb_to_lab(edited) - srgb_to_lab(target)
    return gap.norm(dim=1).mean(dim=(1, 2))
