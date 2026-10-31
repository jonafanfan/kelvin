"""The AI: looks at a photo and chooses a value for each of the 10 sliders.

What's inside
-------------
A "neural network" is a long chain of simple maths steps, each with its own
adjustable numbers (called "weights"). Training nudges those weights until
the chain's output is useful. Ours has two parts:

1. Eyes: MobileNet, a small network that has already learned to recognise
   what's in photos from a million examples (the "ImageNet" collection).
   Its early layers detect edges and colours, later ones textures and
   objects like skies, faces and foliage. We reuse all of that instead of
   starting from scratch, which would need far more than 4,000 photos. This
   is called "transfer learning", and the reused network is "pretrained".
   MobileNet boils each photo down to 576 numbers describing what it sees.

2. Decision: a small new network (the "head") that turns those 576 numbers,
   plus the photo's histograms, into 10 slider values. It starts from
   nothing and learns everything from our photos.

Why add the histograms?
-----------------------
MobileNet was trained to say "that's a dog" whether the photo is bright or
dark, warm or cool. So it learned to partly *ignore* exactly what we care
about. The histograms (the same brightness graph a human editor reads, plus
one per colour) hand the head that information directly. It's 64 numbers,
so it costs almost nothing.

Starting point
--------------
The head's last layer starts at zero, so before any training the AI sets
every slider to 0: "do nothing", the floor from step 3.2. Training then has
to climb from there.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from kelvin.editor import SLIDER_NAMES

EYES_SIDE = 224  # MobileNet learned on photos this size, so it sees best at it
EYES_FEATURES = 576  # how many numbers MobileNet boils a photo down to
HIST_BINS = 16  # brightness levels per histogram
HIST_FEATURES = 4 * HIST_BINS  # brightness, red, green and blue
HIDDEN = 128  # size of the head's middle layer

# MobileNet expects photos adjusted the way its training photos were: these
# are the average and spread of red, green and blue across ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def histograms(photo: torch.Tensor) -> torch.Tensor:
    """(batch, 3, H, W) photos -> (batch, 64): four 16-bar histograms
    (brightness, red, green, blue), each bar the share of pixels at that level."""
    small = F.interpolate(photo, size=64, mode="area")  # a rough count is all we need
    lum = 0.2126 * small[:, 0:1] + 0.7152 * small[:, 1:2] + 0.0722 * small[:, 2:3]
    channels = torch.cat([lum, small], dim=1).flatten(2)  # (batch, 4, pixels)
    bar = (channels * HIST_BINS).long().clamp(0, HIST_BINS - 1)
    counts = torch.zeros(len(photo), 4, HIST_BINS).scatter_add_(2, bar, torch.ones_like(channels))
    # Share of pixels, times the number of bars so a flat histogram is all 1s:
    # networks learn best when their inputs are around 1, not around 0.06.
    return (counts / channels.shape[-1] * HIST_BINS).flatten(1)


class SliderNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.eyes = mobilenet_v3_small(weights=weights).features
        self.head = nn.Sequential(
            nn.Linear(EYES_FEATURES + HIST_FEATURES, HIDDEN),
            nn.ReLU(),  # "keep positives, zero out negatives": lets the head learn non-straight-line rules
            nn.Linear(HIDDEN, len(SLIDER_NAMES)),
        )
        nn.init.zeros_(self.head[-1].weight)  # start at "do nothing" (see the top of this file)
        nn.init.zeros_(self.head[-1].bias)
        # register_buffer: saved with the model, but not something training adjusts
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1))

    def forward(self, photo: torch.Tensor) -> torch.Tensor:
        """(batch, 3, H, W) photos, values 0 to 1 -> (batch, 10) slider values, -1 to 1."""
        x = F.interpolate(photo, size=EYES_SIDE, mode="bilinear", antialias=True)
        x = (x - self.mean) / self.std
        # MobileNet's output is a small grid of 576 numbers per spot in the
        # photo. Sliders affect the whole photo, so we average over the spots.
        seen = self.eyes(x).mean(dim=(2, 3))
        both = torch.cat([seen, histograms(photo)], dim=1)
        # tanh squashes any number into -1...1, the sliders' range
        return torch.tanh(self.head(both))
