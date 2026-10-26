"""Hand FiveK photo pairs to PyTorch, split into training and test photos.

A "Dataset" is PyTorch's name for "a numbered list of examples". Ask it for
example number 7 and it gives you one (before, after) pair of photos, ready
to use. Training asks for thousands of these, in a random order.

Training photos and test photos
-------------------------------
The photos are named a0001 ... a5000. Like most research papers, we learn from
the first 4,500 and test on the last 500, which the AI never sees while it
learns. That's how we can tell real skill from memorising: a student who has
only memorised the past papers does badly on a new exam.

Run the download first:   python -m kelvin.data.prepare_fivek
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .prepare_fivek import DATA_DIR, MATCHES_FILE

TRAIN_LAST = 4500  # a0001 ... a4500 are for training, a4501 ... a5000 for testing

# Pairs whose before and after don't line up are left out (see
# prepare_fivek.match_score). Real pairs score 0.68 or more, even when the
# expert changed the photo a lot, so 0.5 only catches genuine mix-ups.
MIN_MATCH = 0.5


def photo_number(name: str) -> int:
    """'a0042-IMG_1234' -> 42"""
    return int(name[1:5])


def badly_matched(data_dir: Path) -> set[str]:
    path = data_dir / MATCHES_FILE
    if not path.exists():
        return set()
    rows = (line.split(",") for line in path.read_text().splitlines() if line)
    return {name for name, score in rows if float(score) < MIN_MATCH}


def split_names(split: str, data_dir: Path = DATA_DIR) -> list[str]:
    """The names of every photo pair in one split ("train" or "test")."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', not {split!r}")
    names = sorted(p.stem for p in (data_dir / "expertC").glob("*.jpg"))
    if not names:
        raise FileNotFoundError(f"No photos in {data_dir}. Run: python -m kelvin.data.prepare_fivek")
    in_train = split == "train"
    skip = badly_matched(data_dir)
    return [n for n in names if (photo_number(n) <= TRAIN_LAST) == in_train and n not in skip]


class FiveK(Dataset):
    """Each example is (before, after): two tensors shaped (3, size, size)
    with values from 0 to 1, in the format kelvin.editor expects.

    Every photo is squashed to a square. That would be wrong for most image
    AI, but our sliders work on the whole photo at once (they don't care
    where a pixel is), so a squashed photo needs the same slider values as
    the real one. Squares are needed so photos can be stacked into a batch.
    """

    def __init__(self, split: str, size: int = 256, data_dir: Path = DATA_DIR):
        self.names = split_names(split, data_dir)
        self.size = size
        self.data_dir = data_dir
        # Only training photos get flipped. Test photos must be identical
        # every time, or scores from different days couldn't be compared.
        self.flip = split == "train"

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[i]
        before = self._load(self.data_dir / "input" / f"{name}.jpg")
        after = self._load(self.data_dir / "expertC" / f"{name}.jpg")
        # A mirror image needs exactly the same edit, so flipping at random is
        # a free way to show the AI "new" photos. It's called data augmentation.
        if self.flip and random.random() < 0.5:
            before, after = before.flip(-1), after.flip(-1)
        return before, after

    def _load(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
