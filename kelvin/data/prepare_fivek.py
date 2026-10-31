"""Download MIT-Adobe FiveK (Expert C) and keep small copies of every photo.

Run it with:   python -m kelvin.data.prepare_fivek

Where the photos come from
--------------------------
The authors of UEGAN (a photo-editing AI paper) exported all 5,000 FiveK
photos from Lightroom, 512 pixels on the short edge, and shared them as two
zip files (about 6 GB together):
  - Original.zip: each photo before editing. This is FiveK's "as shot,
    zeroed" version: the camera's photo with every Lightroom slider at 0.
  - expertC.zip:  the same photos after Expert C edited them
https://github.com/eezkni/UEGAN#preparing-data-for-the-mit-adobe-fivek-dataset

(The smaller 480p copy many papers use is no longer online, and the copies on
Hugging Face are full size: 124 GB for Expert C.)

What we save
------------
  data/downloads/*.zip           the two zips, kept so we can redo this
                                 without downloading again
  data/fivek/input/<name>.jpg    the photo before editing
  data/fivek/expertC/<name>.jpg  Expert C's edit
  data/fivek/matches.csv         how well each pair lines up (see match_score)
Both photos are sRGB JPEGs, at most SAVE_SIDE pixels on the longest side.
"""

import argparse
import io
import zipfile
from pathlib import Path

import gdown
import numpy as np
from PIL import Image, ImageCms

# Google Drive file IDs of the two zips
ZIPS = {
    "Original.zip": "1bzFR3PGpGgFlzP_FJwcKdl2ZCmKJDcrV",
    "expertC.zip": "12LXYca6aeYXdgj8IeW8qCOOjB5tyfxx5",
}

DOWNLOAD_DIR = Path("data/downloads")
DATA_DIR = Path("data/fivek")

# Training will use smaller copies still (around 256 pixels). Saving at 512
# leaves room to try bigger sizes later.
SAVE_SIDE = 512
JPEG_QUALITY = 95  # high enough that JPEG blockiness doesn't teach the AI anything

# Every pair's match score (see match_score) goes in this file, one
# "name,score" line each, so badly matched pairs can be found and left out.
MATCHES_FILE = "matches.csv"
MATCH_SIDE = 128  # matching only needs a rough look, so it uses tiny copies

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def to_srgb(img: Image.Image) -> Image.Image:
    """Photos can be stored in different colour spaces, and the file's colour
    profile says which. These zips are already sRGB, but other copies of FiveK
    store Expert C in "ProPhoto RGB", which looks dull and grey if read as if
    it were sRGB, and the AI would learn that experts make photos dull. So we
    always convert using the profile. Photos without a profile are taken as sRGB."""
    icc = img.info.get("icc_profile")
    if not icc:
        return img.convert("RGB")
    profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    return ImageCms.profileToProfile(img, profile, _SRGB, outputMode="RGB")


def centre_crop(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """The original is a few pixels bigger than the edit (camera RAW files keep
    a thin border that Lightroom trims off). Cutting the same amount off each
    side lines the two photos up again."""
    w, h = img.size
    left, top = (w - size[0]) // 2, (h - size[1]) // 2
    return img.crop((left, top, left + size[0], top + size[1]))


def _edges(img: Image.Image) -> np.ndarray:
    """A small map of where the edges are, scaled so the average is 0.
    We compare edges rather than colours because the expert's edit changes
    colour and brightness, but never where the edges are."""
    a = np.asarray(img.convert("L").resize((MATCH_SIDE, MATCH_SIDE)), dtype=np.float32)
    e = np.abs(np.diff(a, axis=0))[:, :-1] + np.abs(np.diff(a, axis=1))[:-1]
    return (e - e.mean()) / (e.std() + 1e-6)


def match_score(a: Image.Image, b: Image.Image) -> float:
    """How well two photos line up: about 1 for the same picture, about 0 for
    unrelated pictures."""
    return float(np.mean(_edges(a) * _edges(b)))


def upright(original: Image.Image, edited: Image.Image) -> tuple[Image.Image, float]:
    """Cameras store photos sideways and add a note saying "turn this when you
    show it". Some copies of FiveK ignore the note for the originals, so they
    come out on their side. These zips don't (Lightroom exported both), but
    the check is cheap, so we keep it: try all four turns and keep the one
    that lines up best with the edit."""
    best_score, best = -2.0, original
    for turns in range(4):
        turned = original.rotate(90 * turns, expand=True) if turns else original
        if (turned.width > turned.height) != (edited.width > edited.height):
            continue  # wrong shape, can't be it
        if turned.width < edited.width or turned.height < edited.height:
            continue
        candidate = centre_crop(turned, edited.size)
        score = match_score(candidate, edited)
        if score > best_score:
            best_score, best = score, candidate
    if best_score == -2.0:  # no turn fits (shouldn't happen): stretch it, the score will say it's bad
        best = original.resize(edited.size)
        best_score = match_score(best, edited)
    return best, best_score


def shrink(img: Image.Image) -> Image.Image:
    img = img.copy()
    img.thumbnail((SAVE_SIDE, SAVE_SIDE), Image.Resampling.LANCZOS)
    return img


def save_pair(original_bytes: bytes, edited_bytes: bytes, name: str, out: Path) -> float:
    """Save small, lined-up copies of one pair. Returns how well they match."""
    edited = to_srgb(Image.open(io.BytesIO(edited_bytes)))
    original = to_srgb(Image.open(io.BytesIO(original_bytes)))
    original, score = upright(original, edited)
    for folder, img in (("input", original), ("expertC", edited)):
        shrink(img).save(out / folder / f"{name}.jpg", quality=JPEG_QUALITY)
    return score


def download() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for name, file_id in ZIPS.items():
        path = DOWNLOAD_DIR / name
        if path.exists():
            print(f"{name}: already downloaded")
            continue
        print(f"{name}: downloading (about 3 GB)...")
        gdown.download(id=file_id, output=str(path), resume=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only do this many photos (to test the script)")
    args = parser.parse_args()

    download()
    for folder in ("input", "expertC"):
        (DATA_DIR / folder).mkdir(parents=True, exist_ok=True)

    originals = zipfile.ZipFile(DOWNLOAD_DIR / "Original.zip")
    edits = zipfile.ZipFile(DOWNLOAD_DIR / "expertC.zip")
    names = sorted(set(originals.namelist()) & set(edits.namelist()))[: args.limit]
    print(f"Saving {len(names)} photo pairs to {DATA_DIR} ...")
    with (DATA_DIR / MATCHES_FILE).open("w") as matches:
        for i, filename in enumerate(names, 1):
            name = Path(filename).stem  # "a0001-jmac_DSC1459.png" -> "a0001-jmac_DSC1459"
            score = save_pair(originals.read(filename), edits.read(filename), name, DATA_DIR)
            matches.write(f"{name},{score:.3f}\n")
            if i % 500 == 0:
                print(f"  {i}/{len(names)}")
    print("Done.")


if __name__ == "__main__":
    main()
