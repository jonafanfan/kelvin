"""Download MIT-Adobe FiveK (Expert C) and keep small copies of every photo.

Run it with:   python -m kelvin.data.prepare_fivek
Try it first:  python -m kelvin.data.prepare_fivek --chunks 1

Where the photos come from
--------------------------
The small (480p) copy most research papers use is no longer online, so we use
a full-size copy on Hugging Face (logasja/mit-adobe-fivek). It's huge: about
124 GB for Expert C, split into 64 "chunks" of about 2 GB. So we never keep it:
for each chunk we download it, shrink every photo in it, save the small
copies, then delete the chunk. Only one chunk is ever on disk.

Each photo comes as a pair:
  - "original":  the photo before editing (rendered from the camera's RAW file)
  - "augmented": the same photo after Expert C edited it in Lightroom

What we save
------------
  data/fivek/input/<name>.jpg    the photo before editing
  data/fivek/expertC/<name>.jpg  Expert C's edit
Both are sRGB JPEGs, at most SAVE_SIDE pixels on the longest side.

It's safe to stop and rerun: chunks that are already done are skipped.
"""

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageCms

REPO = "logasja/mit-adobe-fivek"
EXPERT = "c"

DATA_DIR = Path("data/fivek")

# Training will use smaller copies still (around 256 pixels). Saving at 512
# leaves room to try bigger sizes later without downloading 124 GB again.
SAVE_SIDE = 512
JPEG_QUALITY = 95  # high enough that JPEG blockiness doesn't teach the AI anything

# Every pair's match score (see match_score) goes in this file, one
# "name,score" line each, so badly matched pairs can be found and left out.
MATCHES_FILE = "matches.csv"
MATCH_SIDE = 128  # matching only needs a rough look, so it uses tiny copies

_SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def to_srgb(img: Image.Image) -> Image.Image:
    """Expert C's photos are stored in "ProPhoto RGB", a colour space with
    more colours than a screen can show. If we read them as if they were
    normal sRGB, every photo looks dull and grey, and the AI would learn that
    experts make photos dull. So we convert, using the profile in the file.
    Photos without a profile are already sRGB."""
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
    show it". Lightroom followed the note for the expert's version, but the
    originals in this copy ignore it, so some come out on their side (and
    occasionally upside down). Which way to turn depends on the camera, so we
    just try all four and keep the one that lines up best with the edit."""
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


def list_chunks() -> list[str]:
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith(f"{EXPERT}/") and f.endswith(".parquet"))


def process_chunk(chunk: str, out: Path) -> int:
    """Download one chunk, save small copies of every photo in it, delete the chunk."""
    local = Path(hf_hub_download(REPO, chunk, repo_type="dataset", local_dir=out / "_download"))
    count = 0
    try:
        table = pq.read_table(local, columns=["original", "augmented"])
        with (out / MATCHES_FILE).open("a") as matches:
            for row in table.to_pylist():
                # e.g. "a3660-jmac_MG_8044.webp" -> "a3660-jmac_MG_8044"
                name = Path(row["augmented"]["path"]).stem
                score = save_pair(row["original"]["bytes"], row["augmented"]["bytes"], name, out)
                matches.write(f"{name},{score:.3f}\n")
                count += 1
    finally:
        local.unlink()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=int, help="only do this many chunks (to test the script)")
    args = parser.parse_args()

    for folder in ("input", "expertC"):
        (DATA_DIR / folder).mkdir(parents=True, exist_ok=True)
    done_file = DATA_DIR / "chunks_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()

    chunks = list_chunks()
    todo = [c for c in chunks if c not in done][: args.chunks]
    print(f"{len(chunks)} chunks in total, {len(done)} already done, doing {len(todo)} now.")
    for i, chunk in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {chunk} ...", end=" ", flush=True)
        count = process_chunk(chunk, DATA_DIR)
        with done_file.open("a") as f:
            f.write(chunk + "\n")
        print(f"{count} photos")

    total = len(list((DATA_DIR / "expertC").glob("*.jpg")))
    print(f"Done. {total} photo pairs in {DATA_DIR}.")


if __name__ == "__main__":
    main()
