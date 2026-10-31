"""Put Kelvin online as a Hugging Face Space (a free web host for apps like this).

Once:   hf auth login         (paste a "write" token from huggingface.co/settings/tokens)
Then:   python -m kelvin.deploy
Rerun it whenever you want the website to get your latest changes.

What gets uploaded
------------------
Only what the app needs to run: app.py, the kelvin/ code, the app icon and the
trained AI's weights (checkpoints/slider_net.pt, ~4 MB, which aren't in git).
Not the FiveK photos, and not the tests.

The Space gets its own README (Spaces read their settings from the top of it)
and its own requirements list, which asks for the small CPU-only PyTorch:
the free Spaces have no graphics card, and the full PyTorch is ~2 GB bigger.
"""

import argparse
from pathlib import Path

import gradio
from huggingface_hub import CommitOperationAdd, HfApi

from kelvin.model.train import CHECKPOINT

SPACE_NAME = "kelvin"

SPACE_README = f"""---
title: Kelvin
emoji: 📷
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: {gradio.__version__}
python_version: "3.11"
app_file: app.py
pinned: false
---

An AI photo editor that edits like a person: it sets the same sliders a human
editor would, and says what each change did. Code: https://github.com/jonafanfan/kelvin
"""

SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
numpy
pillow
matplotlib
"""


def files_to_upload() -> list[Path]:
    code = [p for p in Path("kelvin").rglob("*.py") if "__pycache__" not in p.parts]
    return [Path("app.py"), *code, Path("kelvin/ui/icon.png"), CHECKPOINT]


def deploy(space: str | None) -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"No trained AI at {CHECKPOINT}. Train it first:  python -m kelvin.model.train")
    api = HfApi()
    space = space or f"{api.whoami()['name']}/{SPACE_NAME}"
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)

    uploads = [CommitOperationAdd(p.as_posix(), str(p)) for p in files_to_upload()]
    uploads += [
        CommitOperationAdd("README.md", SPACE_README.encode()),
        CommitOperationAdd("requirements.txt", SPACE_REQUIREMENTS.encode()),
    ]
    api.create_commit(space, repo_type="space", operations=uploads, commit_message="Update Kelvin")
    user, name = space.split("/")
    print(f"Uploaded {len(uploads)} files. It takes a few minutes to start up.")
    print(f"  Space page:        https://huggingface.co/spaces/{space}")
    print(f"  App on its own:    https://{user}-{name}.hf.space   (open this one to install it as an app)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", help='"username/name" (default: your username/kelvin)')
    deploy(parser.parse_args().space)
