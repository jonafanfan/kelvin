"""Save the trained AI in a format web browsers can run (step 6: the web app).

Run it with:   python -m kelvin.model.export

ONNX is a standard file format for trained AIs. PyTorch writes the whole
"recipe" (every step of the forward pass, with the trained weights) into one
file, and a browser library called ONNX Runtime Web can then run it, with no
Python or PyTorch involved. That's what lets the web app work offline.

The browser always hands the AI a 256 x 256 photo, the same size and squashed
shape it saw in training. A fixed size keeps the file simple.
"""

from pathlib import Path

import torch

from .net import load_trained
from .train import CHECKPOINT

PHOTO_SIDE = 256
OUTPUT = Path("web/public/slider_net.onnx")


def export(checkpoint: Path = CHECKPOINT, output: Path = OUTPUT) -> None:
    model = load_trained(checkpoint)
    example = torch.rand(1, 3, PHOTO_SIDE, PHOTO_SIDE)  # any photo works: it only shows PyTorch the shapes
    torch.onnx.export(
        model, (example,), output, input_names=["photo"], output_names=["sliders"],
        dynamo=True, opset_version=18, external_data=False, verbose=False,
    )
    print(f"Saved {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    export()
