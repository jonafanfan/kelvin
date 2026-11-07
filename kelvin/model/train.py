"""Teach SliderNet to set the sliders like Expert C.

Run it with:   python -m kelvin.model.train
Quick check:   python -m kelvin.model.train --epochs 1

How training works
------------------
It's the same loop as best_sliders in kelvin/eval/baselines.py, with one
big difference. There, we nudged 10 slider values per photo while looking
at the answer. Here, the AI must *choose* the sliders by looking at the
photo, and we nudge the AI's own 2.5 million weights so its choices get
better. Over and over, for a small pile ("batch") of training photos:
  1. The AI looks at each photo and picks 10 slider values.
  2. The editor applies them.
  3. We measure how far the result is from Expert C's edit (the "loss").
  4. PyTorch works out how every weight in the AI affected the loss, all
     the way back through the editor (the "gradient").
  5. Every weight is nudged a little in the direction that shrinks the loss.
One pass through all 4,000 training photos is an "epoch".

After every epoch, we score the AI on the 500 validation photos it never
learns from. That's how we spot "overfitting": if the training loss keeps
falling but the validation score gets worse, the AI has started memorising
its training photos instead of learning to edit. We keep whichever version
scored best on validation.

Phone photos
------------
Half the training photos are made to look like phone photos first (see
kelvin/data/phone.py), so the AI learns that a photo which already looks
finished needs less doing. Validation is scored twice, on the normal photos
and on phone-style ones, and the saved version is the one with the best
average of the two: we want an AI that's good at both, not just one.

Holding back
------------
The loss only cares about getting close to the expert. When the AI can't
tell what a photo needs, the safest bet for that loss is the *average*
edit, so it applies the average edit to everything (regression to the
mean). The "hold-back" penalty adds a small cost for every slider moved
away from 0, so when it's unsure, leaving a slider alone wins instead.

What gets saved (in checkpoints/, not in git)
---------------------------------------------
  slider_net.pt   the AI's weights, from the epoch with the best validation score
  slider_net_log.csv  loss and validation scores per epoch, to see how learning went
"""

import argparse
import csv
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kelvin.data import FiveK
from kelvin.editor import apply_edits, as_params
from kelvin.eval import delta_e, psnr

from .net import SliderNet

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT = CHECKPOINT_DIR / "slider_net.pt"

EPOCHS = 25
BATCH = 16
# The learning rate is the size of each nudge. The new head learns from
# scratch, so it takes bigger steps. MobileNet already knows a lot, so it
# takes small ones: big steps would wreck what it learned on ImageNet
# before it has a chance to adapt. ("Fine-tuning" means exactly this.)
HEAD_RATE = 1e-3
EYES_RATE = 1e-4
SEED = 0  # fixes every random choice, so a rerun gives the same result
PHONE_SHARE = 0.5  # of training photos made to look like phone photos
HOLD_BACK = 0.002  # cost per slider moved (squared), next to a typical loss of about 0.003


def loss_fn(model: SliderNet, before: torch.Tensor, after: torch.Tensor, hold_back: float = HOLD_BACK) -> torch.Tensor:
    """Mean squared error between our edit and the expert's (the same thing
    PSNR measures, before it's converted to decibels), plus the hold-back
    penalty (see the top of this file)."""
    sliders = model(before)
    edited = apply_edits(before, as_params(sliders))
    return ((edited - after) ** 2).mean() + hold_back * (sliders**2).mean()


def both(scores: dict) -> float:
    """The score used to pick the best epoch: normal and phone-style photos count equally."""
    return (scores["val_psnr"] + scores["phone_psnr"]) / 2


@torch.no_grad()  # scoring only: no nudging, so skip the gradient bookkeeping
def evaluate(model: SliderNet, loader: DataLoader) -> tuple[float, float]:
    """Average PSNR and Delta E over every photo the loader gives."""
    model.eval()
    total_psnr = total_de = count = 0.0
    for before, after in loader:
        edited = apply_edits(before, as_params(model(before)))
        total_psnr += psnr(edited, after).sum().item()
        total_de += delta_e(edited, after).sum().item()
        count += len(before)
    return total_psnr / count, total_de / count


def train(epochs: int = EPOCHS, hold_back: float = HOLD_BACK, checkpoint: Path = CHECKPOINT) -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    # Shuffling the training photos each epoch stops the AI from learning
    # anything from the order they come in. Two helper processes ("workers")
    # load the next photos while the current batch trains.
    train_loader = DataLoader(
        FiveK("train", phone=PHONE_SHARE), batch_size=BATCH, shuffle=True, num_workers=2, persistent_workers=True
    )
    val_loader = DataLoader(FiveK("val"), batch_size=50, num_workers=2, persistent_workers=True)
    phone_loader = DataLoader(FiveK("val", phone=1.0), batch_size=50, num_workers=2, persistent_workers=True)

    def validate() -> dict:
        val_psnr, val_de = evaluate(model, val_loader)
        phone_psnr, phone_de = evaluate(model, phone_loader)
        return {"val_psnr": val_psnr, "val_delta_e": val_de, "phone_psnr": phone_psnr, "phone_delta_e": phone_de}

    def report(scores: dict) -> str:
        return (
            f"validation {scores['val_psnr']:.2f} dB (phone-style {scores['phone_psnr']:.2f} dB), "
            f"Delta E {scores['val_delta_e']:.2f} ({scores['phone_delta_e']:.2f})"
        )

    model = SliderNet()
    optimiser = torch.optim.Adam(
        [
            {"params": model.eyes.parameters(), "lr": EYES_RATE},
            {"params": model.head.parameters(), "lr": HEAD_RATE},
        ]
    )
    # Take smaller and smaller steps as training goes on, so the AI settles
    # into a good spot instead of bouncing around it. "Cosine" is the shape:
    # gentle at first, then down to almost nothing by the last epoch.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    scores = validate()
    print(f"Before training (every slider at 0): {report(scores)}")
    log = [{"epoch": 0, "train_loss": ""} | scores]
    best = both(scores)

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()  # some layers act slightly differently while learning
        total_loss = 0.0
        for before, after in train_loader:
            loss = loss_fn(model, before, after, hold_back)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(before)
        schedule.step()
        train_loss = total_loss / len(train_loader.dataset)

        scores = validate()
        saved = ""
        if both(scores) > best:
            best = both(scores)
            torch.save(model.state_dict(), checkpoint)
            saved = "  <- best so far, saved"
        print(
            f"Epoch {epoch:>2}/{epochs}  training loss {train_loss:.5f}  {report(scores)}  "
            f"({time.time() - start:.0f}s){saved}",
            flush=True,
        )
        log.append({"epoch": epoch, "train_loss": train_loss} | scores)
        with checkpoint.with_name(checkpoint.stem + "_log.csv").open("w", newline="") as f:  # rewritten every epoch, so it's there even if training stops early
            writer = csv.DictWriter(f, fieldnames=log[0].keys())
            writer.writeheader()
            writer.writerows(log)

    print(f"\nBest validation score (average of both): {best:.2f} dB. Weights saved to {checkpoint}")
    print("Score it on the test photos with:  python -m kelvin.eval.baselines --model")


if __name__ == "__main__":  # needed on Windows, where the loader's workers re-run this file
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--hold-back", type=float, default=HOLD_BACK, help="0 turns the penalty off")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    train(args.epochs, args.hold_back, args.checkpoint)
