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

What gets saved (in checkpoints/, not in git)
---------------------------------------------
  slider_net.pt   the AI's weights, from the epoch with the best validation score
  train_log.csv   loss and validation scores per epoch, to see how learning went
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
LOG = CHECKPOINT_DIR / "train_log.csv"

EPOCHS = 25
BATCH = 16
# The learning rate is the size of each nudge. The new head learns from
# scratch, so it takes bigger steps. MobileNet already knows a lot, so it
# takes small ones: big steps would wreck what it learned on ImageNet
# before it has a chance to adapt. ("Fine-tuning" means exactly this.)
HEAD_RATE = 1e-3
EYES_RATE = 1e-4
SEED = 0  # fixes every random choice, so a rerun gives the same result


def loss_fn(model: SliderNet, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Mean squared error between our edit and the expert's: the same thing
    PSNR measures, before it's converted to decibels."""
    edited = apply_edits(before, as_params(model(before)))
    return ((edited - after) ** 2).mean()


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


def train(epochs: int = EPOCHS) -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    # Shuffling the training photos each epoch stops the AI from learning
    # anything from the order they come in. Two helper processes ("workers")
    # load the next photos while the current batch trains.
    train_loader = DataLoader(FiveK("train"), batch_size=BATCH, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(FiveK("val"), batch_size=50, num_workers=2, persistent_workers=True)

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
    val_psnr, val_de = evaluate(model, val_loader)
    print(f"Before training (every slider at 0): validation {val_psnr:.2f} dB, Delta E {val_de:.2f}")
    log = [{"epoch": 0, "train_loss": "", "val_psnr": val_psnr, "val_delta_e": val_de}]
    best = val_psnr

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()  # some layers act slightly differently while learning
        total_loss = 0.0
        for before, after in train_loader:
            loss = loss_fn(model, before, after)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(before)
        schedule.step()
        train_loss = total_loss / len(train_loader.dataset)

        val_psnr, val_de = evaluate(model, val_loader)
        saved = ""
        if val_psnr > best:
            best = val_psnr
            torch.save(model.state_dict(), CHECKPOINT)
            saved = "  <- best so far, saved"
        print(
            f"Epoch {epoch:>2}/{epochs}  training loss {train_loss:.5f}  "
            f"validation {val_psnr:.2f} dB, Delta E {val_de:.2f}  ({time.time() - start:.0f}s){saved}",
            flush=True,
        )
        log.append({"epoch": epoch, "train_loss": train_loss, "val_psnr": val_psnr, "val_delta_e": val_de})
        with LOG.open("w", newline="") as f:  # rewritten every epoch, so it's there even if training stops early
            writer = csv.DictWriter(f, fieldnames=log[0].keys())
            writer.writeheader()
            writer.writerows(log)

    print(f"\nBest validation score: {best:.2f} dB. Weights saved to {CHECKPOINT}")
    print("Score it on the test photos with:  python -m kelvin.eval.baselines --model")


if __name__ == "__main__":  # needed on Windows, where the loader's workers re-run this file
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    train(parser.parse_args().epochs)
