"""Score simple methods on the 500 FiveK test photos: the scores to beat.

Run it with:   python -m kelvin.eval.baselines
(Needs the photos first:  python -m kelvin.data.prepare_fivek)

The methods
-----------
  nothing  Leave the photo alone. The floor: anything worse than this is
           making photos worse.
  auto     The rule-based Auto button from step 2. The score our AI must beat.
  model    (with --model) the trained AI from kelvin.model.train. It only
           sees the photo, never the answer.
  best     For each photo, search for the slider values that bring it closest
           to Expert C's edit, *while looking at the answer*. No real method
           can do that, so it's the ceiling: the best any slider-setting AI
           could ever score. The gap between "best" and a perfect score is
           what our 10 sliders can't express (the expert had all of Lightroom).

Results go to data/results/baselines.csv, one row per photo per method, so
the worst photos can be looked at later.
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kelvin.auto import suggest
from kelvin.data import FiveK
from kelvin.editor import SLIDER_NAMES, apply_edits, as_params

from .metrics import delta_e, psnr

RESULTS = Path("data/results/baselines.csv")
BATCH = 50

# Settings for the "best" search (see best_sliders)
SEARCH_SIDE = 128  # slider values don't depend on photo size, so search on small copies: 4x faster
SEARCH_STEPS = 200
SEARCH_RATE = 0.05


def nothing(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    return torch.zeros(len(before), len(SLIDER_NAMES))


def auto(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    rows = []
    for photo in before:
        params = suggest(photo).params
        rows.append([params[name] for name in SLIDER_NAMES])
    return torch.tensor(rows)


def best_sliders(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Find the slider values that bring each photo closest to the expert's.

    This is "gradient descent", the same idea that will train our AI:
      1. Start with every slider at 0.
      2. Edit the photo and measure the error against the expert's version.
      3. PyTorch works out, for each slider, which way to nudge it to shrink
         the error (that's the "gradient").
      4. Nudge every slider a little that way. Repeat.
    Here we nudge 10 numbers per photo. Training nudges the millions of
    numbers inside the AI instead, so it learns to set sliders for photos
    it has never seen.

    The sliders must stay between -1 and 1, so we search over any number
    ("raw") and squash it into that range with tanh.
    """
    small_before = F.interpolate(before, size=SEARCH_SIDE, mode="area")
    small_after = F.interpolate(after, size=SEARCH_SIDE, mode="area")
    raw = torch.zeros(len(before), len(SLIDER_NAMES), requires_grad=True)
    # Adam is a popular way to do the nudging: it adjusts the step size for
    # each slider as it goes, so it doesn't need careful tuning.
    optimiser = torch.optim.Adam([raw], lr=SEARCH_RATE)
    for _ in range(SEARCH_STEPS):
        edited = apply_edits(small_before, as_params(torch.tanh(raw)))
        error = ((edited - small_after) ** 2).mean()
        optimiser.zero_grad()
        error.backward()
        optimiser.step()
    return torch.tanh(raw).detach()


METHODS = {"nothing": nothing, "auto": auto, "best": best_sliders}


def trained_model(checkpoint: Path):
    """The AI from kelvin.model.train, as a method: it only looks at `before`."""
    from kelvin.model import SliderNet  # imported here: the other methods don't need torchvision

    model = SliderNet(pretrained=False)  # the checkpoint replaces every weight anyway
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    return lambda before, after: model(before)


def main(methods: dict = METHODS) -> None:
    test = FiveK("test")
    loader = DataLoader(test, batch_size=BATCH, shuffle=False)
    rows = []
    for batch_number, (before, after) in enumerate(loader):
        names = test.names[batch_number * BATCH : (batch_number + 1) * BATCH]
        print(f"Photos {batch_number * BATCH + 1}-{batch_number * BATCH + len(before)} of {len(test)} ...", flush=True)
        for method, choose in methods.items():
            with torch.set_grad_enabled(method == "best"):  # only the search needs gradients
                values = choose(before, after)
            with torch.no_grad():
                edited = apply_edits(before, as_params(values))
                scores = zip(psnr(edited, after).tolist(), delta_e(edited, after).tolist())
            for name, (p, e), v in zip(names, scores, values.tolist()):
                rows.append({"photo": name, "method": method, "psnr": p, "delta_e": e} | dict(zip(SLIDER_NAMES, v)))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows, list(methods))


def print_summary(rows: list[dict], methods: list[str]) -> None:
    by_photo = {(r["photo"], r["method"]): r for r in rows}
    photos = sorted({r["photo"] for r in rows})
    print(f"\nScores on {len(photos)} test photos (compared with Expert C's edits):\n")
    print(f"{'method':<10}{'PSNR (higher=better)':>22}{'Delta E (lower=better)':>24}{'better than nothing':>22}")
    for method in methods:
        mine = [by_photo[(p, method)] for p in photos]
        mean_psnr = sum(r["psnr"] for r in mine) / len(mine)
        mean_de = sum(r["delta_e"] for r in mine) / len(mine)
        wins = sum(r["psnr"] > by_photo[(r["photo"], "nothing")]["psnr"] for r in mine) / len(mine)
        print(f"{method:<10}{mean_psnr:>19.2f} dB{mean_de:>24.2f}{wins:>21.0%}")
    print(f"\nEvery photo's scores and slider values: {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", nargs="?", const="checkpoints/slider_net.pt", help="also score the trained AI (default: %(const)s)"
    )
    args = parser.parse_args()
    methods = dict(METHODS)
    if args.model:  # listed before "best", so the table reads floor, baseline, AI, ceiling
        methods = {"nothing": nothing, "auto": auto, "model": trained_model(Path(args.model)), "best": best_sliders}
    main(methods)
