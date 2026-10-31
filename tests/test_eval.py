"""Checks that the scores mean what they say, using photos where we know the answer."""

import pytest
import torch

from kelvin.eval import delta_e, psnr
from kelvin.eval.baselines import best_sliders
from kelvin.eval.metrics import srgb_to_lab
from kelvin.editor import SLIDER_NAMES, apply_edits


def flat(r, g, b, size=8):
    """A photo that's one colour all over."""
    return torch.tensor([r, g, b], dtype=torch.float32).reshape(1, 3, 1, 1).expand(1, 3, size, size).clone()


def test_identical_photos_score_perfectly():
    img = torch.rand(2, 3, 16, 16)
    assert (psnr(img, img) >= 99).all()
    assert torch.allclose(delta_e(img, img), torch.zeros(2), atol=1e-4)


def test_psnr_of_a_known_error():
    # every pixel off by 0.1 -> squared error 0.01 -> 10 * log10(1 / 0.01) = 20 dB
    assert psnr(flat(0.5, 0.5, 0.5), flat(0.6, 0.6, 0.6)).item() == pytest.approx(20, abs=1e-3)


def test_lab_matches_published_values():
    """Reference values for sRGB white, black and pure red (CIE Lab, D65)."""
    assert srgb_to_lab(flat(1, 1, 1))[0, :, 0, 0].tolist() == pytest.approx([100, 0, 0], abs=0.1)
    assert srgb_to_lab(flat(0, 0, 0))[0, :, 0, 0].tolist() == pytest.approx([0, 0, 0], abs=0.1)
    assert srgb_to_lab(flat(1, 0, 0))[0, :, 0, 0].tolist() == pytest.approx([53.24, 80.09, 67.20], abs=0.1)


def test_black_and_white_are_100_apart():
    assert delta_e(flat(0, 0, 0), flat(1, 1, 1)).item() == pytest.approx(100, abs=0.1)


def test_bigger_mistakes_score_worse():
    target = torch.rand(1, 3, 16, 16)
    small, big = (target + 0.02).clamp(0, 1), (target + 0.2).clamp(0, 1)
    assert psnr(small, target) > psnr(big, target)
    assert delta_e(small, target) < delta_e(big, target)


def test_best_sliders_finds_a_known_edit():
    """Edit some photos with known sliders, then check the search gets back
    close to the target photo (the slider values themselves may differ:
    some sliders can stand in for each other)."""
    torch.manual_seed(0)
    before = torch.rand(4, 3, 32, 32) * 0.6 + 0.2
    known = torch.zeros(4, len(SLIDER_NAMES))
    known[:, SLIDER_NAMES.index("exposure")] = torch.tensor([0.3, -0.3, 0.1, 0.0])
    known[:, SLIDER_NAMES.index("temp")] = torch.tensor([0.2, 0.0, -0.4, 0.3])
    after = apply_edits(before, {n: known[:, i] for i, n in enumerate(SLIDER_NAMES)})

    found = best_sliders(before, after)
    assert found.shape == known.shape
    assert found.abs().max() <= 1
    edited = apply_edits(before, {n: found[:, i] for i, n in enumerate(SLIDER_NAMES)})
    assert (psnr(edited, after) > 35).all()
