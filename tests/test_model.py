"""Checks for the AI. They use an untrained MobileNet (pretrained=False), so
they run without downloading its weights."""

import torch

from kelvin.editor import SLIDER_NAMES, apply_edits, as_params
from kelvin.model import SliderNet
from kelvin.model.net import HIST_BINS, histograms
from kelvin.model.train import loss_fn


def test_one_value_per_slider_in_range():
    model = SliderNet(pretrained=False)
    values = model(torch.rand(3, 3, 64, 80))  # any size and shape of photo works
    assert values.shape == (3, len(SLIDER_NAMES))
    assert values.abs().max() <= 1


def test_untrained_model_does_nothing():
    """The head's last layer starts at zero, so training starts from the floor."""
    model = SliderNet(pretrained=False)
    photos = torch.rand(2, 3, 32, 32)
    assert torch.equal(model(photos), torch.zeros(2, len(SLIDER_NAMES)))


def test_histograms_count_every_pixel():
    photos = torch.rand(2, 3, 64, 64)
    h = histograms(photos).reshape(2, 4, HIST_BINS)
    # each histogram's bars are shares times HIST_BINS, so they add up to HIST_BINS
    assert torch.allclose(h.sum(dim=2), torch.full((2, 4), float(HIST_BINS)))


def test_histograms_see_brightness():
    dark, bright = torch.full((1, 3, 64, 64), 0.1), torch.full((1, 3, 64, 64), 0.9)
    lum_dark, lum_bright = histograms(dark)[0, :HIST_BINS], histograms(bright)[0, :HIST_BINS]
    assert lum_dark.argmax() < lum_bright.argmax()


def test_one_training_step_reduces_the_loss():
    """The whole chain must be learnable: from the loss, back through the
    editor and the head, into MobileNet itself."""
    torch.manual_seed(0)
    model = SliderNet(pretrained=False)
    before = torch.rand(4, 3, 64, 64) * 0.5 + 0.1
    target = torch.zeros(4, len(SLIDER_NAMES))
    target[:, SLIDER_NAMES.index("exposure")] = 0.4  # the "expert" brightened every photo
    after = apply_edits(before, as_params(target))

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = loss_fn(model, before, after)
    optimiser.zero_grad()
    first.backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    optimiser.step()
    for _ in range(20):  # a few more steps: after the first, gradients reach MobileNet too
        loss = loss_fn(model, before, after)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.eyes.parameters())
    assert loss_fn(model, before, after) < first / 2
