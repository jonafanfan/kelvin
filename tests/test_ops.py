"""Automatic checks that the editor behaves sensibly.

Run them with:   python -m pytest
Each test is a small promise about how a slider should behave. If someone
changes the maths later and breaks a promise, the test fails and tells them.
"""

import pytest
import torch

from kelvin.editor import SLIDER_NAMES, apply_edits, luminance, neutral_params


@pytest.fixture
def photo():
    # A fixed random "photo" so every run of the tests sees the same pixels.
    torch.manual_seed(0)
    return torch.rand(3, 32, 32)


def test_all_sliders_at_zero_leave_photo_unchanged(photo):
    edited = apply_edits(photo, neutral_params())
    assert torch.allclose(edited, photo, atol=1e-4)


@pytest.mark.parametrize("name", SLIDER_NAMES)
@pytest.mark.parametrize("value", [-1.0, 1.0])
def test_every_slider_does_something_and_stays_in_range(photo, name, value):
    edited = apply_edits(photo, {name: value})
    assert not torch.allclose(edited, photo, atol=1e-3), f"{name} had no effect"
    assert edited.min() >= 0 and edited.max() <= 1


def test_exposure_up_brightens(photo):
    assert apply_edits(photo, {"exposure": 0.5}).mean() > photo.mean()


def test_warmth_up_makes_photo_more_orange(photo):
    edited = apply_edits(photo, {"temp": 0.5})
    assert edited[0].mean() > photo[0].mean()  # more red
    assert edited[2].mean() < photo[2].mean()  # less blue


def test_shadows_change_dark_pixels_more_than_bright_ones():
    dark = torch.full((3, 4, 4), 0.25)
    bright = torch.full((3, 4, 4), 0.85)
    dark_change = (apply_edits(dark, {"shadows": 1.0}) - dark).mean()
    bright_change = (apply_edits(bright, {"shadows": 1.0}) - bright).mean()
    assert dark_change > bright_change > -1e-6


def test_saturation_down_fully_makes_grey(photo):
    edited = apply_edits(photo, {"saturation": -1.0})
    # In a grey pixel, red, green and blue are all equal.
    assert torch.allclose(edited[0], edited[1], atol=1e-5)
    assert torch.allclose(edited[1], edited[2], atol=1e-5)


def test_batch_with_different_values_per_photo(photo):
    batch = torch.stack([photo, photo])
    edited = apply_edits(batch, {"exposure": torch.tensor([-0.5, 0.5])})
    assert luminance(edited[0:1]).mean() < luminance(edited[1:2]).mean()


def test_gradients_reach_every_slider(photo):
    """The most important test for the AI step: PyTorch must be able to work
    out how each slider affects the result, or the AI can't learn to use it."""
    params = {name: torch.tensor(0.1, requires_grad=True) for name in SLIDER_NAMES}
    # Keep values away from 0 and 1 so nothing gets cut off at the edges.
    apply_edits(photo * 0.6 + 0.2, params).mean().backward()
    for name, value in params.items():
        assert value.grad is not None and torch.isfinite(value.grad), name
        assert value.grad.abs() > 0, f"{name} has no effect the AI could learn from"


def test_unknown_slider_name_is_an_error(photo):
    with pytest.raises(ValueError):
        apply_edits(photo, {"brightnes": 0.5})
