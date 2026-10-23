"""Automatic checks for the rule-based Auto button.

Each test builds a photo with one obvious problem and checks that Auto moves
the right slider in the right direction (and explains why).
"""

import pytest
import torch

from kelvin.auto import suggest
from kelvin.editor import SLIDER_NAMES, apply_edits, luminance


def gradient_photo(low=0.0, high=1.0, size=64):
    """A grey photo that goes smoothly from `low` brightness to `high`,
    with a little colour so the colour rules have something to look at."""
    ramp = torch.linspace(low, high, size).repeat(size, 1)
    photo = torch.stack([ramp, ramp, ramp])
    torch.manual_seed(0)
    return (photo + 0.1 * (torch.rand(3, size, size) - 0.5)).clamp(0, 1)


def test_suggestion_covers_every_slider_within_range():
    s = suggest(gradient_photo())
    assert set(s.params) == set(SLIDER_NAMES)
    assert all(-1 <= v <= 1 for v in s.params.values())


def test_every_moved_slider_has_a_reason_and_no_others_do():
    s = suggest(gradient_photo(0.0, 0.4))
    moved = {name for name, v in s.params.items() if v != 0}
    assert moved, "a dark photo should get some changes"
    assert set(s.reasons) == moved
    assert all(reason.strip() for reason in s.reasons.values())


def test_dark_photo_gets_brighter():
    photo = gradient_photo(0.0, 0.35)
    s = suggest(photo)
    assert s.params["exposure"] > 0
    assert luminance(apply_edits(photo, s.params)[None]).median() > luminance(photo[None]).median()


def test_bright_photo_gets_darker():
    assert suggest(gradient_photo(0.6, 1.0)).params["exposure"] < 0


def test_orange_photo_gets_cooler_and_blue_photo_gets_warmer():
    photo = gradient_photo(0.2, 0.8)
    orange = photo * torch.tensor([1.0, 0.9, 0.78]).reshape(3, 1, 1)
    blue = photo * torch.tensor([0.78, 0.9, 1.0]).reshape(3, 1, 1)
    assert suggest(orange).params["temp"] < 0
    assert suggest(blue).params["temp"] > 0


def test_photo_with_nothing_grey_keeps_its_colours():
    """A field of green grass says nothing about the colour of the light,
    so Auto shouldn't try to "fix" it into grey."""
    photo = gradient_photo(0.2, 0.8) * torch.tensor([0.3, 1.0, 0.2]).reshape(3, 1, 1)
    s = suggest(photo)
    assert s.params["temp"] == 0 and s.params["tint"] == 0


def test_flat_photo_gets_more_contrast():
    assert suggest(gradient_photo(0.35, 0.6)).params["contrast"] > 0


def test_washed_out_photo_gets_blacks_and_whites_stretched():
    s = suggest(gradient_photo(0.25, 0.75))
    assert s.params["blacks"] < 0  # the darkest point moves down to black
    assert s.params["whites"] > 0  # the brightest point moves up to white


def test_dull_photo_gets_more_vibrance():
    photo = gradient_photo(0.2, 0.8)
    grey = luminance(photo[None])[0]
    dull = grey + 0.2 * (photo - grey)
    assert suggest(dull).params["vibrance"] > 0


@pytest.mark.parametrize("low,high", [(0.0, 1.0), (0.05, 0.95)])
def test_well_exposed_photo_is_mostly_left_alone(low, high):
    s = suggest(gradient_photo(low, high))
    assert abs(s.params["exposure"]) < 0.2
    assert s.params["contrast"] == 0


def test_large_photo_works():
    torch.manual_seed(0)
    s = suggest(torch.rand(3, 900, 1200) * 0.4)
    assert s.params["exposure"] > 0
