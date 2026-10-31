"""Checks for the AI version of Auto. They use a stand-in "AI" that always
picks the same sliders, so they run without the trained weights."""

import re

import torch

from kelvin.auto import ai_suggest
from kelvin.editor import SLIDER_NAMES
from kelvin.model import SliderNet, load_trained


def fake_ai(**sliders):
    values = torch.tensor([[sliders.get(name, 0.0) for name in SLIDER_NAMES]])
    return lambda photos: values.expand(len(photos), -1)


def photo(size=64):
    torch.manual_seed(0)
    return torch.rand(3, size, size) * 0.5 + 0.1


def test_uses_the_ai_values_and_drops_tiny_ones():
    s = ai_suggest(photo(), fake_ai(exposure=0.304, contrast=-0.2, vibrance=0.02))
    assert s.params["exposure"] == 0.3
    assert s.params["contrast"] == -0.2
    assert s.params["vibrance"] == 0  # too small to notice
    assert set(s.params) == set(SLIDER_NAMES)


def test_every_moved_slider_is_explained_and_no_others():
    s = ai_suggest(photo(), fake_ai(temp=-0.3, exposure=0.4, shadows=0.5, saturation=-0.2))
    assert set(s.reasons) == {"temp", "exposure", "shadows", "saturation"}
    assert "cooler" in s.reasons["temp"]
    assert "greyer" in s.reasons["saturation"]


def test_explanations_measure_the_change():
    s = ai_suggest(photo(), fake_ai(exposure=0.5))
    reason = s.reasons["exposure"]
    assert reason.startswith("Brightened")
    before, after = map(int, re.search(r"(\d+)% → (\d+)%", reason).groups())
    assert after > before


def test_load_trained_round_trip(tmp_path):
    model = SliderNet(pretrained=False)
    torch.nn.init.normal_(model.head[-1].weight)  # so it isn't "do nothing"
    torch.save(model.state_dict(), tmp_path / "net.pt")
    loaded = load_trained(tmp_path / "net.pt")
    assert not loaded.training
    model.eval()
    photos = torch.rand(2, 3, 64, 64)
    assert torch.equal(loaded(photos), model(photos))
