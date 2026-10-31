"""Checks for the FiveK loader. They use a few tiny fake photos, so they
run without downloading anything."""

import io

import numpy as np
import pytest
import torch
from PIL import Image, ImageCms

from kelvin.data import FiveK, split_names
from kelvin.data.prepare_fivek import centre_crop, match_score, save_pair, to_srgb, upright


@pytest.fixture
def fake_fivek(tmp_path):
    """Five fake photo pairs: three training, one validation and one test photo."""
    for folder in ("input", "expertC"):
        (tmp_path / folder).mkdir()
    for name in ("a0001-x", "a0002-y", "a4000-z", "a4500-v", "a4501-w"):
        Image.new("RGB", (60, 40), (200, 100, 50)).save(tmp_path / "input" / f"{name}.jpg")
        Image.new("RGB", (60, 40), (220, 120, 60)).save(tmp_path / "expertC" / f"{name}.jpg")
    return tmp_path


def test_split_is_by_photo_number(fake_fivek):
    assert split_names("train", fake_fivek) == ["a0001-x", "a0002-y", "a4000-z"]
    assert split_names("val", fake_fivek) == ["a4500-v"]
    assert split_names("test", fake_fivek) == ["a4501-w"]


def test_splits_never_share_a_photo(fake_fivek):
    train, val, test = (set(split_names(s, fake_fivek)) for s in ("train", "val", "test"))
    assert not (train & val or train & test or val & test)


def test_badly_matched_pairs_are_left_out(fake_fivek):
    (fake_fivek / "matches.csv").write_text("a0001-x,0.95\na0002-y,0.10\na4000-z,0.70\n")
    assert split_names("train", fake_fivek) == ["a0001-x", "a4000-z"]


def test_unknown_split_is_refused(fake_fivek):
    with pytest.raises(ValueError):
        split_names("validation", fake_fivek)


def test_missing_data_says_how_to_get_it(tmp_path):
    (tmp_path / "expertC").mkdir()
    with pytest.raises(FileNotFoundError, match="prepare_fivek"):
        split_names("train", tmp_path)


def test_examples_are_square_tensors_from_0_to_1(fake_fivek):
    data = FiveK("train", size=32, data_dir=fake_fivek)
    assert len(data) == 3
    before, after = data[0]
    for img in (before, after):
        assert img.shape == (3, 32, 32)
        assert img.dtype == torch.float32
        assert 0 <= img.min() and img.max() <= 1
    # the expert's fake photo is the brighter one
    assert after.mean() > before.mean()


def test_only_training_photos_are_flipped(fake_fivek):
    assert FiveK("train", data_dir=fake_fivek).flip
    assert not FiveK("val", data_dir=fake_fivek).flip
    assert not FiveK("test", data_dir=fake_fivek).flip


def test_photo_without_a_profile_is_left_alone():
    img = Image.new("RGB", (4, 4), (10, 200, 30))
    assert to_srgb(img).getpixel((0, 0)) == (10, 200, 30)


def test_photo_with_a_profile_goes_through_colour_conversion():
    """Pillow can't make a ProPhoto profile to test with, so this embeds an
    sRGB one: converting sRGB to sRGB must leave the colour (nearly) as it was.
    (The real ProPhoto files were checked by eye on downloaded photos.)"""
    img = Image.new("RGB", (4, 4), (10, 200, 30))
    img.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    assert to_srgb(img).getpixel((0, 0)) == pytest.approx((10, 200, 30), abs=2)


def test_centre_crop_trims_evenly():
    img = Image.new("RGB", (110, 60))
    cropped = centre_crop(img, (100, 50))
    assert cropped.size == (100, 50)


def _pattern(size=(300, 200), seed=0, block=20) -> Image.Image:
    """Random blocks: a picture with plenty of edges and no symmetry."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 255, (size[1] // block, size[0] // block, 3), dtype=np.uint8)
    return Image.fromarray(blocks).resize(size, Image.Resampling.NEAREST)


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_upright_undoes_any_rotation(turns):
    edited = _pattern()
    sideways = edited.rotate(90 * turns, expand=True)
    fixed, score = upright(sideways, edited)
    assert fixed.size == edited.size
    assert score > 0.9
    assert np.array_equal(np.asarray(fixed), np.asarray(edited))


def test_match_score_is_low_for_different_pictures():
    other = _pattern(seed=1, block=17).resize((300, 200))
    assert match_score(_pattern(), other) < 0.3


def test_save_pair_lines_up_sizes(tmp_path):
    for folder in ("input", "expertC"):
        (tmp_path / folder).mkdir()

    def as_bytes(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    save_pair(as_bytes(Image.new("RGB", (1210, 806))), as_bytes(Image.new("RGB", (1200, 800))), "a0001-x", tmp_path)
    before = Image.open(tmp_path / "input" / "a0001-x.jpg")
    after = Image.open(tmp_path / "expertC" / "a0001-x.jpg")
    assert before.size == after.size == (512, 341)
