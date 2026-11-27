from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pocketform_distill.preprocess import FACTOR, MIN_PIXELS, image_sha256, png_bytes, prepare_image, target_size


@pytest.mark.parametrize("size", [(2446, 1228), (1234, 516), (4032, 3024), (1234, 291), (800, 600)])
def test_size_fits_budget_in_blocks_of_32(size):
    (w, h), (pw, ph) = target_size(*size, max_pixels=819200)
    assert pw % FACTOR == 0 and ph % FACTOR == 0
    assert pw * ph <= 819200
    assert w <= size[0] and h <= size[1]  # never upscales a normal image
    assert w <= pw < w + FACTOR and h <= ph < h + FACTOR  # padding is less than one block
    assert abs(w / h - size[0] / size[1]) < 0.01  # shape kept


def test_an_image_within_budget_keeps_its_pixels():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (516, 1234, 3), dtype=np.uint8))
    out = prepare_image(img, 819200)
    assert out.size == (1248, 544)
    assert np.array_equal(np.asarray(out)[:516, :1234], np.asarray(img))  # untouched, only padded


def test_big_photos_are_shrunk():
    out = prepare_image(Image.new("RGB", (4032, 3024), "white"), 819200)
    assert out.width * out.height <= 819200 and out.width % FACTOR == 0 and out.height % FACTOR == 0


def test_tiny_images_are_raised_to_the_student_minimum():
    (_, _), (pw, ph) = target_size(100, 60, max_pixels=819200)
    assert pw * ph >= MIN_PIXELS


def test_prepare_is_deterministic_and_uprights_exif(tmp_path):
    img = Image.new("RGB", (1300, 600), (200, 180, 150))
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways: show rotated 90 degrees
    path = tmp_path / "photo.jpg"
    img.save(path, exif=exif)
    a, b = prepare_image(path, 400000), prepare_image(Path(path), 400000)
    assert a.size[0] < a.size[1]  # turned upright
    assert png_bytes(a) == png_bytes(b) and image_sha256(a) == image_sha256(b)
