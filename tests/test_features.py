"""Unit tests for vials/features.py."""

from PIL import Image
import pytest
import torch

from vials.features import (
    calculate_image_statistics,
    define_transforms,
    get_transforms,
    load_image_specs,
)

IMAGE_SIZE = (8, 8)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# define_transforms


def test_define_transforms_resizes_to_image_size():
    image = Image.new("RGB", (40, 25), (10, 20, 30))
    train_t, val_t = define_transforms(IMAGE_SIZE, MEAN, STD)

    assert train_t(image).shape == (3, *IMAGE_SIZE)
    assert val_t(image).shape == (3, *IMAGE_SIZE)


# get_transforms / config


def test_get_transforms_uses_image_size_from_config():
    image_size = tuple(load_image_specs()["image_size"])
    _, val_t = get_transforms()

    out = val_t(Image.new("RGB", (100, 50)))

    assert out.shape == (3, *image_size)


# calculate_image_statistics


def test_calculate_image_statistics_on_constant_images():
    # 2 batches of 3 images where every pixel is (0.2, 0.4, 0.6)
    batch = torch.ones(3, 3, 4, 4) * torch.tensor([0.2, 0.4, 0.6]).view(1, 3, 1, 1)
    labels = torch.zeros(3)
    fake_loader = [(batch, labels), (batch, labels)]

    stats = calculate_image_statistics(fake_loader)
    # if every pixel in every image is exactly the same colour, the average must be that colour, and the spread must be 0
    assert stats["mean"] == pytest.approx([0.2, 0.4, 0.6], abs=1e-6)
    assert stats["std"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert stats["total_samples"] == 6