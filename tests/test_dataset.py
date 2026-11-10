"""Unit tests for vials/dataset.py.

Every test builds a few tiny images in pytest
tmp_path folder, so the tests run in a second and work in CI.
"""

from pathlib import Path

from PIL import Image
import pytest
import torch

from vials import dataset # import test subject
from vials.dataset import (
    GlassVialDataset,
    count_classes,
    create_dataloaders,
    load_image_dataset,
    load_split,
    save_split,
    split_dataset,
)
from vials.features import define_transforms # some tests need transforms and normalization to make a dataset

IMAGE_SIZE = (8, 8) # keeping small to test fast, this is the size the image should become after the transforms resize it, the real setting is 64×64
MEAN = [0.485, 0.456, 0.406] # same as training_config.yaml
STD = [0.229, 0.224, 0.225] # same as training_config.yaml

# HELPER FUNCTIONS
# _ at the start of the name tells pytest it Is not a test, Pytest only runs functions that start with test_
def _make_image(path, mode="RGB", size=(20, 30)): # size=(20, 30) is the size of the image before processing, playing the part of the raw photo on disk, real vial photos are 1600×768, keeping small for faster tests
    """Writing a small solid colour image to disk and return its path."""
    colour = (120, 60, 30) if mode == "RGB" else 128 # mode="L" makes a grayscale image (one channel, so the colour is one number, 128). (120, 60, 30) means red = 120, green = 60, blue = 30, which is a brownish colour
    Image.new(mode, size, colour).save(path)
    return path

# fake list of labelled images
def _fake_dataset(n_good=80, n_defective=20):
    """(path, label) pairs like load_image_dataset returns. """
    good = [(Path(f"good_{i}.png"), 0) for i in range(n_good)]
    defective = [(Path(f"defective_{i}.png"), 1) for i in range(n_defective)]
    return good + defective

# -----------------------------------------------------------------------------
# TEST FUNCTIONS

# count classes check

def test_count_classes_counts_each_label():
    assert count_classes([0, 1, 1, 0, 1]) == {0: 2, 1: 3}


# split dataset


def test_split_dataset_uses_70_15_15_ratio():
    train, val, test = split_dataset(_fake_dataset(80, 20))

    assert (len(train), len(val), len(test)) == (70, 15, 15)


def test_split_dataset_has_no_overlap_and_loses_nothing():
    """Checks: no image is in two splits, and no image disappears"""
    data = _fake_dataset()
    train, val, test = split_dataset(data)

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train) | set(val) | set(test) == set(data)


def test_split_dataset_keeps_class_balance_in_every_split():
    """Checks: every part has the same share of defective images as the full data"""
    train, val, test = split_dataset(_fake_dataset(80, 20))

    for split in (train, val, test):
        defective_share = sum(label for _, label in split) / len(split)
        assert defective_share == pytest.approx(0.2)


def test_split_dataset_is_reproducible_with_same_seed():
    """Checks: if you call split_dataset twice with the same random_seed, you get the same result"""
    data = _fake_dataset()

    first = split_dataset(data, random_seed=42)
    second = split_dataset(data, random_seed=42)

    assert first == second


# load image dataset
   

def test_load_image_dataset_skips_missing_files(tmp_path, monkeypatch):
    """Checks: loading an image dataset skips files that are missing from the image directory."""
    _make_image(tmp_path / "a.png")
    _make_image(tmp_path / "b.png")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("filename,class\na.png,0\nb.png,1\nmissing.png,1\n") # creating csv
    monkeypatch.setattr(dataset, "IMAGE_DATA_DIR", tmp_path) # temporary changing folder in dataset.py to the pytest tmp_path folder, so the test doesn't need to use the real data folder
    monkeypatch.setattr(dataset, "METADATA_PATH", metadata)

    result = load_image_dataset()

    assert result == [(tmp_path / "a.png", 0), (tmp_path / "b.png", 1)]


def test_load_image_dataset_raises_when_metadata_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "METADATA_PATH", tmp_path / "does_not_exist.csv")

    with pytest.raises(FileNotFoundError):
        load_image_dataset()


# save_split / load_split
# if saving or loading changed anything, training would use the wrong images or labels

def test_save_split_and_load_split_round_trip(tmp_path, monkeypatch):

    monkeypatch.setattr(dataset, "IMAGE_DATA_DIR", tmp_path)
    split = [(tmp_path / "a.png", 0), (tmp_path / "b.png", 1)]
    csv_path = tmp_path / "train.csv"

    save_split(split, csv_path)

    assert load_split(csv_path) == split


# GlassVialDataset


def test_glass_vial_dataset_returns_tensor_and_label(tmp_path):
    image_path = _make_image(tmp_path / "vial.png")
    _, val_transforms = define_transforms(IMAGE_SIZE, MEAN, STD)
    ds = GlassVialDataset([image_path], [1], val_transforms)

    image, label = ds[0]

    assert len(ds) == 1
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, *IMAGE_SIZE)
    assert label == 1


def test_glass_vial_dataset_converts_grayscale_to_rgb(tmp_path):
    image_path = _make_image(tmp_path / "gray.png", mode="L")
    _, val_transforms = define_transforms(IMAGE_SIZE, MEAN, STD)
    ds = GlassVialDataset([image_path], [0], val_transforms)

    image, _ = ds[0]

    assert image.shape[0] == 3


# create_dataloaders


def test_create_dataloaders_gives_batches_of_the_right_shape(tmp_path):
    paths = [_make_image(tmp_path / f"img_{i}.png") for i in range(5)]
    data = [(p, i % 2) for i, p in enumerate(paths)]
    train_t, val_t = define_transforms(IMAGE_SIZE, MEAN, STD)

    train_loader, val_loader, test_loader = create_dataloaders(
        data, data[:2], data[:1], train_t, val_t, batch_size=2
    )
    images, labels = next(iter(val_loader))

    assert (len(train_loader), len(val_loader), len(test_loader)) == (3, 1, 1)
    assert images.shape == (2, 3, *IMAGE_SIZE)
    assert labels.tolist() == [0, 1]