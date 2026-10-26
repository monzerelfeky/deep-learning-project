import json
from collections import Counter
from pathlib import Path

import pytest

from src.data.dataset import load_cub_dataset


SPLIT_PATH = Path("configs/split_indices.json")


@pytest.fixture(scope="module")
def dataset():
    """Load the original training dataset once."""
    return load_cub_dataset("train")


@pytest.fixture(scope="module")
def split_data():
    """Read the saved train/validation indices."""
    assert SPLIT_PATH.exists(), (
        f"Missing split file: {SPLIT_PATH}"
    )

    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def test_expected_dataset_size(dataset):
    assert len(dataset) == 5994


def test_expected_split_sizes(split_data):
    assert len(split_data["train_indices"]) == 5394
    assert len(split_data["val_indices"]) == 600


def test_no_overlap(split_data):
    train_indices = set(split_data["train_indices"])
    val_indices = set(split_data["val_indices"])

    assert train_indices.isdisjoint(val_indices)


def test_all_samples_accounted_for(dataset, split_data):
    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]

    combined = train_indices + val_indices

    assert len(combined) == len(dataset)
    assert set(combined) == set(range(len(dataset)))


def test_valid_labels(dataset):
    labels = dataset["label"]

    assert len(set(labels)) == 200
    assert all(0 <= label < 200 for label in labels)


def test_stratification(dataset, split_data):
    labels = dataset["label"]

    train_labels = [
        labels[i] for i in split_data["train_indices"]
    ]

    val_labels = [
        labels[i] for i in split_data["val_indices"]
    ]

    original_counts = Counter(labels)
    val_counts = Counter(val_labels)

    for class_id, original_count in original_counts.items():
        expected_val_count = original_count * 0.10

        assert abs(
            val_counts[class_id] - expected_val_count
        ) <= 1


def test_saved_seed(split_data):
    assert split_data["random_seed"] == 42
    assert split_data["validation_size"] == 0.10