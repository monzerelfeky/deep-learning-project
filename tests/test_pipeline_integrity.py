import json
from collections import Counter

import pytest
import torch

from src.data.captions import clean_captions, preprocess_captions
from src.data.dataloaders import create_dataloaders, create_datasets
from src.data.dataset import (
    dataset_fingerprint,
    load_cub_dataset,
    load_data_config
)
from src.data.splitting import SPLIT_PATH, load_partitions

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dataset():
    """Load the original training dataset once."""
    return load_cub_dataset("train")


@pytest.fixture(scope="module")
def test_dataset():
    """Load the official test dataset once."""
    return load_cub_dataset("test")


@pytest.fixture(scope="module")
def partitions():
    """Canonical train / val / test partitions."""
    return load_partitions()


@pytest.fixture(scope="module")
def split_data():
    """Read the saved train/validation indices."""
    assert SPLIT_PATH.exists(), (
        f"Missing split file: {SPLIT_PATH}"
    )

    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def test_expected_dataset_size(dataset, test_dataset):
    assert len(dataset) == 5994
    assert len(test_dataset) == 5794


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
    labels = list(dataset["label"])

    assert len(set(labels)) == 200
    assert all(0 <= label < 200 for label in labels)


def test_stratification(dataset, split_data):
    labels = list(dataset["label"])

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


def test_manifest_matches_dataset(dataset, split_data):
    config = load_data_config()["dataset"]

    assert split_data["dataset_revision"] == config["train_revision"]

    for key, value in dataset_fingerprint(dataset).items():
        assert split_data[key] == value


def test_canonical_partition_sizes(partitions):
    assert len(partitions["train"]) == 5393
    assert len(partitions["val"]) == 598
    assert len(partitions["test"]) == 5794


def test_leakage_exclusions(dataset, split_data, partitions):
    names = list(dataset["file_name"])
    leakage = [
        entry["file_name"]
        for entry in load_data_config()["exclusions"]["train_leakage"]
    ]

    # Where the saved split had put them: #774 and #5277 in val, #5903 in train
    positions = {name: names.index(name) for name in leakage}
    assert sorted(positions.values()) == [774, 5277, 5903]
    in_val = {name for name, i in positions.items() if i in set(split_data["val_indices"])}
    assert len(in_val) == 2

    used = set(partitions["train"]["file_name"]) | set(partitions["val"]["file_name"])
    assert used.isdisjoint(leakage)
    assert len(used) + len(leakage) == len(dataset)


def test_every_class_in_train_and_validation(partitions):
    train_counts = Counter(partitions["train"]["label"])
    val_counts = Counter(partitions["val"]["label"])

    assert len(train_counts) == 200
    assert len(val_counts) == 200
    assert min(val_counts.values()) >= 2


def test_test_set_untouched(test_dataset, partitions):
    assert dataset_fingerprint(partitions["test"]) == dataset_fingerprint(test_dataset)


def test_label_mapping_identical(dataset, test_dataset):
    assert dataset.features["label"].names == test_dataset.features["label"].names
    assert len(test_dataset.features["label"].names) == 200


def test_file_names_unique_and_disjoint(dataset, test_dataset):
    train_names = list(dataset["file_name"])
    test_names = list(test_dataset["file_name"])

    assert len(set(train_names)) == len(train_names)
    assert len(set(test_names)) == len(test_names)
    assert set(train_names).isdisjoint(test_names)


def test_captions(dataset, test_dataset):
    for split in (dataset, test_dataset):
        descriptions = list(split["description"])

        assert all(len(preprocess_captions(d)) == 10 for d in descriptions)

        kept = [len(clean_captions(d)) for d in descriptions]
        assert min(kept) >= 7 and max(kept) == 10


def test_colorization_list_resolves(dataset, test_dataset):
    names = {
        "train": set(dataset["file_name"]),
        "test": set(test_dataset["file_name"]),
    }

    for entry in load_data_config()["colorization_exclude"]:
        assert entry["file_name"] in names[entry["split"]]


def test_both_models_use_identical_partitions(partitions):
    scratch = create_datasets("scratch_cnn", partitions=partitions)
    resnet = create_datasets("resnet18", partitions=partitions)

    for a, b in zip(scratch, resnet):
        assert list(a.dataset["file_name"]) == list(b.dataset["file_name"])


@pytest.mark.parametrize("model_type", ["scratch_cnn", "resnet18"])
def test_real_batches(partitions, model_type):
    loaders = create_dataloaders(
        model_type=model_type,
        batch_size=8,
        partitions=partitions
    )

    for loader in loaders:
        images, labels = next(iter(loader))

        assert images.shape == (8, 3, 224, 224)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert 0 <= labels.min() and labels.max() < 200
