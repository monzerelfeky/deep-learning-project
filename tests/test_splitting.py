import json
import re
from collections import Counter

import pytest

from src.data.dataset import dataset_fingerprint, load_data_config
from src.data.splitting import (
    apply_exclusions,
    create_stratified_split,
    load_split_indices
)
from tests.conftest import N_CLASSES, PER_CLASS


def test_stratified_split_is_reproducible(fake_train, tmp_path):

    first, second = tmp_path / "a.json", tmp_path / "b.json"

    create_stratified_split(fake_train, save_path=first)
    create_stratified_split(fake_train, save_path=second)

    split_a = json.loads(first.read_text())
    split_b = json.loads(second.read_text())

    assert split_a == split_b
    assert split_a["random_seed"] == 42
    assert split_a["validation_size"] == 0.10


def test_stratified_split_integrity(fake_train, tmp_path):

    path = tmp_path / "split.json"
    train_subset, val_subset = create_stratified_split(fake_train, save_path=path)

    train_indices, val_indices = load_split_indices(fake_train, path)

    assert set(train_indices).isdisjoint(val_indices)
    assert sorted(train_indices + val_indices) == list(range(len(fake_train)))
    assert len(train_subset) + len(val_subset) == len(fake_train)

    # 10% of 10 images per class -> exactly one validation image per class
    assert Counter(val_subset["label"]) == {c: PER_CLASS // 10 for c in range(N_CLASSES)}


def test_manifest_records_dataset_fingerprint(fake_train, tmp_path):

    path = tmp_path / "split.json"
    create_stratified_split(fake_train, save_path=path)

    saved = json.loads(path.read_text())
    config = load_data_config()["dataset"]

    assert saved["num_rows"] == len(fake_train)
    assert saved["file_names_sha256"] == dataset_fingerprint(fake_train)["file_names_sha256"]
    assert saved["dataset_revision"] == config["train_revision"]


def test_reordered_dataset_is_rejected(fake_train, tmp_path):

    path = tmp_path / "split.json"
    create_stratified_split(fake_train, save_path=path)

    with pytest.raises(ValueError, match="does not match"):
        load_split_indices(fake_train.shuffle(seed=1), path)


def test_manifest_without_fingerprint_is_rejected(fake_train, tmp_path):

    path = tmp_path / "split.json"
    create_stratified_split(fake_train, save_path=path)

    saved = json.loads(path.read_text())
    del saved["file_names_sha256"]
    path.write_text(json.dumps(saved))

    with pytest.raises(ValueError, match="file_names_sha256"):
        load_split_indices(fake_train, path)


def test_different_split_is_not_overwritten(fake_train, tmp_path):

    path = tmp_path / "split.json"
    create_stratified_split(fake_train, random_seed=1, save_path=path)
    saved = path.read_text()

    with pytest.raises(ValueError, match="different split"):
        create_stratified_split(fake_train, random_seed=42, save_path=path)

    assert path.read_text() == saved

    create_stratified_split(fake_train, random_seed=42, save_path=path, overwrite=True)
    assert json.loads(path.read_text())["random_seed"] == 42


def test_exclusions_apply_to_train_and_validation(fake_train, tmp_path):

    path = tmp_path / "split.json"
    create_stratified_split(fake_train, save_path=path)
    train_indices, val_indices = load_split_indices(fake_train, path)

    names = list(fake_train["file_name"])
    excluded_names = [names[train_indices[3]], names[val_indices[1]]]

    kept_train, kept_val, excluded = apply_exclusions(
        fake_train, train_indices, val_indices, excluded_names
    )

    assert excluded == sorted([train_indices[3], val_indices[1]])
    assert len(kept_train) == len(train_indices) - 1
    assert len(kept_val) == len(val_indices) - 1
    assert len(kept_train) + len(kept_val) + len(excluded) == len(fake_train)
    assert not set(excluded) & (set(kept_train) | set(kept_val))


def test_unknown_exclusion_is_rejected(fake_train, tmp_path):

    path = tmp_path / "split.json"
    create_stratified_split(fake_train, save_path=path)
    train_indices, val_indices = load_split_indices(fake_train, path)

    with pytest.raises(ValueError, match="not found"):
        apply_exclusions(fake_train, train_indices, val_indices, ["missing.jpg"])


def test_data_config():

    config = load_data_config()

    for split in ("train", "test"):
        assert re.fullmatch(r"[0-9a-f]{40}", config["dataset"][f"{split}_revision"])

    assert len(config["exclusions"]["train_leakage"]) == 3

    splits = Counter(entry["split"] for entry in config["colorization_exclude"])
    assert splits == {"train": 5, "test": 4}
