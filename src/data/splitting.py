import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data.dataset import (
    REPO_ROOT,
    dataset_fingerprint,
    load_cub_dataset,
    load_data_config
)

SPLIT_PATH = REPO_ROOT / "configs" / "split_indices.json"


def create_stratified_split(
    dataset,
    validation_size=0.10,
    random_seed=42,
    save_path=None,
    overwrite=False,
    config_path=None
):
    """
    Create reproducible stratified training and validation splits.

    Parameters
    ----------
    dataset:
        Original Hugging Face training dataset.

    validation_size:
        Fraction of images used for validation.

    random_seed:
        Ensures reproducibility.

    save_path:
        Location for saving split indices
        (default: configs/split_indices.json).

    overwrite:
        Replace a saved split with different indices.
        Without it, only an identical split is re-saved.

    config_path:
        Data config that names the dataset revision.

    Returns
    -------
    train_dataset, val_dataset
    """

    if not 0 < validation_size < 1:
        raise ValueError(
            "validation_size must be between 0 and 1."
        )

    # Get all image indices
    indices = list(range(len(dataset)))

    # Get the class labels
    labels = list(dataset["label"])

    # Create stratified split
    train_indices, val_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_seed,
        stratify=labels
    )

    # Sort indices for consistent ordering
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)

    save_path = Path(save_path) if save_path is not None else SPLIT_PATH

    if save_path.exists() and not overwrite:

        with open(save_path, "r", encoding="utf-8") as file:
            saved = json.load(file)

        if (saved.get("train_indices") != train_indices
                or saved.get("val_indices") != val_indices):
            raise ValueError(
                f"{save_path} holds a different split. "
                "Pass overwrite=True to replace it."
            )

    # Save indices for other team members
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    config = load_data_config(config_path)["dataset"]

    split_data = {
        "dataset_repo": config["train_repo"],
        "dataset_revision": config["train_revision"],
        **dataset_fingerprint(dataset),
        "random_seed": random_seed,
        "validation_size": validation_size,
        "train_indices": train_indices,
        "val_indices": val_indices
    }

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(split_data, file, indent=2)

    # Create the subsets
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    return train_dataset, val_dataset


def load_split_indices(dataset, split_path=None):
    """
    Read the saved split and check that it belongs to dataset.

    Returns
    -------
    train_indices, val_indices
    """

    split_file = Path(split_path) if split_path is not None else SPLIT_PATH

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. "
            "Run the dataset splitting step first."
        )

    with open(split_file, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    # Indices are row positions, so they only mean something
    # for a dataset with the same rows in the same order
    for key, value in dataset_fingerprint(dataset).items():

        if split_data.get(key) != value:
            raise ValueError(
                f"{split_file} does not match the loaded dataset "
                f"({key}: saved {split_data.get(key)}, loaded {value}). "
                "Check the dataset revision in configs/data.yaml."
            )

    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]

    # Validate indices
    all_indices = train_indices + val_indices

    if not all_indices:
        raise ValueError("Split indices cannot be empty.")

    if len(all_indices) != len(dataset):
        raise ValueError(
            "Split indices do not cover the original training dataset."
        )

    if len(set(all_indices)) != len(all_indices):
        raise ValueError(
            "Duplicate indices detected in train/validation splits."
        )

    if min(all_indices) < 0 or max(all_indices) >= len(dataset):
        raise ValueError("Split indices are out of range.")

    return train_indices, val_indices


def apply_exclusions(dataset, train_indices, val_indices, file_names):
    """
    Remove records by file name after the split is loaded.

    The saved indices stay valid; excluded records are dropped
    from whichever partition they landed in.

    Returns
    -------
    train_indices, val_indices, excluded_indices
    """

    all_names = list(dataset["file_name"])
    position = {name: i for i, name in enumerate(all_names)}

    if len(position) != len(all_names):
        raise ValueError("Duplicate file names in the dataset.")

    missing = [name for name in file_names if name not in position]

    if missing:
        raise ValueError(
            f"Excluded file names not found in the dataset: {missing}"
        )

    excluded = {position[name] for name in file_names}

    kept_train = [i for i in train_indices if i not in excluded]
    kept_val = [i for i in val_indices if i not in excluded]

    if len(kept_train) + len(kept_val) + len(excluded) != len(dataset):
        raise ValueError(
            "Train, validation and excluded records "
            "do not add up to the dataset."
        )

    print(
        f"Excluded {len(excluded)} records: "
        f"{len(train_indices) - len(kept_train)} from train, "
        f"{len(val_indices) - len(kept_val)} from validation"
    )

    return kept_train, kept_val, sorted(excluded)


def load_partitions(split_path=None, config_path=None, cache_dir=None):
    """
    Load the canonical train / validation / test partitions.

    Validation is the saved stratified 10% of the official
    training set; the train-to-test leakage images listed in
    configs/data.yaml are removed from train and validation.
    The official test set is returned unchanged.

    Returns
    -------
    dict
        {"train": ..., "val": ..., "test": ...} Hugging Face datasets.
    """

    config = load_data_config(config_path)

    original_train = load_cub_dataset(
        "train",
        cache_dir=cache_dir,
        config_path=config_path
    )

    original_test = load_cub_dataset(
        "test",
        cache_dir=cache_dir,
        config_path=config_path
    )

    train_indices, val_indices = load_split_indices(
        original_train,
        split_path
    )

    leakage = [
        entry["file_name"]
        for entry in config["exclusions"]["train_leakage"]
    ]

    train_indices, val_indices, _ = apply_exclusions(
        original_train,
        train_indices,
        val_indices,
        leakage
    )

    return {
        "train": original_train.select(train_indices),
        "val": original_train.select(val_indices),
        "test": original_test
    }
