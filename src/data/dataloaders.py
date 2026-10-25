import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    CUBDataset,
    load_cub_dataset
)

from src.data.transforms import get_transforms


def create_dataloaders(
    model_type="scratch_cnn",
    batch_size=32,
    image_size=224,
    num_workers=0,
    split_path="configs/split_indices.json",
    seed=42
):
    """
    Create reproducible PyTorch DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    # Load official datasets
    original_train = load_cub_dataset("train")
    original_test = load_cub_dataset("test")

    # Load saved split indices
    split_file = Path(split_path)

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. "
            "Run the dataset splitting step first."
        )

    with open(split_file, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]

    # Validate indices
    all_indices = train_indices + val_indices

    if len(all_indices) != len(original_train):
        raise ValueError(
            "Split indices do not cover the original training dataset."
        )

    if len(set(all_indices)) != len(all_indices):
        raise ValueError(
            "Duplicate indices detected in train/validation splits."
        )

    if not all_indices:
        raise ValueError("Split indices cannot be empty.")

    if min(all_indices) < 0 or max(all_indices) >= len(original_train):
        raise ValueError("Split indices are out of range.")

    # Create Hugging Face subsets
    train_subset = original_train.select(train_indices)
    val_subset = original_train.select(val_indices)

    # Training transforms
    train_transform = get_transforms(
        model_type=model_type,
        image_size=image_size,
        is_training=True
    )

    # Validation and test transforms
    eval_transform = get_transforms(
        model_type=model_type,
        image_size=image_size,
        is_training=False
    )

    # Wrap subsets as PyTorch datasets
    train_dataset = CUBDataset(
        train_subset,
        transform=train_transform
    )

    val_dataset = CUBDataset(
        val_subset,
        transform=eval_transform
    )

    test_dataset = CUBDataset(
        original_test,
        transform=eval_transform
    )

    # Generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader