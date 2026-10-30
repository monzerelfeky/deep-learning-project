import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CUBDataset
from src.data.splitting import load_partitions
from src.data.transforms import get_transforms


def seed_worker(worker_id):
    """
    Seed numpy and random in a DataLoader worker.

    torch is already seeded per worker from the loader's generator.
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_with_captions(batch):
    """
    Batch items of CUBDataset(return_captions=True).

    Images have 7 to 10 cleaned captions, so captions stay a
    list (one list per image) instead of being stacked.
    """

    columns = list(zip(*batch))

    images = torch.stack(columns[0])
    labels = torch.tensor(columns[1], dtype=torch.int64)

    return (images, labels, *[list(column) for column in columns[2:]])


def create_datasets(
    model_type="scratch_cnn",
    image_size=224,
    split_path=None,
    config_path=None,
    partitions=None,
    return_file_name=False,
    train_augmentation=True
):
    """
    Create the train, validation and test CUBDatasets.

    partitions: optional {"train", "val", "test"} Hugging Face
    datasets to use instead of load_partitions().

    Returns
    -------
    train_dataset, val_dataset, test_dataset
    """

    if not isinstance(train_augmentation, bool):
        raise ValueError("train_augmentation must be a boolean.")

    # Training transforms
    train_transform = get_transforms(
        model_type=model_type,
        image_size=image_size,
        is_training=train_augmentation
    )

    # Validation and test transforms
    eval_transform = get_transforms(
        model_type=model_type,
        image_size=image_size,
        is_training=False
    )

    if partitions is None:
        partitions = load_partitions(split_path, config_path)

    # Wrap subsets as PyTorch datasets
    train_dataset = CUBDataset(
        partitions["train"],
        transform=train_transform,
        return_file_name=return_file_name
    )

    val_dataset = CUBDataset(
        partitions["val"],
        transform=eval_transform,
        return_file_name=return_file_name
    )

    test_dataset = CUBDataset(
        partitions["test"],
        transform=eval_transform,
        return_file_name=return_file_name
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    model_type="scratch_cnn",
    batch_size=32,
    image_size=224,
    num_workers=0,
    split_path=None,
    seed=42,
    return_file_name=False,
    config_path=None,
    partitions=None,
    train_augmentation=True
):
    """
    Create reproducible PyTorch DataLoaders.

    Batches are (images, labels), plus file names if
    return_file_name=True.

    With num_workers=0 the augmentation draws from the global
    torch RNG, so call torch.manual_seed(seed) before training.

    Returns
    -------
    train_loader, val_loader, test_loader
    """

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    train_dataset, val_dataset, test_dataset = create_datasets(
        model_type=model_type,
        image_size=image_size,
        split_path=split_path,
        config_path=config_path,
        partitions=partitions,
        return_file_name=return_file_name,
        train_augmentation=train_augmentation
    )

    def make_generator():
        # Every loader draws a seed from its generator on each
        # iter(); without one, evaluating would shift the global
        # RNG and change later training augmentation
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator()
    )

    return train_loader, val_loader, test_loader
