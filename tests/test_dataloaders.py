import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.data.dataloaders import (
    collate_with_captions,
    create_dataloaders,
    create_datasets
)
from src.data.dataset import CUBDataset
from tests.conftest import N_CLASSES


@pytest.mark.parametrize("model_type", ["scratch_cnn", "resnet18"])
def test_batch_shapes_and_types(fake_partitions, model_type):

    loaders = create_dataloaders(
        model_type=model_type,
        batch_size=4,
        partitions=fake_partitions
    )

    for loader in loaders:

        images, labels = next(iter(loader))

        # The fake validation partition has only 3 images
        assert images.shape == (min(4, len(loader.dataset)), 3, 224, 224)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert torch.isfinite(images).all()
        assert 0 <= labels.min() and labels.max() < N_CLASSES


def test_partition_sizes(fake_partitions):

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=4,
        partitions=fake_partitions
    )

    # 36 train + 4 val, minus one exclusion from each
    assert len(train_loader.dataset) == 35
    assert len(val_loader.dataset) == 3
    assert len(test_loader.dataset) == len(fake_partitions["test"])


def test_only_training_is_shuffled_and_augmented(fake_partitions):

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=4,
        partitions=fake_partitions
    )

    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)

    def step_names(loader):
        return [type(step).__name__ for step in loader.dataset.transform.transforms]

    assert "RandomResizedCrop" in step_names(train_loader)
    assert "RandomResizedCrop" not in step_names(val_loader)
    assert "RandomResizedCrop" not in step_names(test_loader)


def test_evaluation_is_deterministic(fake_partitions):

    _, val_loader, test_loader = create_dataloaders(
        batch_size=4,
        partitions=fake_partitions
    )

    for loader in (val_loader, test_loader):
        first = torch.cat([images for images, _ in loader])
        second = torch.cat([images for images, _ in loader])
        assert torch.equal(first, second)


def test_shuffle_order_follows_seed(fake_partitions):

    def file_order(seed):
        train_loader, _, _ = create_dataloaders(
            batch_size=8,
            seed=seed,
            return_file_name=True,
            partitions=fake_partitions
        )
        return [name for _, _, names in train_loader for name in names]

    assert file_order(42) == file_order(42)
    assert sorted(file_order(42)) == sorted(fake_partitions["train"]["file_name"])


def test_return_file_name(fake_partitions):

    _, val_loader, _ = create_dataloaders(
        batch_size=2,
        return_file_name=True,
        partitions=fake_partitions
    )

    images, labels, names = next(iter(val_loader))

    assert images.shape == (2, 3, 224, 224)
    assert list(names) == list(fake_partitions["val"]["file_name"])[:2]


def test_both_models_use_the_same_partitions(fake_partitions):

    scratch = create_datasets("scratch_cnn", partitions=fake_partitions)
    resnet = create_datasets("resnet18", partitions=fake_partitions)

    for a, b in zip(scratch, resnet):
        assert list(a.dataset["file_name"]) == list(b.dataset["file_name"])


def test_workers_can_load_batches(fake_partitions):

    # Windows starts workers with spawn: datasets and transforms are pickled
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=4,
        num_workers=2,
        partitions=fake_partitions
    )

    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 224, 224)

    images, labels = next(iter(val_loader))
    assert images.shape[1:] == (3, 224, 224)


def test_captions_batch(fake_partitions):

    dataset = CUBDataset(
        fake_partitions["val"],
        transform=create_datasets(partitions=fake_partitions)[1].transform,
        return_captions=True,
        return_file_name=True
    )

    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_with_captions)
    images, labels, captions, names = next(iter(loader))

    assert images.shape == (3, 3, 224, 224)
    assert labels.dtype == torch.int64
    assert len(captions) == 3 and all(len(c) == 3 for c in captions)
    assert captions[0][0] == "this bird has a red crown and a white belly"
    assert len(names) == 3

    # The raw description is left unchanged
    assert fake_partitions["val"][0]["description"].startswith("This bird has a RED crown")


def test_invalid_arguments(fake_partitions):

    with pytest.raises(ValueError):
        create_dataloaders(model_type="vgg", partitions=fake_partitions)

    with pytest.raises(ValueError):
        create_dataloaders(batch_size=0, partitions=fake_partitions)

    with pytest.raises(ValueError):
        create_dataloaders(num_workers=-1, partitions=fake_partitions)
