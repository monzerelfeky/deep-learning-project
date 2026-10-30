import pytest
from datasets import ClassLabel, Dataset, Features, Image as HFImage, Value
from PIL import Image

from src.data.splitting import (
    apply_exclusions,
    create_stratified_split,
    load_split_indices
)

N_CLASSES = 4
PER_CLASS = 10

DESCRIPTION = (
    "This bird has a RED crown, and a white belly.\n"
    "this bird has a red crown and a white belly\n"
    "this\n"
    "a��small��bird��with��brown��wings\n"
    "the bird's wing-tips are grey.\n"
)


def make_fake_cub(prefix, n_classes=N_CLASSES, per_class=PER_CLASS):
    """Small dataset with the CUB schema: image, description, label, file_name, id."""

    rows = {"image": [], "description": [], "label": [], "file_name": [], "id": []}

    for label in range(n_classes):
        for k in range(per_class):
            i = len(rows["id"])
            size = (64, 48) if i % 2 == 0 else (40, 60)  # landscape and portrait
            if i == 0:
                image = Image.new("L", size, color=120)  # CUB has single-channel images
            else:
                image = Image.new("RGB", size, color=(60 * label % 256, 20 * k % 256, 90))
            rows["image"].append(image)
            rows["description"].append(DESCRIPTION)
            rows["label"].append(label)
            rows["file_name"].append(f"{prefix}_Species_{label}_{k:04d}.jpg")
            rows["id"].append(i)

    features = Features({
        "image": HFImage(),
        "description": Value("string"),
        "label": ClassLabel(names=[f"Species {c}" for c in range(n_classes)]),
        "file_name": Value("string"),
        "id": Value("int64"),
    })

    return Dataset.from_dict(rows, features=features)


@pytest.fixture(scope="session")
def fake_train():
    return make_fake_cub("Train")


@pytest.fixture(scope="session")
def fake_test():
    return make_fake_cub("Test", per_class=5)


@pytest.fixture(scope="session")
def fake_partitions(fake_train, fake_test, tmp_path_factory):
    """Canonical partitions built the same way as load_partitions(), on fake data."""

    split_path = tmp_path_factory.mktemp("split") / "split_indices.json"
    create_stratified_split(fake_train, save_path=split_path)

    train_indices, val_indices = load_split_indices(fake_train, split_path)

    # Exclude one image from each partition, like the leakage list does on CUB
    excluded = [
        fake_train[train_indices[0]]["file_name"],
        fake_train[val_indices[0]]["file_name"],
    ]

    train_indices, val_indices, _ = apply_exclusions(
        fake_train, train_indices, val_indices, excluded
    )

    return {
        "train": fake_train.select(train_indices),
        "val": fake_train.select(val_indices),
        "test": fake_test,
    }
