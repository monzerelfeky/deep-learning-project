import json

import pytest
from sklearn.model_selection import train_test_split

from src.data.dataset import load_cub_dataset
from src.data.splitting import SPLIT_PATH

pytestmark = pytest.mark.integration


def test_split_reproducibility():

    dataset = load_cub_dataset("train")

    indices = list(range(len(dataset)))
    labels = list(dataset["label"])

    # Recreate the original split
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.10,
        random_state=42,
        stratify=labels
    )

    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)

    # Load saved indices
    with open(
        SPLIT_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        saved_split = json.load(file)

    # Verify exact reproducibility
    assert train_indices == saved_split["train_indices"]
    assert val_indices == saved_split["val_indices"]
