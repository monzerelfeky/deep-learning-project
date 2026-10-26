import json
from pathlib import Path

from sklearn.model_selection import train_test_split


def create_stratified_split(
    dataset,
    validation_size=0.10,
    random_seed=42,
    save_path="configs/split_indices.json"
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
        Location for saving split indices.

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
    labels = dataset["label"]

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

    # Save indices for other team members
    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    split_data = {
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