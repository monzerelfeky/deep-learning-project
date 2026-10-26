import hashlib
import re
from pathlib import Path

import yaml
from datasets import load_dataset
from torch.utils.data import Dataset

from src.data.captions import clean_captions

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"


def load_data_config(path=None):
    """
    Read configs/data.yaml: dataset revisions and record lists.
    """

    path = Path(path) if path is not None else DATA_CONFIG_PATH

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_cub_dataset(
    split="train",
    revision=None,
    cache_dir=None,
    config_path=None
):
    """
    Load the official CUB-200-2011 dataset.

    Parameters
    ----------
    split : str
        Either 'train' or 'test'.

    revision : str
        Hugging Face commit hash. Defaults to the
        revision pinned in configs/data.yaml.

    cache_dir : str
        Hugging Face cache folder (optional).

    config_path : str
        Data config to read instead of configs/data.yaml.

    Returns
    -------
    datasets.Dataset
        Hugging Face dataset containing images,
        labels, captions, and metadata.
    """

    if split not in ("train", "test"):
        raise ValueError(
            f"Invalid split: {split}. "
            "Choose 'train' or 'test'."
        )

    config = load_data_config(config_path)["dataset"]
    repo = config[f"{split}_repo"]

    if revision is None:
        revision = config[f"{split}_revision"]

    print(f"Loading CUB dataset: {split} ({repo} @ {revision[:7]})")

    dataset = load_dataset(
        repo,
        split=split,
        revision=revision,
        cache_dir=cache_dir
    )

    # Offline (HF_HUB_OFFLINE=1), datasets ignores `revision` and loads
    # the most recent cached copy, so check which copy was actually loaded
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        cache_files = [file["filename"] for file in dataset.cache_files]

        if cache_files and not any(revision in name for name in cache_files):
            raise RuntimeError(
                f"Loaded {repo} from {cache_files[0]}, "
                f"which is not the pinned revision {revision}."
            )

    print("Dataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")

    return dataset


def dataset_fingerprint(hf_dataset):
    """
    Identify a dataset by its size and ordered file names.

    Saved split indices are only valid for a dataset with
    the same fingerprint. Reads one text column; no image
    is decoded.
    """

    file_names = list(hf_dataset["file_name"])

    digest = hashlib.sha256(
        "\n".join(file_names).encode("utf-8")
    ).hexdigest()

    return {
        "num_rows": len(file_names),
        "file_names_sha256": digest
    }


class CUBDataset(Dataset):
    """
    PyTorch Dataset wrapper for CUB-200-2011.

    Returns transformed images and integer class labels:
    (image, label), followed by the cleaned captions if
    return_captions=True and the file name if
    return_file_name=True.
    """

    def __init__(
        self,
        hf_dataset,
        transform=None,
        return_captions=False,
        return_file_name=False
    ):
        self.dataset = hf_dataset
        self.transform = transform
        self.return_captions = return_captions
        self.return_file_name = return_file_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]

        image = sample["image"].convert("RGB")

        label = int(sample["label"])

        if self.transform is not None:
            image = self.transform(image)

        output = [image, label]

        if self.return_captions:

            # Derived field; sample["description"] stays unchanged
            output.append(
                clean_captions(sample["description"])
            )

        if self.return_file_name:
            output.append(sample["file_name"])

        return tuple(output)
