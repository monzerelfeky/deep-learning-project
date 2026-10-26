from datasets import load_dataset
from torch.utils.data import Dataset
from src.data.captions import preprocess_captions

DATASET_NAMES = {
    "train": "Multimodal-Fatima/CUB_train",
    "test": "Multimodal-Fatima/CUB_test",
}


def load_cub_dataset(split="train"):
    """
    Load the official CUB-200-2011 dataset.

    Parameters
    ----------
    split : str
        Either 'train' or 'test'.

    Returns
    -------
    datasets.Dataset
        Hugging Face dataset containing images,
        labels, captions, and metadata.
    """

    if split not in DATASET_NAMES:
        raise ValueError(
            f"Invalid split: {split}. "
            "Choose 'train' or 'test'."
        )

    print(f"Loading CUB dataset: {split}")

    dataset = load_dataset(
        DATASET_NAMES[split],
        split=split
    )

    print("Dataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")

    return dataset


class CUBDataset(Dataset):
    """
    PyTorch Dataset wrapper for CUB-200-2011.

    Returns transformed images and integer class labels.
    """

    def __init__(
        self,
        hf_dataset,
        transform=None,
        return_captions=False
    ):
        self.dataset = hf_dataset
        self.transform = transform
        self.return_captions = return_captions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]

        image = sample["image"].convert("RGB")

        label = int(sample["label"])

        if self.transform is not None:
            image = self.transform(image)

        if self.return_captions:

            captions = preprocess_captions(
                sample["description"]
            )

            return image, label, captions

        return image, label