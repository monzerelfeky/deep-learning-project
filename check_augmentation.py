from src.data.dataset import load_cub_dataset
from src.data.augmentation import visualize_augmentations


def main():

    print("Loading CUB dataset...")

    dataset = load_cub_dataset("train")

    # Select first bird image
    sample = dataset[0]

    image = sample["image"]

    print("Selected image:", sample["file_name"])
    print("Image size:", image.size)

    print("\nGenerating augmentation examples...")

    visualize_augmentations(
        image=image,
        model_type="scratch_cnn",
        num_samples=5,
        save_path="outputs/data/augmentation_examples.png"
    )

    print("\nAugmentation visualization completed!")


if __name__ == "__main__":
    main()