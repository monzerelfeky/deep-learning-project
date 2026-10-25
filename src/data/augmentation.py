import matplotlib.pyplot as plt
import torch

from src.data.transforms import get_transforms


def denormalize_image(tensor, model_type="scratch_cnn"):
    """
    Reverse image normalization for visualization.
    """

    if model_type == "scratch_cnn":
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])

    elif model_type == "resnet18":
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

    else:
        raise ValueError("Unsupported model type.")

    mean = mean[:, None, None]
    std = std[:, None, None]

    image = tensor.detach().cpu() * std + mean

    return image.clamp(0, 1)


def visualize_augmentations(
    image,
    model_type="scratch_cnn",
    num_samples=5,
    save_path=None
):
    """
    Display original image and augmented versions.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    transform = get_transforms(
        model_type=model_type,
        is_training=True
    )

    fig, axes = plt.subplots(
        1,
        num_samples + 1,
        figsize=(3 * (num_samples + 1), 4)
    )

    axes[0].imshow(image.convert("RGB"))
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i in range(num_samples):

        augmented_tensor = transform(image)

        augmented_image = denormalize_image(
            augmented_tensor,
            model_type=model_type
        )

        augmented_image = augmented_image.permute(
            1, 2, 0
        ).numpy()

        axes[i + 1].imshow(augmented_image)
        axes[i + 1].set_title(f"Augmented {i + 1}")
        axes[i + 1].axis("off")

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path

        output_path = Path(save_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

    plt.show()
    plt.close(fig)