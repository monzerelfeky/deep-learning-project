from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataloaders import create_dataloaders
from src.evaluation.run_evaluation import build_model_for_checkpoint
from src.visualization.gradcam import GradCAM, overlay_heatmap


CHECKPOINT = Path(
    "outputs/checkpoints/tuning/grid/"
    "trial_0.001_0.0001_False/best_model.pt"
)

OUTPUT_DIR = Path("outputs/evaluation/figures/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
NUM_CLASSES = 200
MAX_EXAMPLES_PER_TYPE = 2


def denormalize(image, model_type):
    """Convert normalized CHW tensor to displayable RGB."""
    image = image.detach().cpu().float().numpy()

    if model_type == "resnet18":
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
    else:
        # Scratch CNN transforms use their own configured normalization.
        # Retrieve the actual transform statistics below if necessary.
        mean = np.zeros((3, 1, 1), dtype=np.float32)
        std = np.ones((3, 1, 1), dtype=np.float32)

    image = image * std + mean
    image = np.clip(image, 0, 1)
    return np.moveaxis(image, 0, -1)


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    print("Loading checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model, model_type = build_model_for_checkpoint(
        "grid_search",
        checkpoint,
        CHECKPOINT,
    )
    model.to(DEVICE)
    model.eval()

    encoder = model.encoder

    if not hasattr(encoder, "gradcam_layer"):
        raise AttributeError(
            "The selected encoder does not expose gradcam_layer."
        )

    target_layer = encoder.gradcam_layer
    print(f"Architecture: {type(encoder).__name__}")
    print(f"Target layer: {target_layer}")

    print("Loading CUB test dataset...")
    _, _, test_loader = create_dataloaders(
        model_type=model_type,
        batch_size=8,
        image_size=224,
        num_workers=0,
        split_path="configs/split_indices.json",
        config_path="configs/data.yaml",
        seed=42,
        return_file_name=True,
        train_augmentation=False,
    )

    correct = []
    incorrect = []

    print("Finding correctly and incorrectly classified examples...")

    with torch.no_grad():
        for batch in test_loader:
            images, labels, filenames = batch
            logits = model(images.to(DEVICE))
            predictions = logits.argmax(dim=1).cpu()

            for i in range(len(labels)):
                item = {
                    "image": images[i].clone(),
                    "true": int(labels[i]),
                    "pred": int(predictions[i]),
                    "filename": filenames[i],
                }

                if item["true"] == item["pred"]:
                    if len(correct) < MAX_EXAMPLES_PER_TYPE:
                        correct.append(item)
                elif len(incorrect) < MAX_EXAMPLES_PER_TYPE:
                    incorrect.append(item)

                if (
                    len(correct) >= MAX_EXAMPLES_PER_TYPE
                    and len(incorrect) >= MAX_EXAMPLES_PER_TYPE
                ):
                    break

            if (
                len(correct) >= MAX_EXAMPLES_PER_TYPE
                and len(incorrect) >= MAX_EXAMPLES_PER_TYPE
            ):
                break

    selected = [
        (item, "Correct") for item in correct
    ] + [
        (item, "Incorrect") for item in incorrect
    ]

    if not selected:
        raise RuntimeError("No examples were found.")

    print(f"Selected {len(correct)} correct and {len(incorrect)} incorrect examples.")

    with GradCAM(model, target_layer) as gradcam:
        for index, (item, status) in enumerate(selected, start=1):
            image_tensor = item["image"].unsqueeze(0)

            result = gradcam.generate(image_tensor)

            heatmap = result["heatmaps"][0].numpy()
            original = denormalize(item["image"], model_type)
            overlay = overlay_heatmap(original, heatmap, alpha=0.4)

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))

            axes[0].imshow(original)
            axes[0].set_title("Original image")

            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Grad-CAM heatmap")

            axes[2].imshow(overlay)
            axes[2].set_title("Grad-CAM overlay")

            for ax in axes:
                ax.axis("off")

            fig.suptitle(
                f"{status} | True class: {item['true']} | "
                f"Predicted class: {item['pred']}\n"
                f"File: {item['filename']}",
                fontsize=11,
            )
            fig.tight_layout()

            filename = f"gradcam_{index}_{status.lower()}.png"
            fig.savefig(
                OUTPUT_DIR / filename,
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

            print(f"Saved: {filename}")

    print(f"\nGrad-CAM images saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
