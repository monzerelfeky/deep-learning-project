
import torch

from src.data.dataloaders import create_dataloaders
from src.models.scratch_cnn import ScratchCNNEncoder
from src.models.classifier import BirdClassifier


def main():
    # Load the official project data splits and transforms.
    train_loader, val_loader, test_loader = create_dataloaders(
        model_type="scratch_cnn",
        batch_size=4,
        image_size=224,
        num_workers=0,
        seed=42,
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    # Retrieve one batch to verify the dataset pipeline.
    images, labels = next(iter(train_loader))

    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label range:", labels.min().item(), labels.max().item())

    # Test the CNN using a real batch of bird images.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BirdClassifier(
        encoder=ScratchCNNEncoder(),
        num_classes=200,
    ).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(images.to(device))

    print("Device:", device)
    print("CNN output shape:", logits.shape)

    assert images.shape[1:] == (3, 224, 224)
    assert labels.min().item() >= 0
    assert labels.max().item() < 200
    assert logits.shape == (images.shape[0], 200)

    print("DataLoader and CNN integration successful!")


if __name__ == "__main__":
    main()
