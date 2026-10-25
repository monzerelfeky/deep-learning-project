import torch

from src.data.dataloaders import create_dataloaders


def check_model_pipeline(model_type):

    print(f"\nTesting pipeline for: {model_type}")

    train_loader, val_loader, test_loader = create_dataloaders(
        model_type=model_type,
        batch_size=8,
        image_size=224,
        num_workers=0
    )

    print("\nDataset sizes:")
    print("Training:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Testing:", len(test_loader.dataset))

    # Get one training batch
    images, labels = next(iter(train_loader))

    print("\nTraining batch:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    assert images.shape == (8, 3, 224, 224)
    assert labels.shape == (8,)

    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

    assert torch.isfinite(images).all()

    assert labels.min() >= 0
    assert labels.max() < 200

    # Check validation batch
    val_images, val_labels = next(iter(val_loader))

    assert val_images.shape[1:] == (3, 224, 224)
    assert val_labels.dtype == torch.int64

    # Check test batch
    test_images, test_labels = next(iter(test_loader))

    assert test_images.shape[1:] == (3, 224, 224)
    assert test_labels.dtype == torch.int64

    print(f"\n{model_type}: ALL CHECKS PASSED!")


def main():

    check_model_pipeline("scratch_cnn")

    check_model_pipeline("resnet18")

    print("\n================================")
    print("FULL DATA PIPELINE TEST PASSED!")
    print("================================")


if __name__ == "__main__":
    main()