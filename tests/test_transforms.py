import torch
from PIL import Image

from src.data.transforms import get_transforms


def test_preprocessing():

    # Create a sample RGB image
    image = Image.new(
        "RGB",
        (500, 375),
        color=(120, 180, 90)
    )

    # Test CNN preprocessing
    cnn_transform = get_transforms("scratch_cnn")

    cnn_image = cnn_transform(image)

    print("\nCNN preprocessing:")
    print("Shape:", cnn_image.shape)
    print("Data type:", cnn_image.dtype)
    print("Minimum:", cnn_image.min().item())
    print("Maximum:", cnn_image.max().item())

    assert cnn_image.shape == (3, 224, 224)
    assert cnn_image.dtype == torch.float32
    assert torch.isfinite(cnn_image).all()

    print("CNN preprocessing passed!")

    # Test ResNet18 preprocessing
    resnet_transform = get_transforms("resnet18")

    resnet_image = resnet_transform(image)

    print("\nResNet18 preprocessing:")
    print("Shape:", resnet_image.shape)
    print("Data type:", resnet_image.dtype)

    assert resnet_image.shape == (3, 224, 224)
    assert resnet_image.dtype == torch.float32
    assert torch.isfinite(resnet_image).all()

    print("ResNet18 preprocessing passed!")

    # Test grayscale input
    grayscale_image = Image.new(
        "L",
        (300, 400),
        color=120
    )

    grayscale_output = cnn_transform(grayscale_image)

    assert grayscale_output.shape == (3, 224, 224)

    print("\nGrayscale-to-RGB conversion passed!")

    # Test invalid model type
    try:
        get_transforms("invalid_model")
    except ValueError:
        print("Invalid model validation passed!")
    else:
        raise AssertionError(
            "Expected ValueError for invalid model type."
        )

    print("\nALL PREPROCESSING TESTS PASSED!")


if __name__ == "__main__":
    test_preprocessing()