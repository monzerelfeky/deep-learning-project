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

def test_training_pipeline_matches_eda_decisions():

    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    steps = get_transforms("scratch_cnn", is_training=True).transforms
    by_type = {type(step): step for step in steps}

    crop = by_type[T.RandomResizedCrop]
    assert crop.size == (224, 224)
    assert crop.scale == (0.6, 1.0)
    assert crop.ratio == (3 / 4, 4 / 3)

    assert by_type[T.RandomHorizontalFlip].p == 0.5

    rotation = by_type[T.RandomRotation]
    assert rotation.degrees == [-15.0, 15.0]
    assert rotation.interpolation == InterpolationMode.BILINEAR

    jitter = by_type[T.ColorJitter]
    assert tuple(jitter.brightness) == (0.8, 1.2)
    assert tuple(jitter.contrast) == (0.8, 1.2)
    assert tuple(jitter.saturation) == (0.8, 1.2)
    assert jitter.hue is None

    assert T.RandomVerticalFlip not in by_type


def test_eval_pipeline_is_resize_then_center_crop():

    from torchvision import transforms as T

    steps = get_transforms("resnet18", is_training=False).transforms
    names = [type(step).__name__ for step in steps]

    assert names == ["ToRGB", "Resize", "CenterCrop", "ToTensor", "Normalize"]
    assert steps[1].size == 224
    assert steps[2].size == (224, 224)

    transform = get_transforms("resnet18", is_training=False)

    # Landscape, portrait and an image smaller than 224 px
    for size in [(500, 375), (375, 500), (121, 120)]:
        output = transform(Image.new("RGB", size, color=(10, 200, 30)))
        assert output.shape == (3, 224, 224)

    assert T.RandomResizedCrop not in map(type, steps)


def test_eval_keeps_the_center_of_the_image():

    # A 4:3 image with black bars on the left and right: Resize(224)
    # + CenterCrop(224) keeps the full height and cuts the bars off,
    # instead of squashing the whole image to a square
    image = Image.new("RGB", (400, 300), color=(0, 0, 0))
    image.paste((255, 255, 255), (40, 0, 360, 300))

    output = get_transforms("scratch_cnn")(image)
    white = get_transforms("scratch_cnn")(Image.new("RGB", (300, 300), color=(255, 255, 255)))

    assert torch.allclose(output, white)


def test_normalization_per_model():

    from src.data.transforms import (
        CNN_MEAN,
        CNN_STD,
        IMAGENET_MEAN,
        IMAGENET_STD,
        get_normalization
    )

    assert get_normalization("resnet18") == (IMAGENET_MEAN, IMAGENET_STD)
    assert get_normalization("scratch_cnn") == (CNN_MEAN, CNN_STD)

    for model_type in ("scratch_cnn", "resnet18"):
        for is_training in (True, False):
            normalize = get_transforms(model_type, is_training=is_training).transforms[-1]
            assert (tuple(normalize.mean), tuple(normalize.std)) == get_normalization(model_type)


def test_cnn_normalization_matches_eda_summary():

    import json
    from pathlib import Path

    from src.data.transforms import CNN_MEAN, CNN_STD

    summary_path = Path(__file__).resolve().parents[1] / "outputs" / "eda" / "summary.json"
    stats = json.loads(summary_path.read_text())["images"]["normalization_train_partition"]

    assert list(CNN_MEAN) == stats["mean"]
    assert list(CNN_STD) == stats["std"]


def test_transforms_are_picklable():

    import pickle

    image = Image.new("L", (300, 200), color=90)

    for model_type in ("scratch_cnn", "resnet18"):
        for is_training in (True, False):
            transform = pickle.loads(pickle.dumps(get_transforms(model_type, is_training=is_training)))
            assert transform(image).shape == (3, 224, 224)


def test_denormalize_inverts_normalization():

    from src.data.augmentation import denormalize_image

    image = Image.new("RGB", (300, 300), color=(51, 102, 204))
    expected = torch.tensor([51, 102, 204]).float()[:, None, None] / 255

    for model_type in ("scratch_cnn", "resnet18"):
        restored = denormalize_image(get_transforms(model_type)(image), model_type)
        assert torch.allclose(restored, expected.expand_as(restored), atol=1e-5)


if __name__ == "__main__":
    test_preprocessing()
