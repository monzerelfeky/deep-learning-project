import torch
from PIL import Image

from src.data.transforms import get_transforms


def test_training_augmentation():

    image = Image.new(
        "RGB",
        (500, 375),
        color=(120, 180, 90)
    )

    transform = get_transforms(
        model_type="scratch_cnn",
        is_training=True
    )

    for _ in range(10):

        output = transform(image)

        assert output.shape == (3, 224, 224)

        assert output.dtype == torch.float32

        assert torch.isfinite(output).all()


def test_validation_is_deterministic():

    image = Image.new(
        "RGB",
        (500, 375),
        color=(120, 180, 90)
    )

    transform = get_transforms(
        model_type="scratch_cnn",
        is_training=False
    )

    output1 = transform(image)
    output2 = transform(image)

    assert torch.equal(output1, output2)


def test_resnet_augmentation():

    image = Image.new(
        "RGB",
        (500, 375),
        color=(120, 180, 90)
    )

    transform = get_transforms(
        model_type="resnet18",
        is_training=True
    )

    output = transform(image)

    assert output.shape == (3, 224, 224)
    assert torch.isfinite(output).all()