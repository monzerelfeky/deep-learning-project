from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ImageNet normalization (ResNet18 was pretrained with it)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CNN normalization: CUB statistics of the training partition
# (outputs/eda/summary.json, images.normalization_train_partition)
CNN_MEAN = (0.4849, 0.4990, 0.4324)
CNN_STD = (0.2271, 0.2225, 0.2613)

NORMALIZATION = {
    "scratch_cnn": (CNN_MEAN, CNN_STD),
    "resnet18": (IMAGENET_MEAN, IMAGENET_STD),
}


class ToRGB:
    """
    Convert a PIL image to RGB (8 CUB images are single-channel).

    A class instead of a lambda so the transform can be pickled
    by DataLoader workers on Windows.
    """

    def __call__(self, image):
        return image.convert("RGB")

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def get_normalization(model_type="scratch_cnn"):
    """
    Return (mean, std) used to normalize images for model_type.
    """

    if model_type not in NORMALIZATION:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            "Choose 'scratch_cnn' or 'resnet18'."
        )

    return NORMALIZATION[model_type]


def get_transforms(
    model_type="scratch_cnn",
    image_size=224,
    is_training=False
):
    """
    Create preprocessing and augmentation transforms.

    Parameters
    ----------
    model_type : str
        'scratch_cnn' or 'resnet18'.

    image_size : int
        Output image height and width.

    is_training : bool
        Whether to apply random training augmentation.

    Returns
    -------
    torchvision.transforms.Compose
        Image transformation pipeline.
    """

    mean, std = get_normalization(model_type)

    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer.")

    # Convert all images to RGB
    transform_list = [ToRGB()]

    if is_training:

        # A lower scale bound of 0.6 keeps most of the bird:
        # only 0.22% of crops lose more than half of it
        transform_list.append(
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.6, 1.0),
                interpolation=InterpolationMode.BILINEAR
            )
        )

        # Horizontal flipping
        transform_list.append(
            transforms.RandomHorizontalFlip(p=0.5)
        )

        # Small rotations
        transform_list.append(
            transforms.RandomRotation(
                degrees=15,
                interpolation=InterpolationMode.BILINEAR
            )
        )

        # No hue jitter: a hue shift can turn one species into another
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.0
            )
        )

    else:

        # Deterministic preprocessing for validation/test.
        # Resizing the shorter side to the crop size keeps
        # more of the bird than Resize(256) + CenterCrop(224)
        transform_list.append(
            transforms.Resize(
                image_size,
                interpolation=InterpolationMode.BILINEAR
            )
        )

        transform_list.append(
            transforms.CenterCrop(image_size)
        )

    # Convert to tensor
    transform_list.append(
        transforms.ToTensor()
    )

    # Apply model-specific normalization
    transform_list.append(
        transforms.Normalize(
            mean=mean,
            std=std
        )
    )

    return transforms.Compose(transform_list)
