from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CNN normalization
CNN_MEAN = (0.5, 0.5, 0.5)
CNN_STD = (0.5, 0.5, 0.5)


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

    if model_type == "scratch_cnn":
        mean = CNN_MEAN
        std = CNN_STD

    elif model_type == "resnet18":
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            "Choose 'scratch_cnn' or 'resnet18'."
        )

    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer.")

    # Convert all images to RGB
    transform_list = [
        transforms.Lambda(lambda image: image.convert("RGB"))
    ]

    if is_training:

        # Randomly crop and resize the image
        transform_list.append(
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.85, 1.0),
                ratio=(0.90, 1.10),
                interpolation=InterpolationMode.BILINEAR
            )
        )

        # Horizontal flipping
        transform_list.append(
            transforms.RandomHorizontalFlip(p=0.5)
        )

        # Small rotations
        transform_list.append(
            transforms.RandomRotation(degrees=15)
        )

        # Mild color adjustments
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.0
            )
        )

    else:

        # Deterministic preprocessing for validation/test
        transform_list.append(
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR
            )
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