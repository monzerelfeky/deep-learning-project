# CUB-200-2011 Data Preprocessing Pipeline

## Overview

This module provides a reusable data preprocessing pipeline for the CUB-200-2011 bird classification dataset.

It supports both a CNN trained from scratch and a pretrained ResNet18 model.

The pipeline includes dataset loading, stratified splitting, image preprocessing, augmentation, caption cleaning, and PyTorch DataLoaders.

## Dataset

The project uses the CUB-200-2011 dataset available through Hugging Face.

* Original training images: 5,994
* Official testing images: 5,794
* Bird classes: 200
* Captions: 10 descriptions per image, expected

The original training dataset is divided into training and validation subsets using stratified sampling.

## Dataset Splitting

The dataset is divided using:

* Training percentage: 90%
* Validation percentage: 10%
* Random seed: 42

The resulting subsets contain:

| Subset     | Images |
| ---------- | -----: |
| Training   |  5,394 |
| Validation |    600 |
| Testing    |  5,794 |

The official test dataset is kept separate from training and validation.

The split indices are stored in `configs/split_indices.json` to ensure that different models use identical partitions.

## Image Preprocessing

All images are converted to RGB and resized or cropped to 224 × 224 pixels.

The pipeline provides model-specific normalization:

**Scratch CNN**

Mean: `(0.5, 0.5, 0.5)`

Standard deviation: `(0.5, 0.5, 0.5)`

**ResNet18**

ImageNet mean: `(0.485, 0.456, 0.406)`

ImageNet standard deviation: `(0.229, 0.224, 0.225)`

## Data Augmentation

Training images undergo random augmentation, including:

* Random resized cropping
* Horizontal flipping
* Small rotations
* Brightness, contrast, and saturation adjustments

Validation and testing images use deterministic preprocessing without random augmentation.

## Caption Preprocessing

The `description` field contains multiple image captions.

The caption preprocessing module:

* Separates captions using newline characters.
* Removes leading and trailing whitespace.
* Normalizes repeated whitespace.
* Removes empty captions.
* Validates the expected number of captions.

The original caption content is preserved without aggressive text cleaning.

## PyTorch Dataset

The `CUBDataset` class provides a standard PyTorch Dataset interface.

For classification, each sample returns:

```python
image, label
```

The image is a transformed tensor, and the label is an integer class identifier.

Caption retrieval is also supported using `return_captions=True`.

## DataLoader Usage

### CNN From Scratch

```python
from src.data.dataloaders import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    model_type="scratch_cnn",
    batch_size=32
)
```

### Pretrained ResNet18

```python
from src.data.dataloaders import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    model_type="resnet18",
    batch_size=32
)
```

Both configurations use the same saved dataset partitions.

## Testing

The preprocessing pipeline includes tests for:

* Image transformations
* Data augmentation
* Caption preprocessing
* Dataset split integrity
* Split reproducibility

Run all tests from the project root:

```bash
python -m pytest tests/ -v
```

Run the complete pipeline smoke test:

```bash
python check_pipeline.py
```

## Project Files

| File              | Purpose                                       |
| ----------------- | --------------------------------------------- |
| `dataset.py`      | Dataset loading and PyTorch Dataset wrapper   |
| `splitting.py`    | Stratified train/validation splitting         |
| `transforms.py`   | Image preprocessing and normalization         |
| `augmentation.py` | Augmentation visualization                    |
| `captions.py`     | Caption cleaning and validation               |
| `dataloaders.py`  | Training, validation, and testing DataLoaders |

## Reproducibility

The pipeline uses a fixed random seed for dataset splitting and DataLoader shuffling.

The saved split indices allow the scratch CNN and ResNet18 models to use identical training and validation samples.

For full reproducibility, the same dataset version, software environment, and random-number-generator settings must also be used.

## Integration

This module is designed for use by:

* Member 3: CNN trained from scratch.
* Member 4: ResNet18 transfer learning.

Both members can import `create_dataloaders()` without implementing separate dataset-loading or preprocessing logic.
