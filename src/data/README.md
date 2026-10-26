# CUB-200-2011 Data Pipeline

## Overview

This module provides the data pipeline for the CUB-200-2011 bird dataset. It covers dataset loading,
stratified splitting, image preprocessing, augmentation, caption cleaning, and PyTorch DataLoaders.

It supports both a CNN trained from scratch and a pretrained ResNet18 model. Both get exactly the same
train / validation / test images.

Every preprocessing choice comes from the EDA. The evidence is in [`docs/eda_findings.md`](../../docs/eda_findings.md),
and the EDA itself (`src/data/eda.py`) loads the data through this same module.

## Dataset

Hugging Face datasets, pinned to a commit in [`configs/data.yaml`](../../configs/data.yaml):

| Split | Repository | Revision | Images |
| --- | --- | --- | ---: |
| Train | `Multimodal-Fatima/CUB_train` | `bd3988995e6db193427255abdebd522afe24ce76` | 5,994 |
| Test | `Multimodal-Fatima/CUB_test` | `82447a42c5e05c7fe21cd36accfdf399148e8668` | 5,794 |

Columns: `image` (PIL), `description` (10 captions separated by `\n`), `label` (ClassLabel, 0–199, identical
200 names in both splits), `file_name`, `id`.

* `file_name` is the unique image key. `id` is only the row number inside each split, so it cannot be used
  to compare splits.
* The Hugging Face split is the official CUB split: 0 differences for all 11,788 images.
* No missing values and no decode errors were found. 8 images are single-channel, and every image is converted
  to RGB on load.

## Dataset Splitting

The official training set is divided into training and validation subsets with a stratified split. The official
test set is used only for final evaluation and is never modified.

* Validation percentage: 10%, stratified by class
* Random seed: 42
* Split indices: [`configs/split_indices.json`](../../configs/split_indices.json), version-controlled

The manifest also stores the dataset revision, row count and a SHA-256 hash of the ordered file names.
Loading it against a dataset with a different row order raises an error, so saved indices are never applied
silently to the wrong images.

**Leakage exclusions.** 3 training images are copies of test images (`exclusions.train_leakage` in
`configs/data.yaml`). They are removed by `file_name` after the manifest is loaded, from whichever partition they
landed in. The saved indices therefore stay valid. 2 of them were in validation and 1 in train.

| Subset | Images | Per class |
| --- | ---: | --- |
| Training | 5,393 | 26–27 |
| Validation | 598 | 2–3 |
| Excluded (leakage) | 3 | |
| Testing | 5,794 | 11–30 |

5,393 + 598 + 3 = 5,994. The training data is balanced, so no class weights are needed. The test set is
imbalanced, so report macro-F1 alongside accuracy.

## Image Preprocessing

Images are mostly 4:3 (median 500 × 375), and 148 have a shorter side below 224 px. All images are converted to
RGB and become 224 × 224 tensors.

| | Transform |
| --- | --- |
| Training | `RandomResizedCrop(224, scale=(0.6, 1.0))` → `RandomHorizontalFlip()` → `RandomRotation(15)` → `ColorJitter(0.2, 0.2, 0.2, hue=0)` → `ToTensor` → `Normalize` |
| Validation / test | `Resize(224)` (shorter side) → `CenterCrop(224)` → `ToTensor` → `Normalize` |

* `Resize(224)` + `CenterCrop(224)` keeps 94% of the bird on average. `Resize(256)` + `CenterCrop(224)` cuts part
  of the bird in two thirds of the images.
* Validation and test preprocessing has no randomness.

## Normalization

| Model | Mean | Standard deviation | Source |
| --- | --- | --- | --- |
| `scratch_cnn` | `(0.4849, 0.4990, 0.4324)` | `(0.2271, 0.2225, 0.2613)` | CUB training partition (`outputs/eda/summary.json` → `images.normalization_train_partition`) |
| `resnet18` | `(0.485, 0.456, 0.406)` | `(0.229, 0.224, 0.225)` | ImageNet, as the network was pretrained |

`get_normalization(model_type)` returns these values. `augmentation.denormalize_image` uses it to turn tensors
back into images for plots.

## Data Augmentation

The augmentation applies to training images only.

| Augmentation | Why |
| --- | --- |
| Random resized crop, scale 0.6–1.0 | The bird fills a median of 32% of the image. At the default scale of 0.08, 33% of crops lose more than half the bird; at 0.6, only 0.22% do |
| Horizontal flip | Birds look the same mirrored; only 0.04% of captions mention left or right |
| Rotation ±15° | Small camera or perch tilts. A larger angle leaves large black corners |
| Brightness, contrast, saturation ±0.2 | Brightness varies from 0.08 to 0.96 across photos |
| No hue jitter, no vertical flip | Color identifies the species and appears in 97.6% of captions. Upside-down birds never occur |

## Caption Preprocessing

The raw `description` is never modified. Cleaned captions are a derived field.

| Function | Result |
| --- | --- |
| `preprocess_captions(description)` | Raw captions: split on `\n`, stripped, whitespace collapsed, empty lines dropped. Case and punctuation are kept. Always 10 per image |
| `clean_caption(text)` | One normalized caption: U+FFFD → space, lowercase, apostrophes removed (`bird's` → `birds`), other punctuation → space, whitespace collapsed. The EDA uses the same function |
| `clean_captions(description)` | Model-ready captions: cleaned, captions under 4 tokens dropped, exact repeats within the image dropped. 7–10 per image |
| `validate_captions(captions)` | Checks for 10 non-empty captions |
| `build_vocabulary(captions, min_freq=5)` | `<pad>`, `<start>`, `<end>`, `<unk>`, then the words by frequency. Pass training-partition captions only |

Over all 11,788 images, `clean_captions` drops 3 captions with fewer than 4 tokens and 1,312 repeats, and keeps
116,565 of the 117,880 raw captions.

Constants: `MIN_TOKENS = 4`, `MIN_FREQ = 5` (1,186 words on the training partition), and `MAX_LEN = 26`
(24 tokens + `<start>`/`<end>`; about 5% of captions are truncated).

## PyTorch Dataset

`CUBDataset(hf_dataset, transform, return_captions=False, return_file_name=False)` returns `(image, label)`:
`image` is a `float32` tensor `[3, 224, 224]` and `label` is an `int` in 0–199.

* `return_captions=True` appends the list of cleaned captions.
* `return_file_name=True` appends the file name, for error analysis.

Batches with captions need `collate_fn=collate_with_captions` (from `dataloaders.py`), because images have
different numbers of captions.

## DataLoader Usage

### CNN From Scratch (Member 3)

```python
from src.data.dataloaders import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    model_type="scratch_cnn",
    batch_size=32
)
```

### Pretrained ResNet18 (Member 4)

```python
from src.data.dataloaders import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    model_type="resnet18",
    batch_size=32
)
```

Each batch is `(images, labels)`, with `images` of shape `[batch_size, 3, 224, 224]` (`torch.float32`) and
`labels` of shape `[batch_size]` (`torch.int64`).

Other arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `image_size` | `224` | Output height and width |
| `num_workers` | `0` | Worker processes. Values above 0 work on Windows too: datasets and transforms can be pickled |
| `seed` | `42` | Seeds the shuffle order and the worker seeds |
| `return_file_name` | `False` | Batches become `(images, labels, file_names)` |
| `split_path`, `config_path` | `configs/...` | Alternative split manifest or data config |

* Training is shuffled and augmented.
* Validation and test use `shuffle=False` and deterministic transforms.
* `create_datasets(...)` returns the three `CUBDataset` objects without wrapping them in DataLoaders.
* `load_partitions()` (in `splitting.py`) returns the underlying Hugging Face datasets, for example to build
  the caption vocabulary from `partitions["train"]["description"]`.

## Reproducibility

* The dataset revision is pinned, and the split manifest is tied to the exact row order.
  `python check_dataset.py` regenerates the split from the seed and stops if it differs from the saved file.
* With `num_workers=0`, augmentation draws from the global torch random generator. Call
  `torch.manual_seed(seed)` at the start of training.
* Each DataLoader has its own seeded generator, so running validation does not change the training augmentation
  that follows.
* For identical results, also use the same software environment (`requirements.txt`).

## Testing

Run all tests from the project root:

```bash
python -m pytest tests/ -v                    # all tests
python -m pytest tests/ -v -m "not integration"   # offline unit tests only (small synthetic dataset)
```

* Offline unit tests use a small synthetic dataset.
* Tests marked `integration` load the real CUB data. The first run downloads it from Hugging Face; later runs
  use the local cache.

Pipeline checks on the real data:

```bash
python check_dataset.py         # split reproducibility and exclusions
python check_label_mapping.py   # same classes in train and test, no shared file names
python check_captions.py        # 10 raw captions per image, cleaned counts
python check_pipeline.py        # batches for both models, also with num_workers=2
python check_augmentation.py    # saves augmentation examples to outputs/data/
```

## Project Files

| File | Purpose |
| --- | --- |
| `dataset.py` | Data config, pinned dataset loading, dataset fingerprint, `CUBDataset` |
| `splitting.py` | Stratified split, manifest validation, leakage exclusions, `load_partitions()` |
| `transforms.py` | Image preprocessing, augmentation and normalization |
| `augmentation.py` | Augmentation visualization |
| `captions.py` | Caption splitting, cleaning, validation and vocabulary |
| `dataloaders.py` | `create_datasets()`, `create_dataloaders()`, `collate_with_captions()` |
| `eda.py` | Exploratory data analysis (`python -m src.data.eda --out outputs/eda`) |

`configs/data.yaml` also lists the 9 grayscale images to exclude from colorization (Milestone 2), and one test photo
that appears under two species. Both lists are recorded only; classification keeps all of those images.

## Integration

This module is used by:

* Member 3: CNN trained from scratch (`model_type="scratch_cnn"`).
* Member 4: ResNet18 transfer learning (`model_type="resnet18"`).

Both members import `create_dataloaders()` and need no dataset-loading or preprocessing code of their own.
