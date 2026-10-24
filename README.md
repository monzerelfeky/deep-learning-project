# From Pixels to Language — CUB-200-2011

CSE 488 Deep Learning final project (Ain Shams University).

One shared CNN **encoder** is trained in Phase 1 (classification) and reused in Phase 2
(colorization and captioning).

```
CUB-200-2011 images ──► Encoder (scratch CNN | ResNet18) ──► feature maps [B, C, H, W]
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
      Classification head    Colorization decoder    Caption decoder
          (Phase 1)              (Phase 2)              (Phase 2)
```

## Repository layout

```
configs/                 YAML experiment configs (one file per experiment family)
data/                    Dataset goes here (git-ignored) — see data/README.md
notebooks/eda.ipynb      Exploratory data analysis
outputs/                 Checkpoints, logs, figures (git-ignored)
src/
├── cli.py               Entry point: train / evaluate / predict / tune / gradcam
├── data/
│   ├── dataset.py       CUBDataset, split creation, DataLoader factory
│   ├── transforms.py    Preprocessing + augmentation pipelines
│   └── captions.py      Caption loading + cleaning (stats in Phase 1, used in Phase 2)
├── models/
│   ├── encoder.py       BaseEncoder — the contract every backbone implements
│   ├── scratch_cnn.py   4-block CNN trained from scratch
│   ├── transfer.py      ResNet18 (ImageNet-pretrained) encoder
│   ├── classifier.py    Encoder + linear head → logits [B, 200]
│   └── factory.py       build_model(config) — the only place models are chosen
├── training/
│   ├── trainer.py       Trainer: fit / train_one_epoch / validate / checkpoints
│   └── tuning.py        Grid Search + Random Search
├── evaluation/
│   ├── metrics.py       Accuracy, macro F1/precision/recall, confusion matrix
│   └── evaluate.py      Run a model over a loader → predictions + metrics
├── visualization/
│   ├── gradcam.py       Grad-CAM heatmaps
│   └── plots.py         Training curves, confusion matrix, EDA figures
└── utils/
    ├── config.py        YAML loading + CLI overrides
    ├── reproducibility.py  Seeding, device selection
    └── tracking.py      MLflow wrapper
tests/test_contracts.py  Shape/interface checks every module must pass
```

## Shared interfaces (agree before coding)

| Contract | Definition |
| --- | --- |
| Dataset item | `CUBDataset[i] -> (image: FloatTensor[3, 224, 224], label: int in [0, 199])` |
| Encoder | `encoder.forward_features(x) -> [B, C, H, W]`, `encoder(x) -> [B, encoder.out_dim]`, `encoder.gradcam_layer` |
| Classifier | `model(x) -> logits [B, 200]` |
| Trainer | `Trainer(model, loaders, optimizer, config).fit() -> history dict` |
| Metrics | `compute_metrics(y_true, y_pred) -> {"accuracy", "macro_f1", "macro_precision", "macro_recall"}` |
| Tracking | Every run logs its full config, per-epoch metrics, and the best checkpoint path |

## Data split

- Official CUB split: 5,994 train / 5,794 test. The **test set is used only for final evaluation**.
- Validation: stratified 10% of the training set (`data.val_fraction`), fixed by `seed`.
- Split indices are written to `outputs/splits/` so every member uses identical splits.

## Usage

```bash
pip install -r requirements.txt

python -m src.cli train    --config configs/scratch.yaml
python -m src.cli train    --config configs/transfer.yaml --set train.lr=0.0001
python -m src.cli evaluate --config configs/transfer.yaml --checkpoint outputs/<run>/best.pt
python -m src.cli predict  --config configs/transfer.yaml --checkpoint outputs/<run>/best.pt --image bird.jpg
python -m src.cli tune     --config configs/tuning.yaml --strategy grid
python -m src.cli gradcam  --config configs/transfer.yaml --checkpoint outputs/<run>/best.pt --num-samples 8

pytest tests/
```

## Phase 1 experiments

| ID | Experiment | Config |
| --- | --- | --- |
| E1 | Scratch CNN baseline | `scratch.yaml` with `data.augment=false` |
| E2 | Scratch CNN + augmentation | `scratch.yaml` |
| E3 | ResNet18 frozen backbone | `transfer.yaml` with `model.freeze_backbone=true` |
| E4 | ResNet18 fine-tuning | `transfer.yaml` |
| E5 | Grid Search | `tuning.yaml --strategy grid` |
| E6 | Random Search | `tuning.yaml --strategy random` |

## Git workflow

- Branch per module: `feature/data`, `feature/scratch-cnn`, `feature/transfer`, `feature/training`,
  `feature/evaluation`, `feature/gradcam-cli`.
- Pull request into `main`, reviewed by one other member.
- Never commit the dataset or checkpoints; share them via the team Drive folder.
