"""Train frozen-feature or fine-tuned ResNet18 on CUB-200-2011.

Examples (run from the repository root):
    python -m src.training.train_resnet18 --mode frozen
    python -m src.training.train_resnet18 --mode finetune

A short pipeline smoke test (not a meaningful experiment):
    python -m src.training.train_resnet18 --mode frozen --epochs 1 \
        --max-train-batches 2 --max-val-batches 2 --run-name frozen_smoke --no-pretrained
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.models.classifier import BirdClassifier
from src.models.transfer import ResNet18Encoder

try:
    import mlflow
except ImportError:  # CSV/checkpoint logging still works without optional local setup.
    mlflow = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transfer.yaml")
    parser.add_argument("--mode", choices=("frozen", "finetune"), required=True)
    parser.add_argument("--run-name", default=None,
                        help="Unique run name; defaults to the selected mode.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override configured epoch count.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override configured batch size.")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Optional training batches per epoch, for smoke tests only.")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Optional validation batches per epoch, for smoke tests only.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download ImageNet weights; for smoke tests only.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_one_epoch_limited(model, loader, criterion, device, max_batches=None):
    """Validation matching the metric definition, optionally batch-limited."""
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be a positive integer or None.")

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(
            tqdm(loader, desc="Validation", leave=False)
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            count = labels.size(0)
            running_loss += loss.item() * count
            correct += (predictions == labels).sum().item()
            total += count
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if total == 0:
        raise RuntimeError("Validation processed no samples; check the DataLoader.")

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(200)),
        average="macro",
        zero_division=0,
    )
    return running_loss / total, correct / total, macro_f1


def _mlflow_context(run_name: str):
    if mlflow is None:
        print("MLflow is not installed; metrics will still be saved to CSV and checkpoints.")
        return nullcontext()
    tracking_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("cub200-resnet18-transfer-learning")
    return mlflow.start_run(run_name=run_name)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        full_config = yaml.safe_load(file)

    project = full_config["project"]
    data_config = full_config["data"]
    experiment = full_config["experiments"][args.mode]

    seed = int(project.get("seed", 42))
    epochs = args.epochs if args.epochs is not None else int(experiment["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(data_config["batch_size"])
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers.")
    for name, value in (("max_train_batches", args.max_train_batches),
                        ("max_val_batches", args.max_val_batches)):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be a positive integer or omitted.")

    run_name = args.run_name or f"resnet18_{args.mode}"
    pretrained = bool(experiment.get("pretrained", True)) and not args.no_pretrained
    freeze_backbone = args.mode == "frozen"
    image_size = int(project.get("image_size", 224))
    num_classes = int(project.get("num_classes", 200))
    backbone_lr = float(experiment["backbone_learning_rate"])
    head_lr = float(experiment["head_learning_rate"])
    weight_decay = float(experiment.get("weight_decay", 0.0001))
    seed_everything(seed)

    # Import dataset/training helpers only after argparse has handled --help.
    from src.data.dataloaders import create_dataloaders
    from src.training.train_scratch import train_one_epoch, save_epoch_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run: {run_name} | mode={args.mode} | device={device}")
    print(f"Pretrained ImageNet weights: {pretrained}")
    print(f"Batch size: {batch_size} | epochs: {epochs} | seed: {seed}")

    # Shared team split and model-specific ImageNet normalization.
    train_loader, val_loader, _ = create_dataloaders(
        model_type=data_config.get("model_type", "resnet18"),
        batch_size=batch_size,
        image_size=image_size,
        num_workers=int(data_config.get("num_workers", 0)),
        split_path=data_config.get("split_config"),
        seed=seed,
    )

    encoder = ResNet18Encoder(
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    model = BirdClassifier(encoder=encoder, num_classes=num_classes).to(device)

    if freeze_backbone:
        optimizer = AdamW(
            model.classifier.parameters(),
            lr=head_lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = AdamW(
            [
                {"params": model.encoder.parameters(), "lr": backbone_lr},
                {"params": model.classifier.parameters(), "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )

    scheduler_config = experiment.get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config.get("factor", 0.5)),
        patience=int(scheduler_config.get("patience", 3)),
        min_lr=float(scheduler_config.get("min_lr", 1e-6)),
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path("checkpoints") / "resnet18" / run_name
    best_path = checkpoint_dir / "best_model.pt"
    last_path = checkpoint_dir / "last_model.pt"
    history_path = Path("outputs") / "training" / f"{run_name}.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid accidentally mixing metrics from a previous run with new settings.
    if best_path.exists() or last_path.exists() or history_path.exists():
        raise FileExistsError(
            f"Artifacts already exist for run '{run_name}'. Choose a new --run-name."
        )

    run_config = {
        "architecture": "ResNet18Encoder",
        "mode": args.mode,
        "pretrained_imagenet": pretrained,
        "freeze_backbone": freeze_backbone,
        "num_classes": num_classes,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "head_learning_rate": head_lr,
        "backbone_learning_rate": backbone_lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "selection_metric": "val_macro_f1",
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
    }

    with _mlflow_context(run_name):
        if mlflow is not None:
            mlflow.log_params({key: str(value) for key, value in run_config.items()})
            mlflow.log_param("best_checkpoint_path", str(best_path))

        best_f1 = -1.0
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            train_loss, train_accuracy = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                max_batches=args.max_train_batches,
            )
            val_loss, val_accuracy, val_f1 = validate_one_epoch_limited(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                max_batches=args.max_val_batches,
            )
            scheduler.step(val_loss)
            current_lrs = [group["lr"] for group in optimizer.param_groups]

            print(
                f"Train loss={train_loss:.4f}, accuracy={train_accuracy:.4f} | "
                f"Val loss={val_loss:.4f}, accuracy={val_accuracy:.4f}, "
                f"macro-F1={val_f1:.4f} | LR={current_lrs}"
            )

            is_best = val_f1 > best_f1
            if is_best:
                best_f1 = val_f1

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_macro_f1": best_f1,
                "config": run_config,
            }
            if is_best:
                torch.save(checkpoint, best_path)
            torch.save(checkpoint, last_path)

            save_epoch_metrics(
                csv_path=history_path,
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                val_f1=val_f1,
            )
            if mlflow is not None:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "val_macro_f1": val_f1,
                        "best_val_macro_f1": best_f1,
                        "learning_rate_backbone": current_lrs[0] if not freeze_backbone else 0.0,
                        "learning_rate_head": current_lrs[-1],
                    },
                    step=epoch,
                )

        if mlflow is not None:
            mlflow.log_artifact(str(history_path), artifact_path="history")

    summary = {
        "run_name": run_name,
        "mode": args.mode,
        "best_val_macro_f1": best_f1,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_csv": str(history_path),
        "config": run_config,
    }
    summary_path = history_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTraining complete.")
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Epoch history: {history_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
