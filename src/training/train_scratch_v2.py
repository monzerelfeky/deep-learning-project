
"""
Train the residual scratch CNN (V2) on CUB-200-2011.

This experiment is independent of V1:
- Same shared data pipeline
- AdamW optimizer
- ReduceLROnPlateau learning-rate scheduler
- Separate checkpoints and training history
- Automatic resume support
"""

from pathlib import Path
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataloaders import create_dataloaders
from src.models.classifier import BirdClassifier
from src.models.scratch_cnn_v2 import ScratchCNNEncoderV2
from src.training.train_scratch import (
    train_one_epoch,
    validate_one_epoch,
    save_epoch_metrics,
)


def main():
    # Experiment configuration
    seed = 42
    batch_size = 4
    image_size = 224
    num_classes = 200
    num_epochs = 20

    learning_rate = 0.001
    weight_decay = 0.0001

    experiment_dir = Path(
        "checkpoints/scratch_cnn/experiment_02"
    )
    history_path = Path(
        "outputs/training/scratch_cnn_experiment_02.csv"
    )

    best_path = experiment_dir / "best_model.pt"
    last_path = experiment_dir / "last_model.pt"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("SCRATCH CNN V2 — RESIDUAL CNN TRAINING")
    print("=" * 55)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")

    # Use the unchanged team DataLoader.
    train_loader, val_loader, _ = create_dataloaders(
        model_type="scratch_cnn",
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,
        seed=seed,
    )

    model = BirdClassifier(
        encoder=ScratchCNNEncoderV2(),
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Reduce LR when validation loss stops improving.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    config = {
        "architecture": "ScratchCNNEncoderV2",
        "batch_size": batch_size,
        "image_size": image_size,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
    }

    start_epoch = 1
    best_f1 = -1.0

    if last_path.exists():
        checkpoint = torch.load(
            last_path,
            map_location=device,
            weights_only=True,
        )

        if checkpoint.get("config") != config:
            raise ValueError(
                "Experiment configuration differs from checkpoint."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_val_f1"]

        # Confirm history belongs to the resumed checkpoint.
        if not history_path.exists():
            raise FileNotFoundError(
                f"Missing experiment history: {history_path}"
            )

        with history_path.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            recorded_epochs = [
                int(row["epoch"])
                for row in csv.DictReader(file)
            ]

        expected_epochs = list(range(1, start_epoch))

        if recorded_epochs != expected_epochs:
            raise RuntimeError(
                "Checkpoint and CSV history are inconsistent. "
                f"Recorded: {recorded_epochs}; "
                f"Expected: {expected_epochs}"
            )

        print(f"Resumed from epoch {start_epoch - 1}")
        print(f"Previous best macro-F1: {best_f1:.4f}")

    elif best_path.exists() or history_path.exists():
        raise FileExistsError(
            "Existing Experiment 02 artifacts found without "
            "a resumable checkpoint."
        )

    if start_epoch > num_epochs:
        print("Training already completed.")
        return

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'-' * 55}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'-' * 55}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy, val_f1 = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Adjust LR based on validation loss.
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2%}"
        )
        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2%} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )
        print(f"Learning Rate: {current_lr:.6f}")

        is_best = val_f1 > best_f1

        if is_best:
            best_f1 = val_f1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1": best_f1,
            "config": config,
        }

        if is_best:
            torch.save(checkpoint, best_path)
            print(f"New best model! Macro-F1: {best_f1:.4f}")

        save_epoch_metrics(
            csv_path=history_path,
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            val_f1=val_f1,
        )

        torch.save(checkpoint, last_path)
        print("Epoch metrics and checkpoint saved.")

    print("\nTraining completed.")
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
