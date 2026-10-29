
from pathlib import Path
import csv
import random
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.data.dataloaders import create_dataloaders
from src.models.scratch_cnn import ScratchCNNEncoder
from src.models.classifier import BirdClassifier


# ------------------------------------------------------------
# Initial Training Configuration and Design Rationale
# ------------------------------------------------------------
# These hyperparameters are starting points, not assumed
# optimal values. They will be evaluated using validation data.
#
# Batch size = 4:
#   Chosen conservatively for the RTX 3060 Laptop GPU (6 GB VRAM)
#   after earlier GPU memory issues. It can be increased later.
#
# Learning rate = 0.001:
#   A commonly used starting learning rate for Adam.
#
# Optimizer = Adam:
#   Provides adaptive parameter updates and is a practical
#   baseline for training a CNN from random initialization.
#
# Loss = CrossEntropyLoss:
#   Appropriate for single-label classification across
#   200 bird species using raw model logits.
#
# Image size = 224 x 224:
#   Matches the team's preprocessing pipeline and balances
#   spatial detail with computational requirements.
#
# Epochs = 1:
#   Used initially to verify the complete training pipeline.
#   Longer experiments will follow after successful testing.
#
# Model selection = Validation macro-F1:
#   Gives equal importance to each of the 200 bird species.
#   The test set is not used during training or model selection.
#
# Seed = 42:
#   Supports reproducibility and matches the dataset split seed.
# ------------------------------------------------------------


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    max_batches=None,
):
    """
    Train the model for one epoch or a limited number of batches.

    Args:
        max_batches:
            None -> train on the entire DataLoader.
            Integer -> process at most this many batches.

    Returns:
        Average training loss and classification accuracy.
    """

    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be a positive integer or None.")

    # Enable training behavior for BatchNorm and other layers.
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Show the actual number of batches planned for this run.
    total_batches = (
        len(train_loader)
        if max_batches is None
        else min(max_batches, len(train_loader))
    )

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        total=total_batches,
        leave=False,
    )

    for batch_idx, (images, labels) in enumerate(progress_bar):

        # Stop after the requested number of batches.
        # This is used only for short GPU smoke tests.
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Transfer the current batch to the selected device.
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous optimization step.
        optimizer.zero_grad(set_to_none=True)

        # Forward pass: obtain raw scores for all 200 classes.
        logits = model(images)

        # Calculate classification loss.
        loss = criterion(logits, labels)

        # Backpropagate gradients through the CNN.
        loss.backward()

        # Update trainable parameters using Adam.
        optimizer.step()

        # Accumulate sample-weighted loss.
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        # Count correct predictions.
        predictions = logits.argmax(dim=1)
        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        # Display progress and running training accuracy.
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=(
                f"{100 * correct_predictions / total_samples:.2f}%"
            ),
        )

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Evaluate the model without updating its parameters.

    Returns:
        Average validation loss, accuracy, and macro-F1.
    """

    # Disable training-specific behavior such as BatchNorm updates.
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    # Validation does not require gradient calculations.
    with torch.no_grad():
        for images, labels in tqdm(
            val_loader,
            desc="Validation",
            leave=False,
        ):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass only.
            logits = model(images)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

            # Collect predictions on CPU for metric calculation.
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    # Include all 200 classes, even if some are never predicted.
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(200)),
        average="macro",
        zero_division=0,
    )

    return epoch_loss, epoch_accuracy, macro_f1


def save_epoch_metrics(csv_path, epoch, train_loss, train_accuracy,
                       val_loss, val_accuracy, val_f1):
    """
    Append one epoch's results to a CSV file.

    This creates a reproducible record of model performance
    for plotting learning curves and comparing experiments.
    """

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the header only when the file does not exist
    # or is empty.
    write_header = (
        not csv_path.exists() or csv_path.stat().st_size == 0
    )

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
            ])

        writer.writerow([
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            val_f1,
        ])

def load_training_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Restore a previously saved training state.

    Restores:
        - Model weights
        - Adam optimizer state
        - Last completed epoch
        - Best validation macro-F1

    Returns:
        start_epoch: The next epoch to train (1-based).
        best_f1: Best validation macro-F1 so far.
    """

    checkpoint_path = Path(checkpoint_path)

    # Start from scratch if no previous checkpoint exists.
    if not checkpoint_path.exists():
        print("No previous checkpoint found. Starting fresh.")
        return 1, -1.0

    # Load a checkpoint created by our own training script.
    # weights_only=True restricts unpickling for safer loading.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Restore the CNN and classification head parameters.
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore Adam's internal moving averages and step counts.
    # Without this, resumed optimization would behave differently.
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_f1 = checkpoint["best_val_f1"]

    print(f"Resumed from epoch {checkpoint['epoch']}")
    print(f"Previous best validation macro-F1: {best_f1:.4f}")

    return start_epoch, best_f1




def main():
    """
    Train the scratch CNN on CUB-200-2011.

    Supports:
        - Multi-epoch training
        - Validation after each epoch
        - CSV experiment history
        - Best and latest model checkpoints
        - Resuming interrupted training

    All hyperparameters are initial experimental choices
    and can be adjusted based on validation results.
    """

    # --------------------------------------------------------
    # 1. Experiment configuration
    # --------------------------------------------------------

    seed = 42

    # Initial values selected based on GPU memory limitations
    # and standard CNN training practices.
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 20
    num_classes = 200
    image_size = 224

    # None means use the entire training dataset.
    # Set to 10 for a short GPU smoke test.
    max_train_batches = None

    # Enable automatic resumption if a latest checkpoint exists.
    resume_training = True

    # Use a separate directory to preserve our original
    # one-epoch baseline checkpoint.
    experiment_dir = Path("checkpoints/scratch_cnn/experiment_01")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = experiment_dir / "best_model.pt"
    last_checkpoint_path = experiment_dir / "last_model.pt"

    # CSV history is saved separately from model checkpoints.
    history_path = Path("outputs/training/scratch_cnn_experiment_01.csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 2. Reproducibility and device configuration
    # --------------------------------------------------------

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 55)
    print("SCRATCH CNN TRAINING â€” CUB-200-2011")
    print("=" * 55)

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Target epochs: {num_epochs}")
    print(f"Experiment directory: {experiment_dir}")

    if max_train_batches is not None:
        print(f"Smoke test: {max_train_batches} training batches")

    # --------------------------------------------------------
    # 3. Load dataset using the team's shared pipeline
    # --------------------------------------------------------

    train_loader, val_loader, _ = create_dataloaders(
        model_type="scratch_cnn",
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,
        seed=seed,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # --------------------------------------------------------
    # 4. Initialize the model, loss, and optimizer
    # --------------------------------------------------------

    encoder = ScratchCNNEncoder()

    model = BirdClassifier(
        encoder=encoder,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # --------------------------------------------------------
    # 5. Resume training if a checkpoint exists
    # --------------------------------------------------------

    start_epoch = 1
    best_f1 = -1.0

    if resume_training and last_checkpoint_path.exists():

        # Verify that the resumed experiment uses the same
        # configuration as the saved training run.
        saved_checkpoint = torch.load(
            last_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        expected_config = {
            "num_classes": num_classes,
            "image_size": image_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_train_batches": max_train_batches,
            "seed": seed,
        }

        for key, expected_value in expected_config.items():
            if saved_checkpoint.get(key) != expected_value:
                raise ValueError(
                    f"Configuration mismatch for '{key}': "
                    f"checkpoint={saved_checkpoint.get(key)}, "
                    f"current={expected_value}. "
                    "Use a new experiment directory if you "
                    "want to change the configuration."
                )

        # Resume the model and optimizer from the last
        # completed epoch, not the best validation epoch.
        start_epoch, best_f1 = load_training_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        # Avoid appending resumed metrics to an unrelated or
        # missing history file.
        if not history_path.exists():
            raise FileNotFoundError(
                "A training checkpoint exists, but its CSV history "
                f"is missing: {history_path}. Restore the matching "
                "history before resuming."
            )

        # Ensure CSV history matches the last saved epoch.
        # A crash may have occurred after saving the checkpoint
        # but before appending the final CSV row.
        with history_path.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            history_rows = list(csv.DictReader(file))

        recorded_epochs = [
            int(row["epoch"]) for row in history_rows
        ]

        last_saved_epoch = start_epoch - 1

        # The CSV must contain consecutive completed epochs.
        # If the final row is missing, we cannot reconstruct
        # its metrics from the current checkpoint.
        expected_epochs = list(
            range(1, last_saved_epoch + 1)
        )

        if recorded_epochs != expected_epochs:
            raise RuntimeError(
                "Checkpoint and CSV history are inconsistent. "
                f"Checkpoint epoch: {last_saved_epoch}, "
                f"CSV epochs: {recorded_epochs}. "
                "Recover the matching history or start a "
                "new experiment; do not append blindly."
            )

    elif last_checkpoint_path.exists():
        raise FileExistsError(
            f"Existing checkpoint found: {last_checkpoint_path}. "
            "Use a new experiment directory for a fresh run."
        )

    elif (
        best_checkpoint_path.exists()
        or history_path.exists()
    ):
        raise FileExistsError(
            "Existing experiment artifacts were found without "
            "a latest checkpoint. Use a new experiment name "
            "or recover the previous checkpoint."
        )

    else:
        print("Starting a new training experiment.")

    # --------------------------------------------------------
    # 6. Check whether training is already complete
    # --------------------------------------------------------

    if start_epoch > num_epochs:
        print(
            f"Training already completed through epoch "
            f"{start_epoch - 1}."
        )
        print("Increase num_epochs to continue training.")
        return

    # --------------------------------------------------------
    # 7. Multi-epoch training loop
    # --------------------------------------------------------

    for epoch in range(start_epoch, num_epochs + 1):

        print("\n" + "-" * 55)
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 55)

        # Train using the entire dataset unless a smoke-test
        # batch limit is explicitly configured.
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )

        # Evaluate using the validation set only.
        # The test set remains untouched during training.
        val_loss, val_accuracy, val_f1 = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # ----------------------------------------------------
        # 8. Display epoch results
        # ----------------------------------------------------

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2%}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2%} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        # ----------------------------------------------------
        # 9. Update the best validation score
        # ----------------------------------------------------

        is_best = val_f1 > best_f1

        if is_best:
            best_f1 = val_f1

        # ----------------------------------------------------
        # 10. Prepare the checkpoint
        # ----------------------------------------------------

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_f1,
            "num_classes": num_classes,
            "image_size": image_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_train_batches": max_train_batches,
            "seed": seed,
        }

        # ----------------------------------------------------
        # 11. Save the best model
        # ----------------------------------------------------

        if is_best:
            torch.save(checkpoint, best_checkpoint_path)

            print(
                f"New best model saved! "
                f"Macro-F1: {best_f1:.4f}"
            )

        # ----------------------------------------------------
        # 12. Record metrics for the technical report
        # ----------------------------------------------------

        # Save metrics before marking the epoch as completed.
        save_epoch_metrics(
            csv_path=history_path,
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            val_f1=val_f1,
        )

        print(f"Epoch metrics recorded: {history_path}")

        # ----------------------------------------------------
        # 13. Save the latest training state
        # ----------------------------------------------------

        # The latest checkpoint marks the completed epoch.
        # Resume checks will detect any CSV/checkpoint mismatch.
        torch.save(checkpoint, last_checkpoint_path)

        print(f"Latest checkpoint saved: {last_checkpoint_path}")



    # --------------------------------------------------------
    # 14. Final experiment summary
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("TRAINING COMPLETED")
    print("=" * 55)

    print(f"Completed epochs: {num_epochs}")
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(f"Best model: {best_checkpoint_path}")
    print(f"Latest model: {last_checkpoint_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
