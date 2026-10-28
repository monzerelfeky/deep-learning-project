
from pathlib import Path

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


def main():
    """
    Train the scratch CNN on CUB-200-2011.

    The initial run is a GPU smoke test using only 10 training
    batches. Full training begins after pipeline verification.
    """

    # --------------------------------------------------------
    # Initial baseline hyperparameters
    # --------------------------------------------------------
    seed = 42
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 1
    num_classes = 200
    image_size = 224

    # None means train on all batches.
    # This is our first full-epoch baseline experiment.
    max_train_batches = None

    # Improve reproducibility across experiments.
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    if max_train_batches is not None:
        print(
            f"Smoke test enabled: "
            f"{max_train_batches} training batches per epoch"
        )

    # --------------------------------------------------------
    # Load the team's shared preprocessing pipeline
    # --------------------------------------------------------
    train_loader, val_loader, _ = create_dataloaders(
        model_type="scratch_cnn",
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,
        seed=seed,
    )

    # --------------------------------------------------------
    # Initialize the CNN from random weights
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
    # Prepare checkpoint directory
    # --------------------------------------------------------
    checkpoint_dir = Path("checkpoints/scratch_cnn")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0

    # --------------------------------------------------------
    # Training and validation loop
    # --------------------------------------------------------
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )

        # Validate on the complete validation split.
        val_loss, val_accuracy, val_f1 = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

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
        # Save the best checkpoint based on validation macro-F1
        # ----------------------------------------------------
        if val_f1 > best_f1:
            best_f1 = val_f1

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_f1,
                "num_classes": num_classes,
                "image_size": image_size,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "max_train_batches": max_train_batches,
            }

            # Keep smoke-test checkpoints separate from full runs.
            checkpoint_name = (
                "smoke_test.pt"
                if max_train_batches is not None
                else "best_model.pt"
            )

            checkpoint_path = checkpoint_dir / checkpoint_name

            torch.save(checkpoint, checkpoint_path)

            print(f"Best checkpoint saved: {checkpoint_path}")

    print("\nTraining completed.")
    print(f"Best validation macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
