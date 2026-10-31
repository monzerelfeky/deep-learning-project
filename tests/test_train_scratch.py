
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import csv
from pathlib import Path

from src.training.train_scratch import (
    save_epoch_metrics,
    load_training_checkpoint,
)

from src.models.classifier import BirdClassifier
from src.models.scratch_cnn import ScratchCNNEncoder
from src.training.train_scratch import train_one_epoch
from src.training.train_scratch import validate_one_epoch


def test_train_one_epoch():
    """Check that the training loop runs and returns valid metrics."""

    torch.manual_seed(42)

    # Use a tiny synthetic dataset for a fast, offline unit test.
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=1,
    )

    model = BirdClassifier(ScratchCNNEncoder(), num_classes=200)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss, accuracy = train_one_epoch(
        model=model,
        train_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(torch.tensor(loss))
    assert loss >= 0
    assert 0.0 <= accuracy <= 1.0
def test_validate_one_epoch():
    """Verify validation metrics and unchanged model parameters."""

    torch.manual_seed(42)

    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=1,
    )

    model = BirdClassifier(ScratchCNNEncoder(), num_classes=200)
    criterion = nn.CrossEntropyLoss()

    # Save parameters to verify validation performs no updates.
    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    loss, accuracy, macro_f1 = validate_one_epoch(
        model=model,
        val_loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(torch.tensor(loss))
    assert loss >= 0
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= macro_f1 <= 1.0

    # Validation must not modify any trainable parameters.
    for old, new in zip(before, model.parameters()):
        assert torch.equal(old, new)

def test_save_epoch_metrics(tmp_path):
    """Verify CSV headers and epoch metrics are saved correctly."""

    csv_path = tmp_path / "training_history.csv"

    save_epoch_metrics(
        csv_path=csv_path,
        epoch=1,
        train_loss=5.2,
        train_accuracy=0.01,
        val_loss=5.1,
        val_accuracy=0.02,
        val_f1=0.005,
    )

    save_epoch_metrics(
        csv_path=csv_path,
        epoch=2,
        train_loss=4.9,
        train_accuracy=0.03,
        val_loss=4.8,
        val_accuracy=0.04,
        val_f1=0.015,
    )

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["epoch"] == "1"
    assert rows[1]["epoch"] == "2"
    assert float(rows[1]["val_macro_f1"]) == 0.015


def test_load_training_checkpoint(tmp_path):
    """Verify model weights and optimizer state are restored."""

    torch.manual_seed(42)

    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Perform one optimization step to populate Adam's state.
    inputs = torch.randn(3, 4)
    targets = torch.tensor([0, 1, 0])

    loss = nn.CrossEntropyLoss()(model(inputs), targets)
    loss.backward()
    optimizer.step()

    saved_weights = {
        name: parameter.detach().clone()
        for name, parameter in model.state_dict().items()
    }

    checkpoint_path = tmp_path / "last_model.pt"

    torch.save(
        {
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": 0.12,
        },
        checkpoint_path,
    )

    # Create fresh objects to simulate restarting Python.
    restored_model = nn.Linear(4, 2)
    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(), lr=0.001
    )

    start_epoch, best_f1 = load_training_checkpoint(
        checkpoint_path=checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
    )

    assert start_epoch == 4
    assert best_f1 == 0.12

    # Confirm all saved model parameters were restored.
    for name, parameter in restored_model.state_dict().items():
        assert torch.equal(parameter, saved_weights[name])

    # Confirm Adam's internal state was also restored.
    assert len(restored_optimizer.state) > 0

    for original_param, restored_param in zip(
        model.parameters(), restored_model.parameters()
    ):
        original_state = optimizer.state[original_param]
        restored_state = restored_optimizer.state[restored_param]

        for key in original_state:
            original_value = original_state[key]
            restored_value = restored_state[key]

            if torch.is_tensor(original_value):
                assert torch.equal(original_value, restored_value)
            else:
                assert original_value == restored_value


def test_missing_checkpoint_starts_fresh(tmp_path):
    """Verify that missing checkpoints do not crash training."""

    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_epoch, best_f1 = load_training_checkpoint(
        checkpoint_path=tmp_path / "missing.pt",
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert start_epoch == 1
    assert best_f1 == -1.0
