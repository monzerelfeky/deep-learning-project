
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
