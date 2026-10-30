
import torch
from torch import device, nn

from src.models.scratch_cnn import ScratchCNNEncoder
from src.models.classifier import BirdClassifier


def test_scratch_cnn_training_step():
    """Verify that the CNN supports forward and backward passes."""

    device = torch.device("cpu")

    model = BirdClassifier(
        encoder=ScratchCNNEncoder(),
        num_classes=200,
    ).to(device)

    model.train()

    # Synthetic images and valid class labels.
    images = torch.randn(1, 3, 224, 224, device=device)
    labels = torch.randint(0, 200, (1,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Save weights before training to verify an update.
    before = model.classifier.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    assert logits.shape == (1, 200)

    loss = criterion(logits, labels)
    assert torch.isfinite(loss).item()

    loss.backward()

    # Confirm gradients reached the convolutional encoder.
    first_conv = model.encoder.features[0][0]
    assert first_conv.weight.grad is not None
    assert torch.isfinite(first_conv.weight.grad).all().item()

    optimizer.step()

    after = model.classifier.weight.detach()
    assert not torch.equal(before, after)
