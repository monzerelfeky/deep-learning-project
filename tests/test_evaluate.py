import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.evaluate import evaluate_model


class TinyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        x = torch.tensor([float(index), 1.0])
        label = int(index % 2)
        return x, label, f"bird_{index}.jpg"


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    def forward(self, x):
        return self.linear(x)


def test_evaluate_returns_predictions_probabilities_and_filenames():
    model = TinyClassifier()
    model.train()
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)

    result = evaluate_model(model, loader, num_classes=2)

    assert result["num_samples"] == 5
    assert result["y_true"].shape == (5,)
    assert result["y_pred"].shape == (5,)
    assert result["confidence"].shape == (5,)
    assert result["probabilities"].shape == (5, 2)
    assert result["sample_ids"] == [f"bird_{i}.jpg" for i in range(5)]
    assert result["inference_time_seconds"] >= 0
    np.testing.assert_allclose(result["probabilities"].sum(axis=1), 1.0)
    assert model.training  # original mode is restored


def test_empty_loader_raises():
    loader = DataLoader([], batch_size=2)
    with pytest.raises(ValueError, match="no examples"):
        evaluate_model(TinyClassifier(), loader)


def test_invalid_model_output_raises():
    class BadModel(nn.Module):
        def forward(self, x):
            return x.squeeze(0)

    loader = DataLoader([(torch.ones(3), 0)], batch_size=1)
    with pytest.raises(ValueError, match="logits"):
        evaluate_model(BadModel(), loader)
