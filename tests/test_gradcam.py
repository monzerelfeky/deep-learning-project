import numpy as np
import pytest
import torch
from torch import nn

from src.visualization.gradcam import GradCAM, overlay_heatmap


class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_gradcam_generates_normalized_maps_and_closes_hook():
    model = ToyCNN()
    model.train()
    cam = GradCAM(model, model.conv)
    result = cam.generate(torch.rand(2, 3, 32, 32), target_classes=[0, 2])

    assert result["heatmaps"].shape == (2, 32, 32)
    assert result["target_classes"].tolist() == [0, 2]
    assert torch.isfinite(result["heatmaps"]).all()
    assert result["heatmaps"].min() >= 0
    assert result["heatmaps"].max() <= 1
    assert model.training
    cam.close()
    assert cam._forward_handle is None


def test_gradcam_rejects_invalid_target_classes():
    model = ToyCNN()
    with GradCAM(model, model.conv) as cam:
        with pytest.raises(ValueError, match="outside"):
            cam.generate(torch.rand(1, 3, 16, 16), target_classes=[9])


def test_overlay_heatmap_returns_rgb_uint8():
    image = np.ones((16, 16, 3), dtype=np.float32)
    heatmap = np.zeros((8, 8), dtype=np.float32)
    result = overlay_heatmap(image, heatmap)
    assert result.shape == (16, 16, 3)
    assert result.dtype == np.uint8
