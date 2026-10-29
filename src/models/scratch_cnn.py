
import torch
from torch import nn

from .encoder import BaseEncoder


class ScratchCNNEncoder(BaseEncoder):
    """
    Four-block CNN encoder trained entirely from scratch.

    Input:  [B, 3, 224, 224]
    Output: [B, 256] after global average pooling
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Extract features and halve spatial dimensions."""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @property
    def out_dim(self) -> int:
        """Dimension of the pooled feature vector."""
        return 256

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last convolutional layer for Grad-CAM visualization."""
        return self.features[3][0]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial features without global average pooling."""
        return self.features(x)
