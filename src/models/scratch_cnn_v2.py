
import torch
from torch import nn

from .encoder import BaseEncoder


class ResidualBlock(nn.Module):
    """
    Two-convolution residual block.

    A projection shortcut is used when the number of channels
    changes or the spatial resolution is reduced.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        return self.relu(out)


class ScratchCNNEncoderV2(BaseEncoder):
    """
    Residual CNN trained entirely from scratch.

    Input:            [B, 3, 224, 224]
    Spatial features: [B, 256, 14, 14]
    Pooled features:  [B, 256]
    """

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3,
                stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.stage1 = ResidualBlock(32, 32, stride=2)
        self.stage2 = ResidualBlock(32, 64, stride=2)
        self.stage3 = ResidualBlock(64, 128, stride=2)
        self.stage4 = ResidualBlock(128, 256, stride=2)

    @property
    def out_dim(self) -> int:
        return 256

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.stage4.conv2

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
