"""ImageNet-pretrained ResNet18 encoder for transfer learning on CUB-200-2011."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .encoder import BaseEncoder


class ResNet18Encoder(BaseEncoder):
    """ResNet18 feature extractor compatible with  ``BaseEncoder``.

    Args:
        pretrained: Load torchvision's default ImageNet-1K pretrained weights.
            This may download weights on the first run. Set False only for
            architecture/smoke tests; random initialization is not transfer learning.
        freeze_backbone: If True, freeze all ResNet parameters and keep BatchNorm
            layers in evaluation mode while training the external classifier head.

    At a 224x224 input resolution, ``forward_features`` returns
    ``[B, 512, 7, 7]`` and the base class ``forward`` returns ``[B, 512]``.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        network = resnet18(weights=weights)

        # Keep only the convolutional backbone. The 1000-class ImageNet head
        # is intentionally discarded; BirdClassifier adds a 200-class head.
        self.backbone = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        self._frozen = False
        self.set_frozen(freeze_backbone)

    @property
    def out_dim(self) -> int:
        """Number of channels in the final ResNet18 feature map."""
        return 512

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last residual block's second convolution, suitable for Grad-CAM."""
        return self.backbone[-1][-1].conv2

    @property
    def is_frozen(self) -> bool:
        """Whether the backbone is currently frozen."""
        return self._frozen

    def set_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze backbone parameters and configure BatchNorm behavior."""
        self._frozen = bool(frozen)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not self._frozen)
        if self._frozen:
            self.backbone.eval()

    def train(self, mode: bool = True):
        """Keep BatchNorm statistics fixed when the backbone is frozen.

        ``BirdClassifier.train()`` recursively calls ``train()`` on its encoder.
        Without this override, BatchNorm running statistics would still change
        in the frozen-feature experiment even though gradients are disabled.
        """
        super().train(mode)
        if self._frozen:
            self.backbone.eval()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature maps without global average pooling."""
        return self.backbone(x)
