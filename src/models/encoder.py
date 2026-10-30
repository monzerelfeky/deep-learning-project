
from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    """
    Shared interface for image feature extractors.

    Encoders provide spatial features for downstream tasks
    and pooled feature vectors for image classification.
    """

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Number of channels in the encoder's output."""
        raise NotImplementedError

    @property
    @abstractmethod
    def gradcam_layer(self) -> nn.Module:
        """Target convolutional layer for Grad-CAM."""
        raise NotImplementedError

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps.

        Input:  [B, 3, 224, 224]
        Output: [B, C, H, W]
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a feature vector using global average pooling.

        Input:  [B, 3, 224, 224]
        Output: [B, out_dim]
        """
        features = self.forward_features(x)
        return features.mean(dim=(2, 3))
