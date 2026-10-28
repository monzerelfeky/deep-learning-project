
import torch
from torch import nn

from .encoder import BaseEncoder


class BirdClassifier(nn.Module):
    """
    Image classifier built on top of a reusable encoder.

    The encoder extracts visual features, while the
    classification head predicts one of 200 bird classes.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int = 200,
    ):
        super().__init__()

        self.encoder = encoder
        self.classifier = nn.Linear(
            in_features=encoder.out_dim,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, 3, 224, 224]
        Output: [B, 200] unnormalized class logits
        """
        features = self.encoder(x)
        logits = self.classifier(features)

        return logits
