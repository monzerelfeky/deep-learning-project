import torch
from torch import nn

from src.models.classifier import BirdClassifier
from src.models.transfer import ResNet18Encoder


def test_resnet18_encoder_contract_without_downloading_weights():
    """Check feature, pooling, classifier, and Grad-CAM layer contracts offline."""
    encoder = ResNet18Encoder(pretrained=False)
    encoder.eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feature_maps = encoder.forward_features(images)
        pooled = encoder(images)
        logits = BirdClassifier(encoder, num_classes=200)(images)

    assert feature_maps.shape == (2, 512, 2, 2)
    assert pooled.shape == (2, 512)
    assert logits.shape == (2, 200)
    assert isinstance(encoder.gradcam_layer, nn.Conv2d)


def test_frozen_backbone_leaves_classifier_trainable_and_batchnorm_in_eval():
    encoder = ResNet18Encoder(pretrained=False, freeze_backbone=True)
    model = BirdClassifier(encoder=encoder, num_classes=200)
    model.train()

    assert encoder.is_frozen
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert not encoder.backbone.training

    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    before = model.classifier.weight.detach().clone()
    loss = nn.CrossEntropyLoss()(model(images), labels)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item()
    assert not torch.equal(before, model.classifier.weight.detach())
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_finetuning_allows_backbone_gradients():
    encoder = ResNet18Encoder(pretrained=False, freeze_backbone=False)
    model = BirdClassifier(encoder=encoder, num_classes=200)
    model.train()

    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(model(images), labels)
    loss.backward()

    first_conv = encoder.backbone[0]
    assert first_conv.weight.requires_grad
    assert first_conv.weight.grad is not None
    assert torch.isfinite(first_conv.weight.grad).all().item()
