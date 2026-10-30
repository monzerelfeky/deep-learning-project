"""Generic Grad-CAM for convolutional image classifiers."""

import torch
import torch.nn.functional as F


class GradCAM:
    """Compute Grad-CAM maps for a selected convolutional layer."""

    def __init__(self, model, target_layer):
        if model is None:
            raise ValueError("model must not be None.")
        if target_layer is None:
            raise ValueError("target_layer must be a model layer.")
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output
        self.gradients = None
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(self._save_output_gradient)

    def _save_output_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, images, target_classes=None):
        """Return normalized heatmaps [B,H,W], classes, predictions and logits."""
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must be a tensor with shape [B, C, H, W].")

        tensors = list(self.model.parameters()) + list(self.model.buffers())
        original_device = tensors[0].device if tensors else torch.device("cpu")
        images = images.to(original_device)
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        try:
            with torch.enable_grad():
                logits = self.model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                elif isinstance(logits, dict):
                    if "logits" not in logits:
                        raise ValueError("Model output dictionary must contain 'logits'.")
                    logits = logits["logits"]
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise ValueError("Model must return logits [B, classes].")

                predictions = logits.argmax(dim=1)
                if target_classes is None:
                    targets = predictions
                else:
                    targets = torch.as_tensor(
                        target_classes, device=original_device, dtype=torch.long
                    ).reshape(-1)
                    if targets.numel() != images.shape[0]:
                        raise ValueError(
                            "target_classes must contain one class per image."
                        )
                    if torch.any(targets < 0) or torch.any(targets >= logits.shape[1]):
                        raise ValueError("A target class is outside the logits range.")

                if self.activations is None:
                    raise RuntimeError(
                        "No activations captured. Check that target_layer runs "
                        "during the model forward pass."
                    )
                if self.activations.ndim != 4:
                    raise ValueError(
                        "Grad-CAM target layer must output [B, C, H, W] features."
                    )

                scores = logits.gather(1, targets[:, None]).sum()
                scores.backward()
                if self.gradients is None:
                    raise RuntimeError(
                        "No gradients captured. Choose a target layer connected "
                        "to the model output."
                    )
                if self.gradients.ndim != 4:
                    raise ValueError("Target-layer gradients must be four-dimensional.")

                weights = self.gradients.mean(dim=(2, 3), keepdim=True)
                cams = torch.relu(
                    (weights * self.activations).sum(dim=1, keepdim=True)
                )
                cams = F.interpolate(
                    cams, size=images.shape[-2:], mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                flat = cams.flatten(start_dim=1)
                mins = flat.min(dim=1).values[:, None, None]
                maxs = flat.max(dim=1).values[:, None, None]
                cams = (cams - mins) / (maxs - mins).clamp_min(1e-8)

                return {
                    "heatmaps": cams.detach().cpu(),
                    "target_classes": targets.detach().cpu(),
                    "predictions": predictions.detach().cpu(),
                    "logits": logits.detach().cpu(),
                }
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def close(self):
        """Remove the registered forward hook."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Blend a 2D heatmap over an HWC RGB image and return uint8 RGB."""
    import numpy as np
    import matplotlib

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    image = np.asarray(image)
    heatmap = np.asarray(heatmap)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape [H, W, 3].")
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be two-dimensional.")

    image_float = image.astype(np.float32)
    if image_float.max(initial=0) <= 1:
        image_float *= 255.0
    image_float = np.clip(image_float, 0, 255)
    heat_tensor = torch.as_tensor(heatmap, dtype=torch.float32)[None, None]
    heat_tensor = F.interpolate(
        heat_tensor, size=image.shape[:2], mode="bilinear", align_corners=False
    )
    heat = np.clip(heat_tensor[0, 0].numpy(), 0, 1)
    colored = matplotlib.colormaps["jet"](heat)[..., :3] * 255.0
    return np.clip((1 - alpha) * image_float + alpha * colored, 0, 255).astype(
        np.uint8
    )
