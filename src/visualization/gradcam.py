
"""Generic Grad-CAM for convolutional image classifiers."""

import torch
import torch.nn.functional as F


class GradCAM:
    """Compute Grad-CAM maps for a chosen convolutional layer."""

    def __init__(self, model, target_layer):
        if target_layer is None:
            raise ValueError("target_layer must be a model layer.")

        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )
        self._backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, images, target_classes=None):
        """Generate normalized CAMs with shape [B, H, W].

        Args:
            images: Input tensor [B, C, H, W].
            target_classes: Optional class index per image. If omitted,
                the model's predicted class is used.

        Returns:
            A dictionary with heatmaps, target classes, logits and predictions.
        """
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W].")

        device = next(
            iter(list(self.model.parameters()) + list(self.model.buffers())),
            torch.empty(0),
        ).device
        images = images.to(device)

        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        try:
            # Gradients must remain enabled for Grad-CAM.
            with torch.enable_grad():
                logits = self.model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                elif isinstance(logits, dict):
                    logits = logits["logits"]

                if logits.ndim != 2:
                    raise ValueError("Model must return logits [B, classes].")

                predictions = logits.argmax(dim=1)

                if target_classes is None:
                    targets = predictions
                else:
                    targets = torch.as_tensor(
                        target_classes, device=device, dtype=torch.long
                    ).reshape(-1)
                    if targets.numel() != images.shape[0]:
                        raise ValueError(
                            "target_classes must contain one class per image."
                        )
                    if torch.any(targets < 0) or torch.any(targets >= logits.shape[1]):
                        raise ValueError("A target class is outside the logits range.")

                if self.activations is None:
                    raise RuntimeError(
                        "No activations captured. Check that target_layer is "
                        "executed during the model forward pass."
                    )

                self.model.zero_grad(set_to_none=True)
                selected_scores = logits.gather(1, targets[:, None]).sum()
                selected_scores.backward()

                if self.gradients is None:
                    raise RuntimeError(
                        "No gradients captured. Choose a layer connected to "
                        "the model's output."
                    )

                activations = self.activations
                gradients = self.gradients

                if activations.ndim != 4 or gradients.ndim != 4:
                    raise ValueError(
                        "Grad-CAM target layer must output [B, C, H, W] features."
                    )

                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cams = torch.relu(
                    (weights * activations).sum(dim=1, keepdim=True)
                )
                cams = F.interpolate(
                    cams,
                    size=images.shape[-2:],
                    mode="bilinear",
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
        """Remove registered hooks."""
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Create an RGB heatmap overlay as a NumPy array.

    Args:
        image: HWC RGB image, normally scaled to [0, 1] or [0, 255].
        heatmap: 2D Grad-CAM map.
        alpha: Heatmap blend strength between 0 and 1.

    Returns:
        RGB uint8 NumPy array.
    """
    import numpy as np
    import matplotlib.cm as cm

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
    heat = heat_tensor[0, 0].numpy()
    heat = np.clip(heat, 0, 1)

    colored = cm.get_cmap("jet")(heat)[..., :3] * 255.0
    overlay = (1 - alpha) * image_float + alpha * colored
    return np.clip(overlay, 0, 255).astype(np.uint8)
