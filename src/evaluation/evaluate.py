
"""Model inference and evaluation."""

import torch

from src.evaluation.metrics import compute_metrics


def _get_model_device(model):
    """Infer the device from the model's parameters or buffers."""
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    return torch.device("cpu")


def evaluate_model(model, data_loader, device=None, num_classes=None):
    """Evaluate a classifier over a DataLoader.

    Expected batch format:
        (images, labels)
    or:
        (images, labels, extra_information)

    Extra batch elements, such as captions, are ignored.

    Returns a dictionary containing metrics, true labels, predictions,
    confidence scores for predicted classes, and class probabilities.
    """
    if model is None:
        raise ValueError("model must not be None.")
    if data_loader is None:
        raise ValueError("data_loader must not be None.")

    device = torch.device(device) if device is not None else _get_model_device(model)
    model = model.to(device)
    was_training = model.training
    model.eval()

    all_true = []
    all_pred = []
    all_confidence = []
    all_probabilities = []

    try:
        with torch.inference_mode():
            for batch in data_loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError(
                        "Each batch must contain at least images and labels."
                    )

                images, labels = batch[0], batch[1]
                images = images.to(device)
                labels = torch.as_tensor(labels, device=device).long()

                logits = model(images)

                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                elif isinstance(logits, dict):
                    if "logits" not in logits:
                        raise ValueError(
                            "Model output dictionary must contain 'logits'."
                        )
                    logits = logits["logits"]

                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise ValueError(
                        "Model must return logits with shape [batch, classes]."
                    )

                if logits.shape[0] != labels.shape[0]:
                    raise ValueError(
                        "Model output batch size does not match label batch size."
                    )

                if num_classes is not None and logits.shape[1] != num_classes:
                    raise ValueError(
                        f"Expected {num_classes} classes, got {logits.shape[1]}."
                    )

                probabilities = torch.softmax(logits, dim=1)
                confidence, predictions = probabilities.max(dim=1)

                all_true.extend(labels.cpu().tolist())
                all_pred.extend(predictions.cpu().tolist())
                all_confidence.extend(confidence.cpu().tolist())
                all_probabilities.append(probabilities.cpu())

    finally:
        model.train(was_training)

    if not all_true:
        raise ValueError("The DataLoader produced no examples.")

    class_count = num_classes or all_probabilities[0].shape[1]

    y_true = torch.tensor(all_true, dtype=torch.long).numpy()
    y_pred = torch.tensor(all_pred, dtype=torch.long).numpy()
    confidence_scores = torch.tensor(
        all_confidence, dtype=torch.float32
    ).numpy()
    probabilities = torch.cat(all_probabilities, dim=0).numpy()

    metrics = compute_metrics(y_true, y_pred, num_classes=class_count)

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidence_scores,
        "probabilities": probabilities,
        "num_samples": len(y_true),
        "num_classes": class_count,
    }


# Short alias for callers that prefer evaluate(...).
evaluate = evaluate_model
