"""Model inference and evaluation for the CUB-200 classification task."""

from time import perf_counter

import torch

from src.evaluation.metrics import compute_metrics


def _get_model_device(model):
    """Infer the device from the model's parameters or buffers."""
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    return torch.device("cpu")


def evaluate_model(model, data_loader, device=None, num_classes=None):
    """Evaluate a classifier over a DataLoader.

    Batches may be ``(images, labels)`` or
    ``(images, labels, file_names, ...)``. If file names are present,
    they are retained so benchmark runs can prove they used the same examples.

    ``inference_time_seconds`` measures model inference and excludes data loading.
    The original device and training/evaluation mode are restored afterwards.
    """
    if model is None:
        raise ValueError("model must not be None.")
    if data_loader is None:
        raise ValueError("data_loader must not be None.")

    original_device = _get_model_device(model)
    target_device = torch.device(device) if device is not None else original_device
    was_training = model.training
    model.to(target_device)
    model.eval()

    all_true, all_pred, all_confidence = [], [], []
    all_probabilities, all_file_names = [], []
    saw_file_names = False
    inference_seconds = 0.0

    try:
        with torch.inference_mode():
            for batch in data_loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError(
                        "Each batch must contain at least images and labels."
                    )

                images, labels = batch[0], batch[1]
                images = images.to(target_device)
                labels = torch.as_tensor(labels, device=target_device).long()

                if len(batch) >= 3:
                    names = batch[2]
                    if (
                        isinstance(names, (tuple, list))
                        and all(isinstance(name, str) for name in names)
                    ):
                        all_file_names.extend(names)
                        saw_file_names = True
                    else:
                        raise ValueError(
                            "The third batch item must be a sequence of string file names."
                        )
                elif saw_file_names:
                    raise ValueError("File names are missing from one or more batches.")

                if target_device.type == "cuda":
                    torch.cuda.synchronize(target_device)
                started = perf_counter()
                logits = model(images)
                if target_device.type == "cuda":
                    torch.cuda.synchronize(target_device)
                inference_seconds += perf_counter() - started

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
        model.to(original_device)
        model.train(was_training)

    if not all_true:
        raise ValueError("The DataLoader produced no examples.")
    if saw_file_names and len(all_file_names) != len(all_true):
        raise ValueError("The number of file names does not match the examples.")

    class_count = num_classes or all_probabilities[0].shape[1]
    y_true = torch.tensor(all_true, dtype=torch.long).numpy()
    y_pred = torch.tensor(all_pred, dtype=torch.long).numpy()
    confidence_scores = torch.tensor(
        all_confidence, dtype=torch.float32
    ).numpy()
    probabilities = torch.cat(all_probabilities, dim=0).numpy()

    result = {
        "metrics": compute_metrics(y_true, y_pred, num_classes=class_count),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidence_scores,
        "probabilities": probabilities,
        "num_samples": len(y_true),
        "num_classes": class_count,
        "inference_time_seconds": float(inference_seconds),
    }
    if saw_file_names:
        result["sample_ids"] = all_file_names
    return result


evaluate = evaluate_model
