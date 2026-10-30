
"""Plots for classification evaluation."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _finish_figure(fig, save_path=None, show=False):
    """Save and/or display a figure, then return it."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_confusion_matrix(
    matrix,
    class_names=None,
    normalize=False,
    save_path=None,
    show=False,
    max_classes=40,
):
    """Plot a raw or normalized confusion matrix.

    For readability, matrices with more than max_classes classes are
    displayed using the first max_classes labels. Pass a smaller subset
    yourself if you need a specific class selection.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    if max_classes < 1:
        raise ValueError("max_classes must be positive.")

    count = min(matrix.shape[0], max_classes)
    shown = matrix[:count, :count]

    if normalize:
        row_sums = shown.sum(axis=1, keepdims=True)
        shown = np.divide(
            shown,
            row_sums,
            out=np.zeros_like(shown, dtype=float),
            where=row_sums != 0,
        )

    names = (
        list(class_names[:count])
        if class_names is not None
        else [str(i) for i in range(count)]
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(shown, interpolation="nearest", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="Normalized Confusion Matrix" if normalize
        else "Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
    )
    ax.set_xticks(np.arange(count))
    ax.set_yticks(np.arange(count))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    fig.tight_layout()
    return _finish_figure(fig, save_path, show)


def plot_class_f1(
    per_class_f1,
    class_names=None,
    top_n=10,
    save_path=None,
    show=False,
):
    """Plot the classes with the highest and lowest F1 scores."""
    scores = np.asarray(per_class_f1, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("per_class_f1 must be a non-empty 1D array.")
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    names = (
        list(class_names)
        if class_names is not None
        else [str(i) for i in range(scores.size)]
    )
    if len(names) != len(scores):
        raise ValueError("class_names length must match per_class_f1.")

    n = min(top_n, len(scores))
    best = np.argsort(scores)[-n:][::-1]
    worst = np.argsort(scores)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n * 0.35)))

    for ax, indices, title in (
        (axes[0], best, "Highest Class F1"),
        (axes[1], worst, "Lowest Class F1"),
    ):
        ax.barh([names[i] for i in indices][::-1], scores[indices][::-1])
        ax.set_title(title)
        ax.set_xlabel("F1 score")
        ax.set_xlim(0, 1)

    fig.tight_layout()
    return _finish_figure(fig, save_path, show)


def plot_prediction_examples(
    images,
    y_true,
    y_pred,
    confidence,
    class_names=None,
    correct=True,
    max_examples=12,
    save_path=None,
    show=False,
):
    """Plot correct or incorrect predictions with confidence scores.

    images may be a list/array of HWC images or a tensor of NCHW images.
    Images are expected to be displayable as RGB values. Normalized tensors
    should be denormalized before plotting.
    """
    import torch

    if max_examples < 1:
        raise ValueError("max_examples must be positive.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence)

    if not (len(images) == len(y_true) == len(y_pred) == len(confidence)):
        raise ValueError("images and prediction arrays must have equal lengths.")

    if torch.is_tensor(images):
        image_array = images.detach().cpu()
        if image_array.ndim != 4:
            raise ValueError("Tensor images must have shape [N, C, H, W].")
        image_array = image_array.permute(0, 2, 3, 1).numpy()
    else:
        image_array = np.asarray(images)

    selected = np.flatnonzero(y_true == y_pred if correct else y_true != y_pred)
    selected = selected[:max_examples]

    if selected.size == 0:
        raise ValueError("No examples match the requested correct/incorrect filter.")

    names = class_names

    columns = min(4, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, index in zip(axes, selected):
        image = image_array[index]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        image = np.clip(image, 0, 1) if image.dtype.kind == "f" else image

        ax.imshow(image.squeeze(), cmap="gray" if image.ndim == 2 else None)

        true_name = names[y_true[index]] if names is not None else str(y_true[index])
        pred_name = names[y_pred[index]] if names is not None else str(y_pred[index])
        ax.set_title(
            f"True: {true_name}\nPred: {pred_name}\n"
            f"Confidence: {confidence[index]:.1%}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle("Correct Predictions" if correct else "Incorrect Predictions")
    fig.tight_layout()
    return _finish_figure(fig, save_path, show)
