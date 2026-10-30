import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASE = Path("outputs/evaluation")
FIGURES = BASE / "figures"
DETAILS = BASE / "test_details.json"

FIGURES.mkdir(parents=True, exist_ok=True)

if not DETAILS.exists():
    raise FileNotFoundError(f"Missing evaluation results: {DETAILS}")

with DETAILS.open("r", encoding="utf-8") as f:
    results = json.load(f)

if not isinstance(results, list) or not results:
    raise ValueError("test_details.json must contain a non-empty list.")

print(f"Loaded {len(results)} model evaluation results.")


def save_confusion_matrix(matrix, filename, title, normalize=False):
    matrix = np.asarray(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto",
        cmap="Blues"
    )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ticks = np.arange(matrix.shape[0])
    ax.set_xticks(ticks[::10])
    ax.set_yticks(ticks[::10])
    ax.set_xticklabels(ticks[::10])
    ax.set_yticklabels(ticks[::10])

    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


# 1. Model comparison
names = []
accuracies = []
macro_f1s = []

for item in results:
    name = item["name"]
    metrics = item["metrics"]

    names.append(name.replace("_", " ").title())
    accuracies.append(float(metrics["accuracy"]) * 100)
    macro_f1s.append(float(metrics["macro_f1"]) * 100)

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width / 2, accuracies, width, label="Accuracy")
ax.bar(x + width / 2, macro_f1s, width, label="Macro-F1")

ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison - CUB-200 Test Set")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
ax.grid(axis="y", alpha=0.25)

for bars in ax.containers:
    ax.bar_label(bars, fmt="%.2f%%", padding=3)

fig.tight_layout()
fig.savefig(FIGURES / "model_comparison.png", dpi=200)
plt.close(fig)


# 2. Confusion matrices and per-class F1
for item in results:
    name = item["name"]
    metrics = item["metrics"]

    suffix = name.lower()
    cm = np.asarray(metrics["confusion_matrix"])
    normalized_cm = np.asarray(metrics["normalized_confusion_matrix"])
    per_class_f1 = np.asarray(metrics["per_class_f1"], dtype=float)

    save_confusion_matrix(
        cm,
        f"confusion_matrix_{suffix}.png",
        f"Confusion Matrix - {name.replace('_', ' ').title()}"
    )

    save_confusion_matrix(
        normalized_cm,
        f"normalized_confusion_matrix_{suffix}.png",
        f"Normalized Confusion Matrix - {name.replace('_', ' ').title()}"
    )

    # Show the 10 highest and 10 lowest F1 classes.
    n = min(10, len(per_class_f1))
    best = np.argsort(per_class_f1)[-n:][::-1]
    worst = np.argsort(per_class_f1)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, indices, title in (
        (axes[0], best, "Highest Class F1"),
        (axes[1], worst, "Lowest Class F1"),
    ):
        labels = [f"Class {i}" for i in indices][::-1]
        scores = per_class_f1[indices][::-1] * 100

        ax.barh(labels, scores)
        ax.set_title(title)
        ax.set_xlabel("F1 Score (%)")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"Per-Class F1 Analysis - {name.replace('_', ' ').title()}"
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"per_class_f1_{suffix}.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)


print("\nVisualization generation completed.")
print(f"Figures saved in: {FIGURES.resolve()}")

for path in sorted(FIGURES.glob("*.png")):
    print(f"  {path.name} ({path.stat().st_size:,} bytes)")
