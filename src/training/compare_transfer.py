"""Compare best validation results for Scratch CNN and ResNet18 experiments.

Run from the project root:
    python -m src.training.compare_transfer

Default inputs:
    outputs/training/scratch_cnn_experiment_01.csv
    outputs/training/scratch_cnn_experiment_02.csv
    outputs/training/resnet18_frozen_20ep.csv
    outputs/training/resnet18_finetune_20ep.csv

This script reads validation histories only. It does not access the test set.
The selected row is the epoch with the highest validation macro-F1; accuracy
is reported from that same epoch.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def best_validation_row(path: Path, model_name: str) -> dict:
    """Return metrics from the epoch with the highest validation macro-F1."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing history CSV: {path}\n"
            "Make sure the file exists or pass its location using the matching argument."
        )

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"History CSV is empty: {path}")

    required = {"epoch", "val_accuracy", "val_macro_f1"}
    if not required.issubset(rows[0].keys()):
        raise ValueError(
            f"{path} must contain columns: {sorted(required)}. "
            f"Found: {list(rows[0].keys())}"
        )

    try:
        best = max(rows, key=lambda row: float(row["val_macro_f1"]))
        return {
            "model": model_name,
            "history_csv": str(path),
            "best_epoch_by_val_macro_f1": int(best["epoch"]),
            "best_val_accuracy": float(best["val_accuracy"]),
            "best_val_macro_f1": float(best["val_macro_f1"]),
        }
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid metric value in {path}: {exc}") from exc


def save_bar_chart(records: list[dict], metric: str, label: str, filename: str, out_dir: Path) -> None:
    """Save a bar chart for the requested metric."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = [record["model"] for record in records]
    values = [record[metric] for record in records]

    bars = ax.bar(names, values)
    ax.set_title(label)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(0.05, max(values) * 1.15))
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch-v1",
        default="outputs/training/scratch_cnn_experiment_01.csv",
        help="CSV history for Scratch CNN V1.",
    )
    parser.add_argument(
        "--scratch-v2",
        default="outputs/training/scratch_cnn_experiment_02.csv",
        help="CSV history for Scratch CNN V2.",
    )
    parser.add_argument(
        "--frozen",
        default="outputs/training/resnet18_frozen_20ep.csv",
        help="CSV history for frozen-backbone ResNet18.",
    )
    parser.add_argument(
        "--finetune",
        default="outputs/training/resnet18_finetune_20ep.csv",
        help="CSV history for fine-tuned ResNet18.",
    )
    parser.add_argument("--out-dir", default="outputs/comparison")
    args = parser.parse_args()

    experiments = [
        (Path(args.scratch_v1), "Scratch CNN V1"),
        (Path(args.scratch_v2), "Scratch CNN V2"),
        (Path(args.frozen), "ResNet18 frozen backbone"),
        (Path(args.finetune), "ResNet18 fine-tuned"),
    ]
    records = [
        best_validation_row(path, name)
        for path, name in experiments
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "transfer_comparison.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    save_bar_chart(
        records,
        "best_val_accuracy",
        "Validation accuracy at best macro-F1 epoch",
        "best_val_accuracy.png",
        out_dir,
    )
    save_bar_chart(
        records,
        "best_val_macro_f1",
        "Best validation macro-F1",
        "best_val_macro_f1.png",
        out_dir,
    )

    print(f"Comparison table: {summary_path}")
    for record in records:
        print(
            f"{record['model']}: "
            f"best_epoch={record['best_epoch_by_val_macro_f1']}, "
            f"val_accuracy={record['best_val_accuracy']:.4f}, "
            f"val_macro_f1={record['best_val_macro_f1']:.4f}"
        )
    print(f"Charts saved under: {out_dir}")


if __name__ == "__main__":
    main()
