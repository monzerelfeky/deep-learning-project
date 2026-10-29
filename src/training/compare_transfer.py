"""Compare best validation results for the scratch and transfer-learning runs.

Run after the scratch CNN and both ResNet18 modes have completed:
    python -m src.training.compare_transfer

Pass alternative result files with --scratch, --frozen, and --finetune.
This script only reads validation histories; it does not access the test set.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def best_validation_row(path: Path, model_name: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing history CSV: {path}\nTrain that experiment first or pass its path as an argument."
        )
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"History CSV is empty: {path}")
    required = {"epoch", "val_accuracy", "val_macro_f1"}
    if not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    best = max(rows, key=lambda row: float(row["val_macro_f1"]))
    return {
        "model": model_name,
        "history_csv": str(path),
        "best_epoch_by_val_macro_f1": int(best["epoch"]),
        "best_val_accuracy": float(best["val_accuracy"]),
        "best_val_macro_f1": float(best["val_macro_f1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", default="outputs/training/scratch_cnn_experiment_02.csv")
    parser.add_argument("--frozen", default="outputs/training/resnet18_frozen.csv")
    parser.add_argument("--finetune", default="outputs/training/resnet18_finetune.csv")
    parser.add_argument("--out-dir", default="outputs/comparison")
    args = parser.parse_args()

    records = [
        best_validation_row(Path(args.scratch), "Scratch CNN V2"),
        best_validation_row(Path(args.frozen), "ResNet18 frozen backbone"),
        best_validation_row(Path(args.finetune), "ResNet18 fine-tuned"),
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "transfer_comparison.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    for metric, label, filename in (
        ("best_val_accuracy", "Best validation accuracy", "best_val_accuracy.png"),
        ("best_val_macro_f1", "Best validation macro-F1", "best_val_macro_f1.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [record["model"] for record in records]
        values = [record[metric] for record in records]
        bars = ax.bar(names, values)
        ax.set_title(label)
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(0.05, max(values) * 1.15))
        ax.tick_params(axis="x", labelrotation=12)
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

    print(f"Comparison table: {summary_path}")
    for record in records:
        print(
            f"{record['model']}: epoch={record['best_epoch_by_val_macro_f1']}, "
            f"val_accuracy={record['best_val_accuracy']:.4f}, "
            f"val_macro_f1={record['best_val_macro_f1']:.4f}"
        )
    print(f"Charts saved under: {out_dir}")


if __name__ == "__main__":
    main()
