
"""Plot CNN V1 vs CNN V2 training histories."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs/figures")
HISTORY_DIR = Path("outputs/training")


def load_history(filename):
    path = HISTORY_DIR / filename

    with path.open("r", newline="", encoding="utf-8") as file:
        return [
            {
                key: float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(file)
        ]


def plot_metric(histories, metric, ylabel, filename, percent=False):
    plt.figure(figsize=(9, 5))

    for name, history in histories.items():
        epochs = [int(row["epoch"]) for row in history]
        values = [row[metric] for row in history]

        if percent:
            values = [value * 100 for value in values]

        plt.plot(
            epochs,
            values,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=name,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel + " — CNN V1 vs CNN V2")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    histories = {
        "CNN V1": load_history("scratch_cnn_experiment_01.csv"),
        "CNN V2": load_history("scratch_cnn_experiment_02.csv"),
    }

    for metric, ylabel, filename, percent in [
        ("train_accuracy", "Training Accuracy (%)",
         "scratch_train_accuracy.png", True),
        ("val_accuracy", "Validation Accuracy (%)",
         "scratch_val_accuracy.png", True),
        ("train_loss", "Training Loss",
         "scratch_train_loss.png", False),
        ("val_loss", "Validation Loss",
         "scratch_val_loss.png", False),
        ("val_macro_f1", "Validation Macro-F1",
         "scratch_val_f1.png", False),
    ]:
        plot_metric(histories, metric, ylabel, filename, percent)

    print("Saved figures to:", OUTPUT_DIR.resolve())

    for name, history in histories.items():
        best = max(history, key=lambda row: row["val_macro_f1"])
        print(
            f"{name}: best epoch={int(best['epoch'])}, "
            f"val_accuracy={best['val_accuracy']:.2%}, "
            f"macro_f1={best['val_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
