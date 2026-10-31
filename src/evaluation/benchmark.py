
"""Compare and export classification benchmark results."""

import csv
import json
from pathlib import Path

import numpy as np

AGGREGATE_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


def compare_results(
    results,
    split="test",
    require_same_split=True,
):
    """Convert named model evaluation outputs into comparable rows.

    Args:
        results: Sequence of dictionaries. Each item should contain:
            - name: model/run name
            - metrics: metrics dictionary, or an evaluation dictionary
              containing a nested 'metrics' dictionary
            - split: optional split name
            - num_samples: optional evaluated sample count
            - training_time_seconds: optional training duration
            - inference_time_seconds: optional inference duration
        split: Expected split name.
        require_same_split: If True, reject explicitly mismatched splits.

    Returns:
        A list of flat dictionaries, one per model.
    """
    if not results:
        raise ValueError("results must contain at least one model result.")

    rows = []
    sample_counts = set()

    for item in results:
        if not isinstance(item, dict):
            raise ValueError("Each result must be a dictionary.")

        name = item.get("name")
        if not name:
            raise ValueError("Every result must have a non-empty 'name'.")

        result_split = item.get("split", split)
        if require_same_split and result_split != split:
            raise ValueError(
                f"Model {name!r} used split {result_split!r}; expected {split!r}."
            )

        metrics = item.get("metrics", item)
        if not isinstance(metrics, dict):
            raise ValueError(f"Metrics for {name!r} must be a dictionary.")

        missing = [key for key in AGGREGATE_METRICS if key not in metrics]
        if missing:
            raise ValueError(
                f"Result {name!r} is missing metrics: {', '.join(missing)}."
            )

        row = {
            "name": str(name),
            "split": result_split,
            "num_samples": item.get("num_samples"),
            "accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "training_time_seconds": item.get("training_time_seconds"),
            "inference_time_seconds": item.get("inference_time_seconds"),
        }

        if row["num_samples"] is not None:
            sample_counts.add(row["num_samples"])

        rows.append(row)

    if len(sample_counts) > 1:
        raise ValueError(
            "Results use different sample counts; compare models on the same "
            "evaluation examples."
        )

    return sorted(rows, key=lambda row: row["macro_f1"], reverse=True)


def export_benchmark(results, output_path, split="test", format=None):
    """Export benchmark rows to CSV or JSON.

    Format is inferred from the extension unless explicitly supplied.
    Returns the path to the exported file.
    """
    rows = compare_results(results, split=split)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_format = (format or path.suffix.lstrip(".")).lower()

    if output_format == "json":
        with path.open("w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2, ensure_ascii=False)
    elif output_format == "csv":
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("format must be 'csv' or 'json'.")

    return path


def summarize_benchmark(rows):
    """Return the highest macro-F1 row and the mean aggregate metrics."""
    if not rows:
        raise ValueError("rows must not be empty.")

    best = max(rows, key=lambda row: row["macro_f1"])
    averages = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in AGGREGATE_METRICS
    }

    return {
        "best_model": best["name"],
        "best_macro_f1": best["macro_f1"],
        "mean_metrics": averages,
        "num_models": len(rows),
    }
