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
    require_same_samples=False,
):
    """Validate and rank model results by macro-F1.

    If ``require_same_samples`` is True, every result must include
    ``sample_ids`` and the ordered IDs must match exactly. This is stronger
    than comparing sample counts and guards against evaluating different
    examples that happen to have equal-sized splits.
    """
    if not results:
        raise ValueError("results must contain at least one model result.")

    rows = []
    reference_ids = None
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

        sample_ids = item.get("sample_ids")
        if require_same_samples and sample_ids is None:
            raise ValueError(
                f"Result {name!r} has no sample_ids; exact split comparison "
                "cannot be verified."
            )
        if sample_ids is not None:
            ids = tuple(str(value) for value in sample_ids)
            if len(set(ids)) != len(ids):
                raise ValueError(f"Result {name!r} contains duplicate sample IDs.")
            if item.get("num_samples") is not None and len(ids) != item["num_samples"]:
                raise ValueError(
                    f"Result {name!r}: sample_ids length does not match num_samples."
                )
            if reference_ids is None:
                reference_ids = ids
            elif require_same_samples and ids != reference_ids:
                raise ValueError(
                    "Results do not contain the same test examples in the same order."
                )

        count = item.get("num_samples")
        if count is not None:
            sample_counts.add(count)

        rows.append({
            "name": str(name),
            "split": result_split,
            "num_samples": count,
            "accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "training_time_seconds": item.get("training_time_seconds"),
            "inference_time_seconds": item.get("inference_time_seconds"),
        })

    if len(sample_counts) > 1:
        raise ValueError(
            "Results use different sample counts; compare models on the same "
            "evaluation examples."
        )
    return sorted(rows, key=lambda row: row["macro_f1"], reverse=True)


def export_benchmark(results, output_path, split="test", format=None,
                     require_same_samples=False):
    """Export benchmark rows to CSV or JSON and return the output path."""
    rows = compare_results(
        results,
        split=split,
        require_same_samples=require_same_samples,
    )
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
    """Return the highest macro-F1 row and mean aggregate metrics."""
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
