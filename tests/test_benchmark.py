import pytest

from src.evaluation.benchmark import compare_results


def make_result(name, score, sample_ids):
    return {
        "name": name,
        "split": "test",
        "num_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "metrics": {
            "accuracy": score,
            "macro_precision": score,
            "macro_recall": score,
            "macro_f1": score,
        },
    }


def test_compare_results_ranks_by_macro_f1_and_checks_exact_samples():
    rows = compare_results(
        [
            make_result("model_a", 0.4, ["a.jpg", "b.jpg"]),
            make_result("model_b", 0.7, ["a.jpg", "b.jpg"]),
        ],
        require_same_samples=True,
    )
    assert [row["name"] for row in rows] == ["model_b", "model_a"]


def test_equal_counts_but_different_samples_are_rejected():
    with pytest.raises(ValueError, match="same test examples"):
        compare_results(
            [
                make_result("model_a", 0.4, ["a.jpg", "b.jpg"]),
                make_result("model_b", 0.7, ["a.jpg", "c.jpg"]),
            ],
            require_same_samples=True,
        )


def test_missing_ids_rejected_when_exact_comparison_requested():
    result = make_result("model_a", 0.4, ["a.jpg"])
    result.pop("sample_ids")
    with pytest.raises(ValueError, match="no sample_ids"):
        compare_results([result], require_same_samples=True)
