import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics


def test_compute_metrics_hand_checked():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    result = compute_metrics(y_true, y_pred, num_classes=2)

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert result["macro_recall"] == pytest.approx((0.5 + 1.0) / 2)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    np.testing.assert_array_equal(
        result["confusion_matrix"], np.array([[1, 1], [0, 2]])
    )
    np.testing.assert_allclose(
        result["normalized_confusion_matrix"],
        np.array([[0.5, 0.5], [0.0, 1.0]]),
    )
    assert len(result["per_class_f1"]) == 2


def test_absent_class_is_included_as_zero():
    result = compute_metrics([0, 0], [0, 0], num_classes=3)
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["per_class_f1"][1] == 0
    assert result["per_class_f1"][2] == 0


@pytest.mark.parametrize(
    "true,pred,n",
    [
        ([], [], 2),
        ([0, 1], [0], 2),
        ([[0], [1]], [0, 1], 2),
        ([0, 2], [0, 1], 2),
        ([0.0, 1.0], [0, 1], 2),
    ],
)
def test_invalid_inputs_raise(true, pred, n):
    with pytest.raises(ValueError):
        compute_metrics(true, pred, num_classes=n)
