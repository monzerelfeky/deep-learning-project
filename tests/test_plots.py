import numpy as np
import pytest

from src.visualization.plots import (
    plot_class_f1,
    plot_confusion_matrix,
    plot_prediction_examples,
)


def test_confusion_matrix_plot_returns_figure():
    fig = plot_confusion_matrix(np.array([[3, 1], [0, 2]]))
    assert fig.axes
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_class_f1_rejects_mismatched_names():
    with pytest.raises(ValueError, match="length"):
        plot_class_f1([0.2, 0.8], class_names=["only-one"])


def test_prediction_examples_plot():
    images = np.random.rand(3, 8, 8, 3)
    fig = plot_prediction_examples(
        images,
        y_true=[0, 1, 1],
        y_pred=[0, 0, 1],
        confidence=[0.9, 0.7, 0.8],
        correct=True,
    )
    assert fig.axes
    import matplotlib.pyplot as plt
    plt.close(fig)
