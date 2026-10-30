
import pytest

from src.training.tuning import (
    load_tuning_config,
    generate_candidates,
    run_tuning,
)


@pytest.fixture
def config():
    return {
        "experiment": {"seed": 42},
        "tuning": {
            "metric": "val_macro_f1",
            "mode": "maximize",
            "strategies": {
                "grid": {
                    "learning_rate": [0.001, 0.0003],
                    "batch_size": [8],
                },
                "random": {
                    "n_iter": 2,
                    "learning_rate": [0.001, 0.0003, 0.0001],
                    "batch_size": [8],
                },
            },
        },
        "training": {},
        "model": {},
        "data": {},
    }


def test_generate_grid_candidates(config):
    candidates = generate_candidates(config, strategy="grid")

    assert len(candidates) == 2
    assert candidates[0]["learning_rate"] == 0.001
    assert candidates[1]["learning_rate"] == 0.0003


def test_generate_random_candidates(config):
    candidates = generate_candidates(config, strategy="random")

    assert len(candidates) == 2


def test_max_trials_limits_candidates(config):
    candidates = generate_candidates(
        config,
        strategy="grid",
        max_trials=1,
    )

    assert len(candidates) == 1


def test_run_tuning_selects_best_trial(config, tmp_path):
    config_path = tmp_path / "tuning.yaml"

    import yaml

    with open(config_path, "w") as file:
        yaml.safe_dump(config, file)

    def evaluator(params):
        scores = {
            0.001: 0.70,
            0.0003: 0.90,
        }

        return {"val_macro_f1": scores[params["learning_rate"]]}

    results = run_tuning(
        evaluator=evaluator,
        strategy="grid",
        config_path=str(config_path),
    )

    assert len(results["results"]) == 2
    assert results["best_score"] == 0.90
    assert results["best_params"]["learning_rate"] == 0.0003


def test_run_tuning_selects_lowest_metric(tmp_path):
    import yaml

    config = {
        "experiment": {"seed": 42},
        "tuning": {
            "metric": "val_loss",
            "mode": "minimize",
            "strategies": {
                "grid": {"learning_rate": [0.001, 0.0003]},
            },
        },
        "training": {},
        "model": {},
        "data": {},
    }

    config_path = tmp_path / "tuning.yaml"

    with open(config_path, "w") as file:
        yaml.safe_dump(config, file)

    def evaluator(params):
        return {
            "val_loss": 2.0 if params["learning_rate"] == 0.001 else 1.0
        }

    results = run_tuning(
        evaluator=evaluator,
        strategy="grid",
        config_path=str(config_path),
    )

    assert results["best_score"] == 1.0
    assert results["best_params"]["learning_rate"] == 0.0003
