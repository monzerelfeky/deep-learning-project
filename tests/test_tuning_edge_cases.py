
import pytest

from src.training.tuning import (
    generate_candidates,
    load_tuning_config,
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
                },
            },
        },
        "training": {},
        "model": {},
        "data": {},
    }


def test_invalid_strategy_raises_error(config):
    with pytest.raises(ValueError, match="Unknown tuning strategy"):
        generate_candidates(config, strategy="invalid")


@pytest.mark.parametrize("max_trials", [0, -1, 1.5, True])
def test_invalid_max_trials_raises_error(config, max_trials):
    with pytest.raises(ValueError, match="max_trials"):
        generate_candidates(
            config,
            strategy="grid",
            max_trials=max_trials,
        )


def test_empty_hyperparameter_list_raises_error(config):
    config["tuning"]["strategies"]["grid"]["learning_rate"] = []

    with pytest.raises(ValueError, match="non-empty list"):
        generate_candidates(config, strategy="grid")


def test_invalid_mode_raises_error(tmp_path):
    import yaml

    config = {
        "experiment": {},
        "tuning": {
            "metric": "val_macro_f1",
            "mode": "invalid",
            "strategies": {"grid": {"learning_rate": [0.001]}},
        },
        "training": {},
        "model": {},
        "data": {},
    }

    path = tmp_path / "invalid.yaml"

    with open(path, "w") as file:
        yaml.safe_dump(config, file)

    with pytest.raises(ValueError, match="Tuning mode"):
        load_tuning_config(path)


def test_missing_selection_metric_raises_error(config, tmp_path):
    import yaml

    path = tmp_path / "tuning.yaml"

    with open(path, "w") as file:
        yaml.safe_dump(config, file)

    def evaluator(params):
        return {"val_accuracy": 0.85}

    with pytest.raises(KeyError, match="Selection metric"):
        run_tuning(
            evaluator=evaluator,
            strategy="grid",
            config_path=str(path),
        )


def test_evaluator_must_return_dictionary(config, tmp_path):
    import yaml

    path = tmp_path / "tuning.yaml"

    with open(path, "w") as file:
        yaml.safe_dump(config, file)

    def evaluator(params):
        return 0.85

    with pytest.raises(TypeError, match="metrics dictionary"):
        run_tuning(
            evaluator=evaluator,
            strategy="grid",
            config_path=str(path),
        )


def test_none_metric_raises_error(config, tmp_path):
    import yaml

    path = tmp_path / "tuning.yaml"

    with open(path, "w") as file:
        yaml.safe_dump(config, file)

    def evaluator(params):
        return {"val_macro_f1": None}

    with pytest.raises(ValueError, match="cannot be None"):
        run_tuning(
            evaluator=evaluator,
            strategy="grid",
            config_path=str(path),
        )
