import itertools
import random
from pathlib import Path

import yaml


def load_tuning_config(config_path):
    """Load and validate the tuning YAML configuration."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required = ["experiment", "tuning", "training", "model", "data"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing configuration section: {key}")

    tuning = config["tuning"]

    if tuning.get("mode") not in ("minimize", "maximize"):
        raise ValueError("Tuning mode must be 'minimize' or 'maximize'.")

    if not tuning.get("metric"):
        raise ValueError("A tuning metric must be specified.")

    return config


def generate_candidates(config, strategy="grid", max_trials=None):
    """Generate hyperparameter combinations for grid or random search."""
    if max_trials is not None:
        if not isinstance(max_trials, int) or isinstance(max_trials, bool):
            raise ValueError("max_trials must be a positive integer or None.")
        if max_trials <= 0:
            raise ValueError("max_trials must be a positive integer.")

    strategies = config["tuning"]["strategies"]

    if strategy not in strategies:
        raise ValueError(f"Unknown tuning strategy: {strategy}")

    strategy_config = strategies[strategy]
    n_iter = strategy_config.get("n_iter", 4)

    parameters = {
        key: value
        for key, value in strategy_config.items()
        if key != "n_iter"
    }

    if not parameters:
        raise ValueError("No hyperparameters specified.")

    for key, values in parameters.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Hyperparameter '{key}' must contain a non-empty list."
            )

    keys = list(parameters.keys())
    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*(parameters[key] for key in keys))
    ]

    if strategy == "random":
        if not isinstance(n_iter, int) or isinstance(n_iter, bool) or n_iter <= 0:
            raise ValueError("n_iter must be a positive integer.")

        random.seed(config["experiment"].get("seed", 42))
        count = min(n_iter, len(combinations))
        combinations = random.sample(combinations, count)

    if max_trials is not None:
        combinations = combinations[:max_trials]

    return combinations


def run_tuning(
    evaluator,
    strategy="grid",
    config_path="configs/tuning.yaml",
    tracker=None,
    max_trials=None,
    on_trial_complete=None,
    initial_results=None,
    skip_params=None,
):
    """
    Run hyperparameter tuning, optionally retaining recovered results
    and skipping candidates that have already been completed.
    """
    config = load_tuning_config(config_path)
    candidates = generate_candidates(
        config,
        strategy=strategy,
        max_trials=max_trials,
    )

    metric = config["tuning"]["metric"]
    mode = config["tuning"]["mode"]

    results = list(initial_results or [])
    skip_params = list(skip_params or [])
    best_result = None

    # Validate and include recovered results in best-result selection.
    for recovered in results:
        if metric not in recovered.get("metrics", {}):
            raise KeyError(
                f"Recovered trial is missing selection metric '{metric}'."
            )

        score = recovered["metrics"][metric]

        if score is None:
            raise ValueError(
                f"Recovered trial metric '{metric}' cannot be None."
            )

        is_better = (
            best_result is None
            or (
                score > best_result["metrics"][metric]
                if mode == "maximize"
                else score < best_result["metrics"][metric]
            )
        )

        if is_better:
            best_result = recovered

    for trial_number, parameters in enumerate(candidates, start=1):
        if parameters in skip_params:
            print(
                f"Skipping Trial {trial_number}/{len(candidates)} "
                f"(already recovered)."
            )
            continue

        print(f"Trial {trial_number}/{len(candidates)}")
        print(f"Parameters: {parameters}")

        if tracker is not None:
            tracker.start_run(
                run_name=f"{strategy}-trial-{trial_number}"
            )
            tracker.log_params(parameters)

        result = None

        try:
            metrics = evaluator(parameters)

            if not isinstance(metrics, dict):
                raise TypeError(
                    "Evaluator must return a metrics dictionary."
                )

            if metric not in metrics:
                raise KeyError(
                    f"Selection metric '{metric}' missing from evaluator results."
                )

            score = metrics[metric]

            if score is None:
                raise ValueError(
                    f"Metric '{metric}' cannot be None."
                )

            result = {
                "trial": trial_number,
                "params": parameters,
                "metrics": metrics,
                "recovered_from_log": False,
                "checkpoint_available": True,
            }

            results.append(result)

            if tracker is not None:
                tracker.log_metrics(metrics)

            is_better = (
                best_result is None
                or (
                    score > best_result["metrics"][metric]
                    if mode == "maximize"
                    else score < best_result["metrics"][metric]
                )
            )

            if is_better:
                best_result = result

            print(f"{metric}: {score:.5f}")

        finally:
            if tracker is not None:
                tracker.end_run()

        if on_trial_complete is not None:
            on_trial_complete(trial_number, result)

    return {
        "results": results,
        "best_result": best_result,
        "best_params": (
            best_result["params"] if best_result is not None else None
        ),
        "best_score": (
            best_result["metrics"][metric]
            if best_result is not None else None
        ),
        "metric": metric,
        "mode": mode,
        "total_candidates": len(candidates),
    }
