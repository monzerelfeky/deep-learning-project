from pathlib import Path

import mlflow


class ExperimentTracker:
    """Small wrapper around MLflow experiment tracking."""

    def __init__(
        self,
        experiment_name="cub200-hyperparameter-tuning",
        tracking_uri="sqlite:///mlflow.db",
    ):
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name=None):
        """Start a new MLflow run."""
        if mlflow.active_run() is not None:
            raise RuntimeError("An MLflow run is already active.")

        mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        """Log hyperparameters."""
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run.")

        mlflow.log_params({
            str(key): str(value)
            for key, value in params.items()
        })

    def log_metrics(self, metrics, step=None):
        """Log numeric evaluation metrics."""
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run.")

        numeric_metrics = {}

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_metrics[str(key)] = float(value)

        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step=step)

    def log_artifact(self, artifact_path):
        """Log a file or directory as an MLflow artifact."""
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run.")

        path = Path(artifact_path)

        if not path.exists():
            raise FileNotFoundError(path)

        if path.is_dir():
            mlflow.log_artifacts(str(path))
        else:
            mlflow.log_artifact(str(path))

    def end_run(self, status="FINISHED"):
        """End the current MLflow run."""
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
