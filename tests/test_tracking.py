
from pathlib import Path

import mlflow
import pytest

from src.utils.tracking import ExperimentTracker


@pytest.fixture
def tracker(tmp_path):
    """Create an isolated MLflow tracker for each test."""
    tracking_uri = f"sqlite:///{tmp_path / 'test_mlflow.db'}"

    return ExperimentTracker(
        experiment_name="tracking-unit-tests",
        tracking_uri=tracking_uri,
    )


def test_experiment_creation(tracker):
    """Test that an MLflow experiment is created."""
    experiment = mlflow.get_experiment_by_name(
        "tracking-unit-tests"
    )

    assert experiment is not None
    assert experiment.name == "tracking-unit-tests"


def test_log_parameters(tracker):
    """Test logging hyperparameters."""
    tracker.start_run("parameter-test")

    tracker.log_params({
        "learning_rate": 0.001,
        "batch_size": 8,
    })

    run_id = mlflow.active_run().info.run_id

    tracker.end_run()

    run = mlflow.get_run(run_id)

    assert run.data.params["learning_rate"] == "0.001"
    assert run.data.params["batch_size"] == "8"


def test_log_metrics(tracker):
    """Test logging evaluation metrics."""
    tracker.start_run("metric-test")

    tracker.log_metrics({
        "val_macro_f1": 0.85,
        "val_accuracy": 0.90,
    })

    run_id = mlflow.active_run().info.run_id

    tracker.end_run()

    run = mlflow.get_run(run_id)

    assert run.data.metrics["val_macro_f1"] == 0.85
    assert run.data.metrics["val_accuracy"] == 0.90


def test_run_finishes_successfully(tracker):
    """Test that an MLflow run finishes correctly."""
    tracker.start_run("completion-test")

    run_id = mlflow.active_run().info.run_id

    tracker.end_run()

    run = mlflow.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert mlflow.active_run() is None


def test_log_artifact(tracker, tmp_path):
    """Test that artifacts are physically saved."""
    artifact = tmp_path / "test.txt"
    artifact.write_text("Member 5 tracking test")

    tracker.start_run("artifact-test")

    run_id = mlflow.active_run().info.run_id

    tracker.log_artifact(artifact)

    tracker.end_run()

    run = mlflow.get_run(run_id)

    artifact_uri = run.info.artifact_uri

    # Convert the local artifact URI into a filesystem path.
    if artifact_uri.startswith("file://"):
        artifact_dir = Path(artifact_uri[7:])
    else:
        artifact_dir = Path(artifact_uri)

    saved_artifact = artifact_dir / "test.txt"

    assert saved_artifact.exists()
    assert saved_artifact.read_text() == "Member 5 tracking test"


def test_log_missing_artifact_raises_error(tracker, tmp_path):
    """Test that logging a nonexistent artifact raises an error."""
    tracker.start_run("missing-artifact-test")

    missing_file = tmp_path / "missing.txt"

    try:
        with pytest.raises(FileNotFoundError):
            tracker.log_artifact(missing_file)
    finally:
        tracker.end_run()
