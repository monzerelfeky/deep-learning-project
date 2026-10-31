
import random
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataloaders import create_dataloaders
from src.models.classifier import BirdClassifier
from src.models.scratch_cnn import ScratchCNNEncoder
from src.models.scratch_cnn_v2 import ScratchCNNEncoderV2
from src.models.transfer import ResNet18Encoder
from src.training.train_scratch import train_one_epoch, validate_one_epoch
from src.evaluation.evaluate import evaluate_model
from src.training.tuning import run_tuning
from src.utils.tracking import ExperimentTracker


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def backup_mlflow_to_drive():
    """Back up MLflow database and artifacts locally."""
    import sqlite3

    backup_dir = Path("outputs/backup/mlflow")
    backup_dir.mkdir(parents=True, exist_ok=True)

    source_db = Path("mlflow.db")

    if source_db.exists():
        destination_db = backup_dir / "mlflow.db"

        with sqlite3.connect(str(source_db)) as source_conn:
            with sqlite3.connect(str(destination_db)) as destination_conn:
                source_conn.backup(destination_conn)

        print("MLflow database backed up.")

    for artifact_dir in ("mlruns", "mlartifacts"):
        source_artifacts = Path(artifact_dir)

        if source_artifacts.exists():
            shutil.copytree(
                source_artifacts,
                backup_dir / artifact_dir,
                dirs_exist_ok=True,
            )
            print(f"{artifact_dir} backed up.")



def build_model(config):
    """Build the selected encoder and 200-class classifier."""
    model_config = config["model"]
    architecture = model_config["architecture"]
    num_classes = model_config.get("num_classes", 200)

    if architecture == "ScratchCNNEncoder":
        encoder = ScratchCNNEncoder()

    elif architecture == "ScratchCNNEncoderV2":
        encoder = ScratchCNNEncoderV2()

    elif architecture == "ResNet18Encoder":
        pretrained = model_config.get("pretrained", True)
        freeze_backbone = model_config.get("freeze_backbone", False)

        encoder = ResNet18Encoder(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return BirdClassifier(
        encoder=encoder,
        num_classes=num_classes,
    )


def build_optimizer(model, config, parameters):
    """Create the configured optimizer."""
    optimizer_name = config["training"].get("optimizer", "AdamW")
    learning_rate = parameters["learning_rate"]
    weight_decay = parameters["weight_decay"]

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters.")

    if optimizer_name == "Adam":
        return Adam(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "AdamW":
        return AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def run_project_tuning(
    strategy="grid",
    config_path="configs/tuning.yaml",
    tracking=True,
    max_trials=None,
    recovery_path="outputs/results/recovered_trials.json",
):
    """Run the complete project hyperparameter-tuning workflow."""

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["experiment"].get("seed", 42)
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("CUB-200 HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Strategy: {strategy}")
    print(f"Architecture: {config['model']['architecture']}")

    data_config = config["data"]
    training_config = config["training"]

    epochs = training_config.get("epochs", 20)
    num_classes = config["model"].get("num_classes", 200)

    scheduler_config = training_config.get("scheduler", {})

    tracker = None

    if tracking:
        tracker = ExperimentTracker(
            experiment_name=config["experiment"]["name"],
            tracking_uri=config["experiment"].get(
                "tracking_uri", "sqlite:///mlflow.db"
            ),
        )

    def evaluator(parameters):
        """Train and evaluate one hyperparameter configuration."""

        trial_seed = seed
        set_seed(trial_seed)

        model_type = data_config.get("model_type", "scratch_cnn")

        train_loader, val_loader, _ = create_dataloaders(
            model_type=model_type,
            batch_size=parameters["batch_size"],
            image_size=data_config.get("image_size", 224),
            num_workers=data_config.get("num_workers", 0),
            split_path=data_config.get("split_path"),
            config_path=data_config.get("config_path"),
            seed=trial_seed,
            train_augmentation=parameters.get(
                "use_augmentation", True
            ),
        )

        model = build_model(config).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = build_optimizer(
            model,
            config,
            parameters,
        )

        scheduler = None

        if scheduler_config.get("name") == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get("monitor", "val_loss")
                == "val_loss" and "min" or "max",
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 3),
                min_lr=scheduler_config.get("min_lr", 1e-6),
            )

        best_f1 = -1.0
        best_metrics = None
        best_epoch = 0

        checkpoint_dir = (
            Path("outputs/checkpoints/tuning")
            / strategy
            / f"trial_{parameters.get('learning_rate')}_{parameters.get('weight_decay')}_{parameters.get('use_augmentation')}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / "best_model.pt"

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_accuracy = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            val_loss, val_accuracy, val_f1 = validate_one_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
            )

            if scheduler is not None:
                scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Train loss: {train_loss:.4f} | "
                f"Train accuracy: {train_accuracy:.2%}"
            )
            print(
                f"Validation loss: {val_loss:.4f} | "
                f"Validation accuracy: {val_accuracy:.2%} | "
                f"Macro-F1: {val_f1:.4f}"
            )
            print(f"Learning rate: {current_lr:.7f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch

                # Evaluate the best checkpoint to collect all metrics.
                evaluation = evaluate_model(
                    model,
                    val_loader,
                    device=device,
                    num_classes=num_classes,
                )

                best_metrics = evaluation["metrics"]

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_f1,
                    "parameters": parameters,
                    "architecture": config["model"]["architecture"],
                    "config": config,
                }

                torch.save(checkpoint, checkpoint_path)

                print(f"Saved new best checkpoint: {checkpoint_path}")

        if best_metrics is None:
            raise RuntimeError("No valid validation result was produced.")

        return {
            "val_macro_f1": float(best_metrics["macro_f1"]),
            "val_accuracy": float(best_metrics["accuracy"]),
            "val_precision": float(best_metrics["macro_precision"]),
            "val_recall": float(best_metrics["macro_recall"]),
            "val_loss": float(val_loss),
            "best_epoch": int(best_epoch),
            "learning_rate_final": float(
                optimizer.param_groups[0]["lr"]
            ),
        }

    import json
    import pandas as pd

    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    recovery_file = Path(recovery_path)
    recovered_results = []

    if recovery_file.exists():
        with open(recovery_file, "r", encoding="utf-8") as file:
            recovered_results = json.load(file)

    skip_params = [
        trial["params"] for trial in recovered_results
    ]

    csv_path = output_dir / f"{strategy}_results.csv"
    best_config_path = output_dir / f"{strategy}_best_config.json"

    completed_results = []

    def save_results(all_results):
        """Save the combined results and current best configuration."""
        all_results = sorted(
            all_results,
            key=lambda item: item["trial"],
        )

        rows = []

        for trial in all_results:
            rows.append({
                "trial": trial["trial"],
                **trial["params"],
                **trial["metrics"],
                "recovered_from_log": trial.get(
                    "recovered_from_log", False
                ),
                "checkpoint_available": trial.get(
                    "checkpoint_available", True
                ),
            })

        pd.DataFrame(rows).to_csv(csv_path, index=False)

        if not all_results:
            return

        metric = config["tuning"]["metric"]
        mode = config["tuning"]["mode"]

        best = (
            max(
                all_results,
                key=lambda item: item["metrics"][metric],
            )
            if mode == "maximize"
            else min(
                all_results,
                key=lambda item: item["metrics"][metric],
            )
        )

        with open(best_config_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "best_params": best["params"],
                    "best_score": best["metrics"][metric],
                    "selection_metric": metric,
                    "mode": mode,
                    "best_trial": best["trial"],
                    "recovered_from_log": best.get(
                        "recovered_from_log", False
                    ),
                    "checkpoint_available": best.get(
                        "checkpoint_available", True
                    ),
                },
                file,
                indent=4,
            )

    # Preserve the recovered results before starting new trials.
    save_results(recovered_results)

    def record_trial(trial_number, result):
        completed_results.append(result)

        all_results = recovered_results + completed_results
        save_results(all_results)

        if tracking:
            backup_mlflow_to_drive()

        print(f"Saved results after trial {trial_number}.")

    results = run_tuning(
        evaluator=evaluator,
        strategy=strategy,
        config_path=config_path,
        tracker=tracker,
        max_trials=max_trials,
        initial_results=recovered_results,
        skip_params=skip_params,
        on_trial_complete=record_trial,
    )

    # Final export.
    save_results(results["results"])

    print("\\n" + "=" * 60)
    print("TUNING COMPLETED")
    print("=" * 60)
    print("Trials represented:", len(results["results"]))
    print("New trials executed:", len(completed_results))
    print("Best parameters:", results["best_params"])
    print("Best score:", results["best_score"])
    print("Results CSV:", csv_path)
    print("Best configuration:", best_config_path)

    return results
