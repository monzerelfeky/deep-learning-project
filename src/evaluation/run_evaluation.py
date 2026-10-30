"""Run a reproducible evaluation benchmark across available checkpoints.

Example:
    python -m src.evaluation.run_evaluation --split test --device cpu

Optional explicit checkpoints:
    --checkpoint grid_search=outputs/checkpoints/tuning/grid/.../best_model.pt
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.evaluation.benchmark import compare_results, export_benchmark
from src.evaluation.evaluate import evaluate_model
from src.models.classifier import BirdClassifier
from src.models.scratch_cnn import ScratchCNNEncoder
from src.models.scratch_cnn_v2 import ScratchCNNEncoderV2
from src.models.transfer import ResNet18Encoder


DEFAULT_CHECKPOINTS = {
    "scratch_cnn_v1": "checkpoints/scratch_cnn/experiment_01/best_model.pt",
    "scratch_cnn_v2": "checkpoints/scratch_cnn/experiment_02/best_model.pt",
    "resnet18_frozen": "checkpoints/resnet18/resnet18_frozen/best_model.pt",
    "resnet18_finetune": "checkpoints/resnet18/resnet18_finetune/best_model.pt",
    "grid_search": (
        "outputs/checkpoints/tuning/grid/"
        "trial_0.001_0.0001_False/best_model.pt"
    ),
    "random_search": (
        "outputs/checkpoints/tuning/random/"
        "trial_0.001_0.0_False/best_model.pt"
    ),
}


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _model_spec(name, checkpoint, checkpoint_path):
    """Infer architecture and preprocessing family from config/checkpoint."""
    if name == "scratch_cnn_v1":
        return "ScratchCNNEncoder", "scratch_cnn", {}
    if name == "scratch_cnn_v2":
        return "ScratchCNNEncoderV2", "scratch_cnn", {}
    if name == "resnet18_frozen":
        return "ResNet18Encoder", "resnet18", {"pretrained": False, "freeze_backbone": True}
    if name == "resnet18_finetune":
        return "ResNet18Encoder", "resnet18", {"pretrained": False, "freeze_backbone": False}

    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    architecture = checkpoint.get("architecture") or model_config.get("architecture")
    if architecture not in {"ScratchCNNEncoder", "ScratchCNNEncoderV2", "ResNet18Encoder"}:
        raise ValueError(
            f"Cannot infer a supported architecture for {name!r} from "
            f"checkpoint {checkpoint_path}."
        )
    model_type = "resnet18" if architecture == "ResNet18Encoder" else "scratch_cnn"
    extra = {}
    if architecture == "ResNet18Encoder":
        extra = {
            "pretrained": False,
            "freeze_backbone": model_config.get("freeze_backbone", False),
        }
    return architecture, model_type, extra


def build_model_for_checkpoint(name, checkpoint, checkpoint_path, num_classes=200):
    architecture, _, extra = _model_spec(name, checkpoint, checkpoint_path)
    encoders = {
        "ScratchCNNEncoder": ScratchCNNEncoder,
        "ScratchCNNEncoderV2": ScratchCNNEncoderV2,
        "ResNet18Encoder": ResNet18Encoder,
    }
    encoder = encoders[architecture](**extra)
    model = BirdClassifier(encoder=encoder, num_classes=num_classes)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise ValueError(f"{checkpoint_path} has no model_state_dict or state_dict.")
    model.load_state_dict(state, strict=True)
    return model, _model_spec(name, checkpoint, checkpoint_path)[1]


def _parse_checkpoint(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --checkpoint.")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Checkpoint name and path cannot be empty.")
    return name.strip(), Path(path.strip())


def run_benchmark(
    checkpoints=None,
    split="test",
    output_dir="outputs/evaluation",
    device=None,
    batch_size=8,
    num_workers=0,
    split_path="configs/split_indices.json",
    config_path="configs/data.yaml",
):
    """Evaluate all available checkpoints on the same named dataset split."""
    from src.data.dataloaders import create_dataloaders

    checkpoint_map = dict(DEFAULT_CHECKPOINTS)
    if checkpoints:
        checkpoint_map.update({
            name: str(path) for name, path in checkpoints.items()
        })

    available = {
        name: Path(path) for name, path in checkpoint_map.items()
        if Path(path).is_file()
    }
    if not available:
        missing = "\n".join(f"  {name}: {path}" for name, path in checkpoint_map.items())
        raise FileNotFoundError(
            "No evaluation checkpoints were found. Expected one of:\n" + missing
        )

    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    results = []
    split_index = {"val": 1, "test": 2}
    if split not in split_index:
        raise ValueError("split must be 'val' or 'test'.")

    for name, checkpoint_path in available.items():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model, model_type = build_model_for_checkpoint(
            name, checkpoint, checkpoint_path
        )
        _, val_loader, test_loader = create_dataloaders(
            model_type=model_type,
            batch_size=batch_size,
            image_size=224,
            num_workers=num_workers,
            split_path=split_path,
            config_path=config_path,
            seed=42,
            return_file_name=True,
            train_augmentation=False,
        )
        loader = val_loader if split == "val" else test_loader

        evaluation = evaluate_model(
            model, loader, device=target_device, num_classes=200
        )
        results.append({
            "name": name,
            "split": split,
            "num_samples": evaluation["num_samples"],
            "sample_ids": evaluation.get("sample_ids"),
            "metrics": evaluation["metrics"],
            "inference_time_seconds": evaluation["inference_time_seconds"],
        })
        print(
            f"{name}: accuracy={evaluation['metrics']['accuracy']:.4f}, "
            f"macro-F1={evaluation['metrics']['macro_f1']:.4f}, "
            f"samples={evaluation['num_samples']}"
        )

    # Require exact filename/order equality, not merely equal sample counts.
    rows = compare_results(
        results, split=split, require_same_split=True, require_same_samples=True
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    export_benchmark(
        results, output / f"{split}_benchmark.csv",
        split=split, require_same_samples=True,
    )
    export_benchmark(
        results, output / f"{split}_benchmark.json",
        split=split, require_same_samples=True,
    )

    serializable = []
    for item in results:
        serializable.append({
            "name": item["name"],
            "split": item["split"],
            "num_samples": item["num_samples"],
            "sample_ids": item["sample_ids"],
            "inference_time_seconds": item["inference_time_seconds"],
            "metrics": {
                key: (value.tolist() if hasattr(value, "tolist") else value)
                for key, value in item["metrics"].items()
            },
        })
    with (output / f"{split}_details.json").open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2)

    print(f"\nSaved benchmark results to: {output.resolve()}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-path", default="configs/split_indices.json")
    parser.add_argument("--config-path", default="configs/data.yaml")
    parser.add_argument(
        "--checkpoint", action="append", type=_parse_checkpoint, default=[],
        help="Optional NAME=PATH checkpoint; may be repeated.",
    )
    args = parser.parse_args()
    run_benchmark(
        checkpoints=dict(args.checkpoint),
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_path=args.split_path,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
