# Member 6 — Evaluation and Visualization Handoff

This handoff contains only Member 6 files. It does not modify model, data,
training, tuning, or configuration files owned by other team members.

## Implemented

- `src/evaluation/metrics.py`: aggregate and per-class metrics, confusion
  matrices, and normalized confusion matrices.
- `src/evaluation/evaluate.py`: inference, predictions, confidence,
  probabilities, sample filenames, and inference timing. The model's original
  device and train/eval mode are restored.
- `src/evaluation/benchmark.py`: result validation, macro-F1 ranking, CSV/JSON
  export, and optional exact ordered sample-ID validation.
- `src/evaluation/run_evaluation.py`: checkpoint discovery/loading and
  evaluation of available models on the same validation or test examples.
- `src/visualization/gradcam.py`: generic Grad-CAM and heatmap overlay.
- `src/visualization/plots.py`: confusion matrix, per-class F1, and prediction
  example plots.
- `tests/test_metrics.py`, `tests/test_evaluate.py`, `tests/test_benchmark.py`,
  `tests/test_gradcam.py`, `tests/test_plots.py`: unit tests.

## Checkpoints present in the supplied branch archive

The archive contains these two tuning checkpoints:

- `outputs/checkpoints/tuning/grid/trial_0.001_0.0001_False/best_model.pt`
- `outputs/checkpoints/tuning/random/trial_0.001_0.0_False/best_model.pt`

The archive does not contain the baseline ScratchCNN or ResNet18 checkpoints
at the default paths in `run_evaluation.py`. Those entries are skipped
automatically. If your team stores them elsewhere, pass their paths with
`--checkpoint NAME=PATH`.

The recovered Grid Search score 0.19384 is not used: Member 5's report says
its checkpoint is unavailable. The available newly trained Grid Search
checkpoint (validation macro-F1 0.18788) and Random Search checkpoint
(validation macro-F1 0.16925) are used instead.

## Run

From the repository root, after installing `requirements.txt` and making the
CUB data available through the existing dataset pipeline:

```powershell
python -m pytest tests/test_metrics.py tests/test_evaluate.py tests/test_benchmark.py tests/test_gradcam.py tests/test_plots.py -v
python -m src.evaluation.run_evaluation --split test --device cpu
```

For CUDA, replace `cpu` with `cuda`. Use `--split val` for validation only.
The benchmark verifies the ordered filenames are identical across models;
equal sample counts alone are not treated as proof of a fair comparison.

Evaluation output is written under `outputs/evaluation/`:
- `<split>_benchmark.csv`
- `<split>_benchmark.json`
- `<split>_details.json`

Do not select a model using test results. Use validation results for model
selection, then run the held-out test evaluation once the selection is final.

## Important environment note

In the isolated execution environment used to inspect this archive, the
`datasets` package was not installed and the CUB dataset was not included.
Therefore, the new Member 6 unit tests were run independently of the
repository-wide `tests/conftest.py`; the complete repository suite and
end-to-end CUB checkpoint evaluation still need to be run in the project's
configured environment.
