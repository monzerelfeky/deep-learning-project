
# Member 5 — Hyperparameter Tuning and Experiment Tracking

## 1. Objective

Evaluate Grid Search and Random Search for the CUB-200
classification task using validation macro-F1 as the selection metric.

## 2. Experimental Setup

- Model: ScratchCNNEncoderV2
- Dataset: CUB-200
- Image size: 224
- Batch size: 8
- Optimizer: AdamW
- Training budget: 20 epochs per trial
- Scheduler: ReduceLROnPlateau
- Selection metric: validation macro-F1
- Optimization direction: maximize

### Search spaces

Grid Search:
- Learning rate: 0.001, 0.0003, 0.0001
- Weight decay: 0.0, 0.0001
- Augmentation: enabled or disabled
- Total combinations: 12

Random Search:
- Number of trials: 4
- Learning rate, weight decay, and augmentation sampled
  from the same configured search spaces.

## 3. Results

Grid Search:
- 12 trials represented.
- 10 newly trained trials.
- Best overall score: 0.19384 (recovered trial; checkpoint unavailable).
- Best newly trained score: 0.18788.
- Best newly trained checkpoint: available.

Random Search:
- 4 newly trained trials.
- Best validation macro-F1: 0.16925.
- Best checkpoint: available.

For a fair comparison of newly trained trials:
- Grid Search best macro-F1: 0.18788.
- Random Search best macro-F1: 0.16925.
- Difference: 0.01863 (1.86 percentage points).

## 4. Hyperparameter Analysis

Based on the 10 newly trained Grid Search trials:

Learning rate:
- 0.001: mean macro-F1 0.16845.
- 0.0003: mean macro-F1 0.12991.
- 0.0001: mean macro-F1 0.08376.

Weight decay:
- 0.0001: mean macro-F1 0.12848.
- 0.0: mean macro-F1 0.10517.

Augmentation:
- Disabled: mean macro-F1 0.12834.
- Enabled: mean macro-F1 0.10998.

These are observed trends within this experiment, not universal conclusions.
The comparisons have limited sample sizes and some unequal group sizes.

## 5. Best Newly Trained Configurations

Grid Search:
- Trial: 3
- Learning rate: 0.001
- Batch size: 8
- Weight decay: 0.0001
- Augmentation: disabled
- Validation macro-F1: 0.18788
- Checkpoint: available

Random Search:
- Trial: 3
- Learning rate: 0.001
- Batch size: 8
- Weight decay: 0.0
- Augmentation: disabled
- Validation macro-F1: 0.16925
- Checkpoint: available

## 6. Experiment Tracking

MLflow experiment: cub200-hyperparameter-tuning
Experiment ID: 4

- 14 newly trained runs recorded.
- All 14 runs have FINISHED status.
- Hyperparameters and final validation metrics logged.
- Best Grid Search checkpoint uploaded as an MLflow artifact.
- Best Random Search checkpoint uploaded as an MLflow artifact.

Per-epoch training and validation curves were not verified
as logged MLflow artifacts.

The two recovered Grid Search trials were reconstructed from
previous logs and were not newly executed MLflow runs.

## 7. Limitations

- Only four Random Search trials were performed.
- The two recovered Grid Search trials have incomplete metrics.
- The recovered best Grid Search trial has no available checkpoint.
- Hyperparameter comparisons have limited sample sizes.
- The best newly trained Grid Search trial reached its best
  validation score at epoch 20, the end of its training budget.
- No conclusion about test-set performance is made here.
- No repeated-seed experiment was conducted to measure variance.

## 8. Recommendation

Use Grid Search Trial 3 as the primary candidate checkpoint
for subsequent evaluation. Retain Random Search Trial 3 as
a comparison checkpoint.

The final model should be evaluated on the held-out test set
only after the model-selection process is complete.
