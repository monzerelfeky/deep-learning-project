# Scratch CNN — Architecture and Experimental Results

**Project:** From Pixels to Language
**Dataset:** CUB-200-2011
**Task:** Fine-grained bird species classification (200 classes)
**Contribution:** Member 3 — CNN From Scratch

## 1. Objective

The objective was to design, implement, and train a convolutional neural network from scratch, without using pretrained weights, to classify bird images into 200 species.

Two custom architectures were implemented and evaluated to establish a baseline and investigate whether a deeper residual architecture could improve classification performance.

## 2. CNN V1 — Baseline Architecture

The first architecture consists of four convolutional blocks, with channel dimensions of 32, 64, 128, and 256.

Each block contains:
- A 3×3 convolution
- Batch normalization
- ReLU activation
- 2×2 max pooling

Global average pooling converts the final spatial feature maps into a 256-dimensional feature vector, followed by a linear classifier producing 200 logits.

**Parameters:** 440,776
**Optimizer:** Adam
**Learning rate:** 0.001
**Batch size:** 4
**Training epochs:** 20

## 3. CNN V2 — Residual Architecture

The second architecture introduces residual connections to improve feature propagation through the network.

It contains an initial convolutional stem followed by four residual stages with channel dimensions of 32, 64, 128, and 256.

Residual blocks contain two 3×3 convolutions, batch normalization, ReLU activations, and shortcut connections. Projection shortcuts are used where channel dimensions or spatial resolution change.

**Parameters:** 1,278,888
**Optimizer:** AdamW
**Initial learning rate:** 0.001
**Weight decay:** 0.0001
**Scheduler:** ReduceLROnPlateau
**Batch size:** 4
**Training epochs:** 20

## 4. Experimental Results

| Metric | CNN V1 | CNN V2 |
|---|---:|---:|
| Parameters | 440,776 | 1,278,888 |
| Best validation accuracy | 10.87% | 16.05% |
| Best validation macro-F1 | 0.0752 | 0.1350 |
| Best epoch | 20 | 19 |

CNN V2 achieved a 5.18 percentage-point improvement in validation accuracy and a higher macro-F1 score compared with CNN V1.

The improvement cannot be attributed exclusively to residual connections because the optimizer and learning-rate scheduling also changed.

## 5. Training Analysis

Both models improved over the training period, demonstrating that the implementations learned useful classification features.

CNN V2 achieved better classification performance, although validation loss was unstable and some incorrect predictions were highly confident.

The dataset presents a challenging fine-grained classification problem because many bird species share similar visual characteristics.

The results are therefore reported as experimental baselines rather than as an optimized final classification system.

## 6. Encoder Reusability

Both architectures implement the shared BaseEncoder interface.

CNN V2 provides:
- Spatial feature maps: `[B, 256, 14, 14]`
- Pooled feature vectors: `[B, 256]`
- Classifier logits: `[B, 200]`
- Grad-CAM target layer: `stage4.conv2`

The best CNN V2 checkpoint was successfully loaded and tested for output compatibility.

## 7. Selected Scratch Model

CNN V2 was selected as the stronger scratch CNN baseline because it achieved higher validation accuracy and macro-F1.

The CNN V1 implementation and experiment results are retained for comparison.

## 8. Experiment Artifacts

**Models**
- `src/models/scratch_cnn.py`
- `src/models/scratch_cnn_v2.py`

**Training**
- `src/training/train_scratch.py`
- `src/training/train_scratch_v2.py`

**Configuration**
- `configs/scratch.yaml`

**Best checkpoints**
- `checkpoints/scratch_cnn/experiment_01/best_model.pt`
- `checkpoints/scratch_cnn/experiment_02/best_model.pt`

**Training histories**
- `outputs/training/scratch_cnn_experiment_01.csv`
- `outputs/training/scratch_cnn_experiment_02.csv`

**Visualizations**
- `outputs/figures/scratch_train_accuracy.png`
- `outputs/figures/scratch_val_accuracy.png`
- `outputs/figures/scratch_train_loss.png`
- `outputs/figures/scratch_val_loss.png`
- `outputs/figures/scratch_val_f1.png`

## 9. Conclusion

The contribution successfully implements and evaluates two custom CNN architectures trained entirely from scratch.

CNN V2 provides the stronger baseline and a reusable visual encoder for subsequent project components.

Further hyperparameter optimization, transfer learning, Grad-CAM evaluation, and external benchmarking are handled by the respective team members.