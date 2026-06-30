# FFT-STGAN-IDS

This repository contains the PyTorch implementation and experiment utilities for an unsupervised spatio-temporal GAN intrusion detection model for IIoT traffic. The current version replaces the original wavelet branch with a learnable FFT-band prior and further introduces a weighted positional encoding strategy during traffic-to-image construction.

## Core Ideas

- Train only on normal traffic and detect abnormal traffic by reconstruction and latent-consistency based anomaly scoring.
- Convert one-dimensional network-flow feature vectors into two-dimensional traffic images through a correlation-guided feature layout.
- Use stacked consecutive flows to preserve short-term temporal context.
- Replace the DWT/wavelet branch with a learnable FFT-band prior to model low-, mid-, high-, and full-frequency responses without requiring an external wavelet package.
- Add weighted sinusoidal positional encoding and an explicit positional channel so that feature-order information is retained after 1D-to-2D mapping.

## Weighted Positional Encoding

During preprocessing, each ordered feature vector is mapped into a fixed image grid. For a vector arranged into an \(H \times W\) grid, a sinusoidal positional code is generated from the beginning to the end of the one-dimensional feature sequence:

```text
p_i = sin(2 * pi * i / (H * W - 1)),  i = 0, 1, ..., H * W - 1
```

The positional code is reshaped into a positional map \(P \in R^{H \times W}\). The traffic image is then enhanced with a weighted positional term:

```text
I_pos = I + omega * P
```

where `omega` is the positional weight. In the latest experiments, `omega = 0.15` is used as the default value. The model input is constructed as a four-channel image:

1. Three traffic channels built from consecutive flow records with the weighted positional term added.
2. One independent positional channel containing the reshaped positional map \(P\).

This design lets the network observe both the value distribution of network-flow features and their relative position in the correlation-guided layout. Compared with directly converting the vector into an image, the positional channel reduces feature-order ambiguity and makes the image representation more informative for convolutional and spatio-temporal modules.

## Model Components

- `SFEM`: multi-scale spatial feature extraction with attention refinement.
- `TFEM`: temporal feature extraction for short-term traffic context.
- `STFFM`: spatio-temporal feature fusion.
- `CFFM`: cross-layer feature fusion to preserve low-level and high-level information.
- `Learnable FFT Band Prior`: frequency-aware branch that replaces the original wavelet prior.
- `Focal reconstruction loss`: emphasizes difficult reconstruction samples.
- `Warmup-cosine learning rate schedule`: used for the latest formal training runs.

## Main Files

- `src/wstgan_fftids/models.py`: proposed model and comparison-model components.
- `src/wstgan_fftids/trainer.py`: training, checkpoint loading, scoring, and evaluation logic.
- `src/wstgan_fftids/preprocess.py`: CSV preprocessing and traffic-image construction.
- `src/wstgan_fftids/data.py`: dataset loading utilities.
- `train.py`: proposed-model training entry point.
- `train_comparisons.py`: comparison-method training entry point.
- `train_ablation.py`: ablation-study entry point.
- `noise_robustness.py`: noisy/corrupted traffic robustness evaluation.
- `parameter_study.py`: internal parameter-study runner.
- `scripts/create_positional4_from_rgb.py`: utility for constructing four-channel positional inputs from existing three-channel traffic images.
- `scripts/evaluate_checkpoint.py`: checkpoint evaluation utility.

## Quick Start

Use the configured conda environment and run the proposed method:

```powershell
& 'D:\new_1\envs\new_conda1\python.exe' .\train.py --dataset all --epochs 30 --batch-size 256 --image-size 16 --device cuda --eval-every 5
```

For the latest high-throughput positional four-channel experiments, the batch size can be increased when GPU memory allows:

```powershell
& 'D:\new_1\envs\new_conda1\python.exe' .\train.py --dataset all --epochs 500 --batch-size 1024 --image-size 16 --device cuda --lr-policy warmup_cosine --eval-every 10
```

## Outputs

Typical outputs are written under `outputs/`:

- `metrics.json`: final metrics.
- `scores.csv`: anomaly scores and labels.
- `loss_curve.png`: training loss visualization.
- `roc_curve.png`: ROC curve.
- `metrics_bar.png`: metric bar chart.
- parameter-study and robustness figures for paper-ready visualization.

Model checkpoints with `.pt` or `.pth` suffixes are ignored by Git and should be managed separately.
