# Experiment Log

## Environment

- Python: `D:\new_1\envs\new_conda1\python.exe`
- PyTorch: `1.12.0+cu113`
- CUDA: available
- No `pip install` or `pip uninstall` was used.
- Visualization uses PIL because `matplotlib` fails to import in this environment due to a DLL error.

## Dataset Registry

| Dataset | Train normal | Train abnormal | Test normal | Test abnormal |
|---|---:|---:|---:|---:|
| UNSW-NB15 | 18,666 | 0 | 12,332 | 15,112 |
| CIC-IDS2017 | 102,592 | 0 | 37,600 | 90,276 |
| TON_IoT | 5,581 | 0 | 2,391 | 3,296 |

Dataset manifest:

- `data/datasets_manifest.json`

## Current Training Setup

The formal default has been changed to 30 epochs:

```powershell
& 'D:\new_1\envs\new_conda1\python.exe' .\train.py --dataset all --epochs 30 --batch-size 256 --image-size 16 --device cuda --lr-policy lambda --lr-decay-start 15 --eval-every 5
```

Learning rate schedule:

- Epoch 1-15: keep initial LR.
- Epoch 16-30: linearly decay LR to 0.

Remaining-datasets script:

```powershell
.\scripts\run_remaining_cic_toniot_30.ps1
```

## Completed Results

| Dataset | Run | Best epoch | Acc | Prec | Rec | FAR | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| UNSW-NB15 | `formal_160_bs256_20260606_153246` | 45 | 0.9027 | 0.9087 | 0.9152 | 0.1127 | 0.9119 | 0.9560 |
| CIC-IDS2017 | `formal_30_bs256_remaining` | 1 | 0.6850 | 0.8998 | 0.6233 | 0.1667 | 0.7364 | 0.7975 |
| TON_IoT | `formal_30_bs256_remaining` | 30 | 0.9237 | 0.8859 | 0.9967 | 0.1769 | 0.9380 | 0.9324 |

Output directories:

- `outputs/formal_160_bs256_20260606_153246/unsw_20260606_153249`
- `outputs/formal_30_bs256_remaining/cic_20260606_163529`
- `outputs/formal_30_bs256_remaining/toniot_20260606_174051`

Each completed dataset directory contains:

- `best_model.pt`
- `latest_model.pt`
- `training_history.csv`
- `metrics.json`
- `scores.csv`
- `loss_curve.png`
- `roc_curve.png`
- `metrics_bar.png`
- `reconstruction_grid.png`

Combined summary:

- `outputs/summary_metrics.csv`

## Notes

- UNSW was already completed with the earlier 160-epoch run before the epoch count was reduced.
- CIC and TON_IoT were completed after changing the formal setting to 30 epochs.
- CIC shows GAN training instability after early epochs; the best checkpoint was automatically preserved at epoch 1 by AUC.
- TON_IoT improved through the 30th epoch, with the best checkpoint at epoch 30.

## Comparison Experiments

Paper comparison methods:

- IF
- VAE
- f-AnoGAN
- BiGAN
- MTS-DVGAN

Run command:

```powershell
.\scripts\run_comparison_experiments.ps1
```

All PyTorch comparison models use 30 epochs, batch size 256, and the same normal-only training / mixed testing protocol as the main experiment. IF is a one-pass non-epoch baseline.

The unfinished comparison jobs were completed on the remote Tesla P4 server with the preconfigured PyTorch environment:

- Python: `/data/miniconda/envs/torch/bin/python`
- PyTorch: `1.13.0+cu117`
- GPU: Tesla P4
- Remote accelerated settings: batch size 1024, dataloader workers 4

Unified comparison summary:

- `outputs/comparison_30_all_summary.csv`

| Dataset | Method | Best epoch | Acc | Prec | Rec | FAR | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| UNSW-NB15 | IF | 0 | 0.8225 | 0.8268 | 0.8574 | 0.2202 | 0.8418 | 0.8928 |
| UNSW-NB15 | VAE | 10 | 0.8471 | 0.8453 | 0.8841 | 0.1983 | 0.8642 | 0.9099 |
| UNSW-NB15 | f-AnoGAN | 30 | 0.8422 | 0.8844 | 0.8208 | 0.1315 | 0.8514 | 0.9186 |
| UNSW-NB15 | BiGAN | 5 | 0.7868 | 0.7784 | 0.8565 | 0.2987 | 0.8156 | 0.8227 |
| UNSW-NB15 | MTS-DVGAN | 10 | 0.7640 | 0.7738 | 0.8074 | 0.2892 | 0.7902 | 0.8290 |
| CIC-IDS2017 | IF | 0 | 0.8548 | 0.8698 | 0.9341 | 0.3356 | 0.9008 | 0.8939 |
| CIC-IDS2017 | VAE | 5 | 0.6787 | 0.9794 | 0.5566 | 0.0282 | 0.7098 | 0.8510 |
| CIC-IDS2017 | f-AnoGAN | 30 | 0.7465 | 0.9790 | 0.6550 | 0.0338 | 0.7848 | 0.8556 |
| CIC-IDS2017 | BiGAN | 1 | 0.8286 | 0.9267 | 0.8223 | 0.1561 | 0.8714 | 0.8958 |
| CIC-IDS2017 | MTS-DVGAN | 5 | 0.8223 | 0.8767 | 0.8708 | 0.2939 | 0.8737 | 0.8887 |
| TON_IoT | IF | 0 | 0.9455 | 0.9343 | 0.9745 | 0.0945 | 0.9540 | 0.9523 |
| TON_IoT | VAE | 30 | 0.7900 | 0.8295 | 0.8028 | 0.2275 | 0.8159 | 0.8537 |
| TON_IoT | f-AnoGAN | 15 | 0.9182 | 0.8837 | 0.9891 | 0.1794 | 0.9334 | 0.9080 |
| TON_IoT | BiGAN | 30 | 0.6624 | 0.8324 | 0.5228 | 0.1451 | 0.6422 | 0.5247 |
| TON_IoT | MTS-DVGAN | 30 | 0.8233 | 0.8479 | 0.8471 | 0.2095 | 0.8475 | 0.8578 |

## Ablation Experiments

The ablation runner follows the paper's Baseline/Proposed setting and expands it into component-level variants for the new FFT-based method:

- `full`
- `baseline_gan`
- `no_fft_prior`
- `no_temporal_gru`
- `no_st_fusion`
- `no_cffm`
- `no_freq_loss`
- `no_latent_loss`
- `no_adv_loss`
- `rec_only_score`

Run command:

```powershell
.\scripts\run_ablation_experiments.ps1
```

All variants use 30 epochs, the lambda learning-rate decay schedule, normal-only training, and mixed normal/abnormal testing. The remote Tesla P4 run uses the same configured PyTorch environment and may raise batch size/workers to improve GPU utilization.
