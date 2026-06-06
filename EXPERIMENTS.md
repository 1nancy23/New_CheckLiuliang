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

