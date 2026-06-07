$ErrorActionPreference = "Stop"
$Python = "D:\new_1\envs\new_conda1\python.exe"
& $Python .\scripts\prepare_datasets.py
& $Python .\train_ablation.py --dataset all --variants all --epochs 30 --batch-size 256 --image-size 16 --device cuda --lr-policy lambda --lr-decay-start 15 --eval-every 5 --out-root outputs\ablation_30_bs256
