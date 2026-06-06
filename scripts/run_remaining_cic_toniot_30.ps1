$ErrorActionPreference = "Stop"
$Python = "D:\new_1\envs\new_conda1\python.exe"
& $Python .\scripts\prepare_datasets.py
& $Python .\train.py --dataset cic --epochs 30 --batch-size 256 --image-size 16 --device cuda --lr-policy lambda --lr-decay-start 15 --eval-every 5 --out-root outputs\formal_30_bs256_remaining
& $Python .\train.py --dataset toniot --epochs 30 --batch-size 256 --image-size 16 --device cuda --lr-policy lambda --lr-decay-start 15 --eval-every 5 --out-root outputs\formal_30_bs256_remaining
