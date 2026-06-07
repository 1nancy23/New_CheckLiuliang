set -e
cd /root/New_CheckLiuliang_remote
PYTHON=/data/miniconda/envs/torch/bin/python
RUN_ROOT=outputs/ablation_30_remote_fast
RUN_ID=20260606_214514
PYTHONUNBUFFERED=1 $PYTHON train_ablation.py --config configs/datasets_remote.json --dataset all --variants all --epochs 30 --batch-size 4096 --workers 8 --image-size 16 --device cuda --lr-policy lambda --lr-decay-start 15 --eval-every 5 --out-root $RUN_ROOT --run-id $RUN_ID --skip-existing --clean-incomplete --rebuild-summary
