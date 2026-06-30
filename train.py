from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wstgan_fftids.data import load_dataset_specs, make_dataloaders
from wstgan_fftids.trainer import TrainConfig, train_one_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FFT-STGAN-IDS on IDS traffic images.")
    parser.add_argument("--dataset", default="all", choices=["all", "unsw", "cic", "toniot"])
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--augment-noise-std", type=float, default=0.0)
    parser.add_argument("--augment-dropout", type=float, default=0.0)
    parser.add_argument("--augment-scale", type=float, default=0.0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--score-alpha", type=float, default=0.35)
    parser.add_argument("--score-beta", type=float, default=0.35)
    parser.add_argument("--score-gamma", type=float, default=0.20)
    parser.add_argument("--score-delta", type=float, default=0.10)
    parser.add_argument("--lr-policy", default="lambda", choices=["lambda", "cosine", "step", "warmup_cosine"])
    parser.add_argument("--lr-decay-start", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--adv-loss", default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--threshold-objective", default="ba", choices=["ba", "f1", "acc", "f1_acc"])
    parser.add_argument("--selection-metric", default="AUC", choices=["AUC", "F1", "Acc", "BA", "F1_Acc"])
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--ema-start-epoch", type=int, default=1)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--out-root", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = load_dataset_specs(args.config)
    keys = list(specs) if args.dataset == "all" else [args.dataset]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        score_alpha=args.score_alpha,
        score_beta=args.score_beta,
        score_gamma=args.score_gamma,
        score_delta=args.score_delta,
        base_channels=args.base_channels,
        device=args.device,
        seed=args.seed,
        lr_policy=args.lr_policy,
        lr_decay_start=args.lr_decay_start,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        adv_loss=args.adv_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        threshold_objective=args.threshold_objective,
        selection_metric=args.selection_metric,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        ema_start_epoch=args.ema_start_epoch,
        pretrained_path=args.pretrained,
        eval_every=args.eval_every,
    )

    for key in keys:
        spec = specs[key]
        bundle = make_dataloaders(
            spec,
            batch_size=args.batch_size,
            workers=args.workers,
            image_size=args.image_size,
            max_train=args.max_train,
            max_test=args.max_test,
            seed=args.seed,
            cache_images=args.cache_images,
            augment_train=args.augment_train,
            augment_noise_std=args.augment_noise_std,
            augment_dropout=args.augment_dropout,
            augment_scale=args.augment_scale,
        )
        out_dir = Path(args.out_root) / f"{key}_{stamp}"
        print(f"Training {spec.name}: train={bundle.train_size}, test={bundle.test_size}, out={out_dir}")
        train_one_dataset(bundle, cfg, out_dir)


if __name__ == "__main__":
    main()
