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
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-policy", default="lambda", choices=["lambda", "cosine", "step"])
    parser.add_argument("--lr-decay-start", type=int, default=15)
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
        base_channels=args.base_channels,
        device=args.device,
        seed=args.seed,
        lr_policy=args.lr_policy,
        lr_decay_start=args.lr_decay_start,
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
        )
        out_dir = Path(args.out_root) / f"{key}_{stamp}"
        print(f"Training {spec.name}: train={bundle.train_size}, test={bundle.test_size}, out={out_dir}")
        train_one_dataset(bundle, cfg, out_dir)


if __name__ == "__main__":
    main()
