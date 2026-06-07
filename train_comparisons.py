from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wstgan_fftids.comparison_trainer import METHODS, METHOD_NAMES, ComparisonConfig, train_comparison_method
from wstgan_fftids.data import load_dataset_specs, make_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper comparison baselines for FFT-STGAN-IDS.")
    parser.add_argument("--dataset", default="all", choices=["all", "unsw", "cic", "toniot"])
    parser.add_argument("--methods", default="all", help="Comma-separated list: if,vae,f-anogan,bigan,mts-dvgan or all.")
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-decay-start", type=int, default=15)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--out-root", default="outputs/comparison_30_bs256")
    return parser.parse_args()


def append_summary(path: Path, row: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    specs = load_dataset_specs(args.config)
    datasets = list(specs) if args.dataset == "all" else [args.dataset]
    methods = list(METHODS) if args.methods == "all" else [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    cfg = ComparisonConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        device=args.device,
        seed=args.seed,
        lr_decay_start=args.lr_decay_start,
        eval_every=args.eval_every,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / stamp
    summary_path = out_root / "comparison_summary.csv"

    for dataset_key in datasets:
        spec = specs[dataset_key]
        bundle = make_dataloaders(
            spec,
            batch_size=args.batch_size,
            workers=args.workers,
            image_size=args.image_size,
            seed=args.seed,
            cache_images=args.cache_images,
        )
        for method in methods:
            method_name = METHOD_NAMES[method]
            out_dir = out_root / dataset_key / method
            print(f"Training comparison {method_name} on {spec.name}: train={bundle.train_size}, test={bundle.test_size}, out={out_dir}")
            metrics = train_comparison_method(method, bundle, cfg, out_dir)
            append_summary(
                summary_path,
                {
                    "dataset": dataset_key,
                    "name": spec.name,
                    "method": method_name,
                    "out_dir": str(out_dir),
                    "epoch": int(metrics["epoch"]),
                    "Threshold": metrics["Threshold"],
                    "Acc": metrics["Acc"],
                    "Prec": metrics["Prec"],
                    "Rec": metrics["Rec"],
                    "FAR": metrics["FAR"],
                    "F1": metrics["F1"],
                    "AUC": metrics["AUC"],
                    "seconds": metrics["seconds"],
                    "train_size": int(metrics["train_size"]),
                    "test_size": int(metrics["test_size"]),
                },
            )
    print(f"Wrote comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
