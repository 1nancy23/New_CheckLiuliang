from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wstgan_fftids.data import load_dataset_specs, make_dataloaders
from wstgan_fftids.metrics import save_json, save_scores
from wstgan_fftids.models import AblationOptions, build_models
from wstgan_fftids.trainer import TrainConfig, choose_device, evaluate, seed_everything
from wstgan_fftids.visualize import save_metrics_bar, save_roc_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved STGAN-IDS checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--score-alpha", type=float, default=0.0)
    parser.add_argument("--score-beta", type=float, default=0.8)
    parser.add_argument("--score-gamma", type=float, default=0.2)
    parser.add_argument("--score-delta", type=float, default=0.0)
    parser.add_argument("--threshold-objective", default="acc", choices=["ba", "f1", "acc", "f1_acc"])
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    spec = load_dataset_specs(args.config)[args.dataset]
    bundle = make_dataloaders(
        spec,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        max_train=1,
        max_test=args.max_test,
        seed=args.seed,
        cache_images=False,
    )
    device = choose_device(args.device)
    models = build_models(
        in_channels=bundle.input_channels,
        base_channels=args.base_channels,
        device=device,
        ablation=AblationOptions(),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    models.generator.load_state_dict(checkpoint["generator"], strict=True)
    models.discriminator.load_state_dict(checkpoint["discriminator"], strict=True)

    cfg = TrainConfig(
        device=args.device,
        base_channels=args.base_channels,
        score_alpha=args.score_alpha,
        score_beta=args.score_beta,
        score_gamma=args.score_gamma,
        score_delta=args.score_delta,
        threshold_objective=args.threshold_objective,
        selection_metric="Acc",
    )
    metrics, rows, roc_data = evaluate(models, bundle, cfg, device)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["dataset"] = args.dataset
    metrics["test_size"] = float(bundle.test_size)
    metrics["input_channels"] = float(bundle.input_channels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics.json", metrics)
    save_scores(out_dir / "scores.csv", rows)
    fpr, tpr, roc_auc = roc_data
    save_roc_plot(out_dir / "roc_curve.png", fpr, tpr, roc_auc)
    save_metrics_bar(out_dir / "metrics_bar.png", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
