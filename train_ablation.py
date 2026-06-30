from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wstgan_fftids.data import load_dataset_specs, make_dataloaders
from wstgan_fftids.metrics import save_json
from wstgan_fftids.trainer import TrainConfig, train_one_dataset


VARIANT_NOTES = {
    "full": "Complete FFT-STGAN-IDS model.",
    "baseline_gan": "Paper-style baseline: remove spatio-temporal fusion, FFT prior, and cross-layer fusion.",
    "no_fft_prior": "Remove the learnable FFT band prior branch and its frequency score/loss.",
    "no_temporal_gru": "Remove the GRU temporal branch inside spatio-temporal fusion.",
    "no_st_fusion": "Bypass the spatio-temporal/frequency fusion block after each encoder stage.",
    "no_cffm": "Remove decoder cross-layer feature fusion skip aggregation.",
    "no_freq_loss": "Keep FFT branch and score, but remove frequency consistency loss.",
    "no_latent_loss": "Remove discriminator feature matching loss and latent score contribution.",
    "no_adv_loss": "Remove generator adversarial objective and discriminator score contribution.",
    "rec_only_score": "Train the full model but use reconstruction error only for anomaly scoring.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ablation variants for the proposed FFT-STGAN-IDS method.")
    parser.add_argument("--dataset", default="all", choices=["all", "unsw", "cic", "toniot"])
    parser.add_argument("--variants", default="all", help="Comma-separated variant names or all.")
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
    parser.add_argument("--lr-policy", default="warmup_cosine", choices=["lambda", "cosine", "step", "warmup_cosine"])
    parser.add_argument("--lr-decay-start", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.02)
    parser.add_argument("--adv-loss", default="focal", choices=["bce", "focal"])
    parser.add_argument("--focal-alpha", type=float, default=0.35)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--threshold-objective", default="acc", choices=["ba", "f1", "acc", "f1_acc"])
    parser.add_argument("--selection-metric", default="Acc", choices=["AUC", "F1", "Acc", "BA", "F1_Acc"])
    parser.add_argument("--score-alpha", type=float, default=0.0)
    parser.add_argument("--score-beta", type=float, default=0.8)
    parser.add_argument("--score-gamma", type=float, default=0.2)
    parser.add_argument("--score-delta", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-start-epoch", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--out-root", default="outputs/ablation_30")
    parser.add_argument("--pretrained-root", default="", help="Root containing dataset/best_model.pt checkpoints for non-strict ablation initialization.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--clean-incomplete", action="store_true")
    parser.add_argument("--rebuild-summary", action="store_true")
    parser.add_argument("--run-id", default=None, help="Use a fixed run directory name instead of a new timestamp.")
    return parser.parse_args()


def pretrained_for_dataset(root: str, dataset_key: str) -> str:
    if not root:
        return ""
    path = Path(root) / dataset_key / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing pretrained checkpoint for {dataset_key}: {path}")
    return str(path)


def variant_config(name: str, base: TrainConfig) -> TrainConfig:
    cfg = replace(base, ablation_name=name)
    if name == "full":
        return cfg
    if name == "baseline_gan":
        return replace(
            cfg,
            use_fft_prior=False,
            use_temporal=False,
            use_st_fusion=False,
            use_cffm=False,
            w_freq=0.0,
            score_delta=0.0,
        )
    if name == "no_fft_prior":
        return replace(cfg, use_fft_prior=False, w_freq=0.0, score_delta=0.0)
    if name == "no_temporal_gru":
        return replace(cfg, use_temporal=False)
    if name == "no_st_fusion":
        return replace(cfg, use_st_fusion=False)
    if name == "no_cffm":
        return replace(cfg, use_cffm=False)
    if name == "no_freq_loss":
        return replace(cfg, w_freq=0.0)
    if name == "no_latent_loss":
        return replace(cfg, w_lat=0.0, score_gamma=0.0)
    if name == "no_adv_loss":
        return replace(cfg, w_adv=0.0, score_alpha=0.0)
    if name == "rec_only_score":
        return replace(cfg, score_alpha=0.0, score_beta=1.0, score_gamma=0.0, score_delta=0.0)
    raise ValueError(f"Unknown ablation variant: {name}")


def append_summary(path: Path, row: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_completed_metrics(path: Path) -> dict[str, float] | None:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return None
    import json

    return json.loads(metrics_path.read_text(encoding="utf-8"))


def make_summary_row(dataset_key: str, name: str, variant: str, out_dir: Path, metrics: dict) -> dict[str, str | int | float]:
    return {
        "dataset": dataset_key,
        "name": name,
        "variant": variant,
        "note": VARIANT_NOTES[variant],
        "out_dir": str(out_dir),
        "epoch": int(metrics["epoch"]),
        "Threshold": float(metrics["Threshold"]),
        "Acc": float(metrics["Acc"]),
        "Prec": float(metrics["Prec"]),
        "Rec": float(metrics["Rec"]),
        "FAR": float(metrics["FAR"]),
        "F1": float(metrics["F1"]),
        "AUC": float(metrics["AUC"]),
        "seconds": float(metrics["seconds"]),
        "train_size": int(metrics["train_size"]),
        "test_size": int(metrics["test_size"]),
    }


def save_ablation_plot(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1180
    row_h = 28
    header_h = 56
    height = header_h + row_h * len(rows) + 36
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "Ablation Summary: AUC and F1", fill=(20, 30, 45))
    draw.text((24, 42), "dataset / variant", fill=(70, 80, 95))
    draw.text((340, 42), "AUC", fill=(70, 80, 95))
    draw.text((735, 42), "F1", fill=(70, 80, 95))
    auc_color = (37, 99, 235)
    f1_color = (5, 150, 105)
    y = header_h
    for row in rows:
        label = f"{row['dataset']} / {row['variant']}"
        auc = float(row["AUC"])
        f1 = float(row["F1"])
        draw.text((24, y + 6), label[:42], fill=(20, 30, 45))
        draw.rectangle((340, y + 6, 700, y + 20), outline=(210, 215, 225))
        draw.rectangle((735, y + 6, 1095, y + 20), outline=(210, 215, 225))
        draw.rectangle((340, y + 6, 340 + int(360 * max(0.0, min(1.0, auc))), y + 20), fill=auc_color)
        draw.rectangle((735, y + 6, 735 + int(360 * max(0.0, min(1.0, f1))), y + 20), fill=f1_color)
        draw.text((706, y + 4), f"{auc:.4f}", fill=(20, 30, 45))
        draw.text((1100, y + 4), f"{f1:.4f}", fill=(20, 30, 45))
        y += row_h
    image.save(path)


def main() -> None:
    args = parse_args()
    specs = load_dataset_specs(args.config)
    datasets = list(specs) if args.dataset == "all" else [args.dataset]
    variants = list(VARIANT_NOTES) if args.variants == "all" else [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [name for name in variants if name not in VARIANT_NOTES]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    stamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / stamp
    summary_path = out_root / "ablation_summary.csv"
    if args.rebuild_summary and summary_path.exists():
        summary_path.unlink()
    save_json(out_root / "ablation_manifest.json", VARIANT_NOTES)

    base_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        base_channels=args.base_channels,
        device=args.device,
        seed=args.seed,
        lr_policy=args.lr_policy,
        lr_decay_start=args.lr_decay_start,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        eval_every=args.eval_every,
        adv_loss=args.adv_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        threshold_objective=args.threshold_objective,
        selection_metric=args.selection_metric,
        score_alpha=args.score_alpha,
        score_beta=args.score_beta,
        score_gamma=args.score_gamma,
        score_delta=args.score_delta,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        ema_start_epoch=args.ema_start_epoch,
        summary_path="",
    )
    summary_rows: list[dict[str, str | int | float]] = []

    for dataset_key in datasets:
        spec = specs[dataset_key]
        completed = {
            variant: load_completed_metrics(out_root / dataset_key / variant) if args.skip_existing else None
            for variant in variants
        }
        if args.skip_existing and all(metrics is not None for metrics in completed.values()):
            print(f"Skipping completed dataset {spec.name}: all requested ablations are present.")
            for variant in variants:
                out_dir = out_root / dataset_key / variant
                row = make_summary_row(dataset_key, spec.name, variant, out_dir, completed[variant])
                append_summary(summary_path, row)
                summary_rows.append(row)
            continue

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
        for variant in variants:
            out_dir = out_root / dataset_key / variant
            metrics = completed[variant]
            if metrics is None:
                if args.clean_incomplete and out_dir.exists():
                    shutil.rmtree(out_dir)
                cfg = variant_config(variant, base_cfg)
                pretrained_path = pretrained_for_dataset(args.pretrained_root, dataset_key)
                if pretrained_path:
                    cfg = replace(cfg, pretrained_path=pretrained_path, pretrained_strict=False)
                print(
                    f"Training ablation {variant} on {spec.name}: "
                    f"train={bundle.train_size}, test={bundle.test_size}, out={out_dir}"
                )
                metrics = train_one_dataset(bundle, cfg, out_dir)
            else:
                print(f"Skipping completed ablation {variant} on {spec.name}: out={out_dir}")
            row = make_summary_row(dataset_key, spec.name, variant, out_dir, metrics)
            append_summary(summary_path, row)
            summary_rows.append(row)

    save_ablation_plot(out_root / "ablation_auc_f1.png", summary_rows)
    print(f"Wrote ablation summary: {summary_path}")


if __name__ == "__main__":
    main()
