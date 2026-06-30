from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from noise_robustness import (
    CORRUPTIONS,
    METHOD_LABELS,
    Corruption,
    EvalResult,
    apply_noise,
    draw_radar_chart,
    make_noisy_bundle,
    metrics_at_threshold,
    radar_values,
    rank_correlation,
    result_from_rows,
    write_validation_summary,
)
from wstgan_fftids.comparison_models import BiGANModel, FAnoGANModel, MTSDVGANModel, VAEModel
from wstgan_fftids.comparison_trainer import (
    _eval_bigan,
    _eval_fanogan,
    _eval_mtsdvgan,
    _eval_vae,
    train_comparison_method,
)
from wstgan_fftids.data import DataBundle, load_dataset_specs, make_dataloaders
from wstgan_fftids.metrics import best_balanced_accuracy, minmax
from wstgan_fftids.models import AblationOptions, build_models
from wstgan_fftids.trainer import TrainConfig, choose_device, evaluate, seed_everything, train_one_dataset


TRAIN_CORRUPTIONS = (
    Corruption("clean", "Clean", "none", 0.0),
    Corruption("train_gaussian_003", "Train Gaussian Noise", "gaussian", 0.03),
    Corruption("train_gaussian_006", "Train Strong Gaussian", "gaussian", 0.06),
    Corruption("train_speckle_004", "Train Speckle Noise", "speckle", 0.04),
    Corruption("train_saltpepper_002", "Train Salt-and-Pepper", "saltpepper", 0.02),
    Corruption("train_dropout_003", "Train Random Dropout", "dropout", 0.03),
)


class RandomNoiseAugmentedDataset(Dataset):
    def __init__(self, base: Dataset, corruptions: tuple[Corruption, ...], seed: int) -> None:
        self.base = base
        self.corruptions = corruptions
        self.seed = seed
        self.counter = 0

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label, path = self.base[index]
        self.counter += 1
        selector_seed = self.seed + index * 9176 + self.counter * 131
        selector = torch.Generator().manual_seed(selector_seed)
        corruption_index = int(torch.randint(len(self.corruptions), (1,), generator=selector).item())
        corruption = self.corruptions[corruption_index]
        return apply_noise(image, corruption, selector_seed + 17), label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train noise-augmented IDS models and evaluate them under noisy test signals.")
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--dataset", default="all", choices=["all", "unsw", "cic", "toniot"])
    parser.add_argument("--methods", default="all", help="Comma list or all: proposed,if,vae,f-anogan,bigan,mts-dvgan")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-decay-start", type=int, default=15)
    parser.add_argument("--lr-policy", default="warmup_cosine", choices=["lambda", "cosine", "step", "warmup_cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.02)
    parser.add_argument("--base-channels", type=int, default=48)
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
    parser.add_argument("--out-root", default="outputs/noise_augmented_training")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def make_augmented_train_bundle(bundle: DataBundle, args: argparse.Namespace) -> DataBundle:
    train_ds = RandomNoiseAugmentedDataset(bundle.train.dataset, TRAIN_CORRUPTIONS, args.seed)
    loader_kwargs = {}
    if args.workers > 0:
        loader_kwargs = {"persistent_workers": True, "prefetch_factor": 4}
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True if len(train_ds) >= args.batch_size else False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )
    return DataBundle(train_loader, bundle.test, bundle.spec, bundle.train_size, bundle.test_size)


def collect_flat(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, paths = [], [], []
    for image, label, path in loader:
        xs.append(image.flatten(1).numpy())
        ys.append(label.numpy())
        paths.extend(path)
    return np.concatenate(xs), np.concatenate(ys), paths


def fit_noise_augmented_if(bundle: DataBundle, seed: int) -> IsolationForest:
    x_train, _, _ = collect_flat(bundle.train)
    model = IsolationForest(
        n_estimators=100,
        max_samples=min(10000, len(x_train)),
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train)
    return model


def evaluate_if(model: IsolationForest, bundle: DataBundle) -> EvalResult:
    x_test, labels, paths = collect_flat(bundle.test)
    scores = minmax(-model.decision_function(x_test))
    metrics = best_balanced_accuracy(labels, scores)
    return EvalResult(metrics, labels.astype(np.int64), scores.astype(np.float64), paths)


def evaluate_proposed_checkpoint(checkpoint: Path, bundle: DataBundle, device: torch.device, args: argparse.Namespace) -> EvalResult:
    model_bundle = build_models(base_channels=args.base_channels, input_channels=bundle.input_channels, device=device, ablation=AblationOptions())
    ckpt = torch.load(checkpoint, map_location=device)
    model_bundle.generator.load_state_dict(ckpt["generator"])
    model_bundle.discriminator.load_state_dict(ckpt["discriminator"])
    metrics, rows, _ = evaluate(
        model_bundle,
        bundle,
        TrainConfig(
            device=str(device),
            score_alpha=args.score_alpha,
            score_beta=args.score_beta,
            score_gamma=args.score_gamma,
            score_delta=args.score_delta,
            threshold_objective=args.threshold_objective,
        ),
        device,
    )
    return result_from_rows(metrics, rows)


def evaluate_comparison_checkpoint(method: str, checkpoint: Path, bundle: DataBundle, device: torch.device, args: argparse.Namespace) -> EvalResult:
    ckpt = torch.load(checkpoint, map_location=device)
    if method == "vae":
        model = VAEModel(base_channels=args.base_channels, in_channels=bundle.input_channels).to(device)
        model.load_state_dict(ckpt["model"])
        metrics, rows, _ = _eval_vae(model, bundle, device)
    elif method == "f-anogan":
        model = FAnoGANModel(base_channels=args.base_channels, in_channels=bundle.input_channels).to(device)
        model.load_state_dict(ckpt["model"])
        metrics, rows, _ = _eval_fanogan(model, bundle, device)
    elif method == "bigan":
        model = BiGANModel(base_channels=args.base_channels, in_channels=bundle.input_channels).to(device)
        model.load_state_dict(ckpt["model"])
        metrics, rows, _ = _eval_bigan(model, bundle, device)
    elif method == "mts-dvgan":
        model = MTSDVGANModel(base_channels=args.base_channels, in_channels=bundle.input_channels).to(device)
        model.load_state_dict(ckpt["model"])
        metrics, rows, _ = _eval_mtsdvgan(model, bundle, device)
    else:
        raise ValueError(f"Unsupported method checkpoint: {method}")
    return result_from_rows(metrics, rows)


def append_csv(path: Path, row: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing_rows(path: Path) -> list[dict[str, str | float]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_trained_method(
    dataset_key: str,
    method: str,
    model_ref,
    base_bundle: DataBundle,
    args: argparse.Namespace,
    out_root: Path,
    device: torch.device,
) -> list[dict[str, str | float]]:
    summary_path = out_root / "noise_augmented_test_summary.csv"
    clean_result: EvalResult | None = None
    rows: list[dict[str, str | float]] = []
    for corruption in [Corruption(*item) for item in CORRUPTIONS]:
        noisy_bundle = make_noisy_bundle(base_bundle, corruption, args)
        torch.manual_seed(args.seed)
        if method == "proposed":
            result = evaluate_proposed_checkpoint(model_ref, noisy_bundle, device, args)
        elif method == "if":
            result = evaluate_if(model_ref, noisy_bundle)
        else:
            result = evaluate_comparison_checkpoint(method, model_ref, noisy_bundle, device, args)

        if corruption.key == "clean":
            clean_result = result
            transfer = metrics_at_threshold(result.labels, result.scores, result.metrics["Threshold"])
            rank_corr = 1.0
        else:
            assert clean_result is not None
            transfer = metrics_at_threshold(result.labels, result.scores, clean_result.metrics["Threshold"])
            rank_corr = rank_correlation(clean_result.scores, result.scores)

        row = {
            "dataset": dataset_key,
            "dataset_name": base_bundle.spec.name,
            "method_key": method,
            "method": METHOD_LABELS[method],
            "training": "Noise-Augmented",
            "corruption_key": corruption.key,
            "corruption": corruption.label,
            "severity": corruption.severity,
            "Threshold": result.metrics["Threshold"],
            "Acc": result.metrics["Acc"],
            "Prec": result.metrics["Prec"],
            "Rec": result.metrics["Rec"],
            "FAR": result.metrics["FAR"],
            "F1": result.metrics["F1"],
            "AUC": result.metrics["AUC"],
            "RankCorrelation": rank_corr,
            **transfer,
            "test_size": len(result.labels),
        }
        append_csv(summary_path, row)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    specs = load_dataset_specs(args.config)
    datasets = list(specs) if args.dataset == "all" else [args.dataset]
    methods = list(METHOD_LABELS) if args.methods == "all" else [m.strip() for m in args.methods.split(",") if m.strip()]
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    train_summary_path = out_root / "noise_augmented_training_summary.csv"
    eval_summary_path = out_root / "noise_augmented_test_summary.csv"
    all_eval_rows: list[dict[str, str | float]] = load_existing_rows(eval_summary_path) if args.skip_existing else []

    for dataset_key in datasets:
        spec = specs[dataset_key]
        print(f"Loading {spec.name} for noise-augmented training.")
        base_bundle = make_dataloaders(
            spec,
            batch_size=args.batch_size,
            workers=args.workers,
            image_size=args.image_size,
            max_train=args.max_train,
            max_test=args.max_test,
            seed=args.seed,
            cache_images=args.cache_images,
        )
        train_bundle = make_augmented_train_bundle(base_bundle, args)
        for method in methods:
            method_dir = out_root / "models" / dataset_key / method
            if method == "if":
                print(f"Training noise-augmented IF on {spec.name}.")
                start = time.time()
                model_ref = fit_noise_augmented_if(train_bundle, args.seed)
                append_csv(
                    train_summary_path,
                    {
                        "dataset": dataset_key,
                        "dataset_name": spec.name,
                        "method": METHOD_LABELS[method],
                        "out_dir": str(method_dir),
                        "epoch": 0,
                        "Threshold": "",
                        "Acc": "",
                        "Prec": "",
                        "Rec": "",
                        "FAR": "",
                        "F1": "",
                        "AUC": "",
                        "seconds": time.time() - start,
                        "train_size": train_bundle.train_size,
                        "test_size": train_bundle.test_size,
                    },
                )
            else:
                if args.skip_existing and (method_dir / "best_model.pt").exists():
                    print(f"Skipping existing trained model: {method_dir}")
                else:
                    print(f"Training noise-augmented {METHOD_LABELS[method]} on {spec.name}: out={method_dir}")
                    if method == "proposed":
                        cfg = TrainConfig(
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
                        metrics = train_one_dataset(train_bundle, cfg, method_dir)
                    else:
                        from wstgan_fftids.comparison_trainer import ComparisonConfig

                        cfg = ComparisonConfig(
                            epochs=args.epochs,
                            lr=args.lr,
                            batch_size=args.batch_size,
                            base_channels=args.base_channels,
                            device=args.device,
                            seed=args.seed,
                            lr_decay_start=args.lr_decay_start,
                            eval_every=args.eval_every,
                            input_channels=train_bundle.input_channels,
                        )
                        metrics = train_comparison_method(method, train_bundle, cfg, method_dir)
                    append_csv(
                        train_summary_path,
                        {
                            "dataset": dataset_key,
                            "dataset_name": spec.name,
                            "method": METHOD_LABELS[method],
                            "out_dir": str(method_dir),
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
                model_ref = method_dir / "best_model.pt"
            eval_rows = evaluate_trained_method(dataset_key, method, model_ref, base_bundle, args, out_root, device)
            all_eval_rows.extend(eval_rows)

    validation_rows = write_validation_summary(out_root / "noise_augmented_validation_summary.csv", all_eval_rows)
    axes = ["Clean AUC", "Mean Noisy AUC", "Worst Noisy AUC", "Mean Noisy F1", "Transfer F1", "Rank Stability"]
    draw_radar_chart(out_root / "radar_noise_augmented_overall.png", "Noise-Augmented Training Robustness", axes, radar_values(all_eval_rows))
    for dataset_key in datasets:
        draw_radar_chart(
            out_root / f"radar_noise_augmented_{dataset_key}.png",
            f"{specs[dataset_key].name} Noise-Augmented Robustness",
            axes,
            radar_values(all_eval_rows, dataset_key),
        )
    print(f"Wrote noise-augmented training summary: {train_summary_path}")
    print(f"Wrote noise-augmented test summary: {eval_summary_path}")
    print(f"Wrote noise-augmented validation summary: {out_root / 'noise_augmented_validation_summary.csv'}")
    print(f"Validation rows: {len(validation_rows)}")


if __name__ == "__main__":
    main()
