from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from wstgan_fftids.comparison_models import BiGANModel, FAnoGANModel, MTSDVGANModel, VAEModel
from wstgan_fftids.comparison_trainer import _eval_bigan, _eval_fanogan, _eval_mtsdvgan, _eval_vae
from wstgan_fftids.data import DataBundle, DatasetSpec, load_dataset_specs, make_dataloaders
from wstgan_fftids.metrics import best_balanced_accuracy, minmax
from wstgan_fftids.models import AblationOptions, build_models
from wstgan_fftids.trainer import TrainConfig, choose_device, evaluate, seed_everything


METHOD_LABELS = {
    "proposed": "Proposed",
    "if": "IF",
    "vae": "VAE",
    "f-anogan": "f-AnoGAN",
    "bigan": "BiGAN",
    "mts-dvgan": "MTS-DVGAN",
}

CORRUPTIONS = (
    ("clean", "Clean", "none", 0.0),
    ("gaussian_003", "Gaussian Noise", "gaussian", 0.03),
    ("gaussian_008", "Strong Gaussian", "gaussian", 0.08),
    ("speckle_006", "Speckle Noise", "speckle", 0.06),
    ("saltpepper_003", "Salt-and-Pepper", "saltpepper", 0.03),
    ("dropout_005", "Random Dropout", "dropout", 0.05),
)


@dataclass(frozen=True)
class Corruption:
    key: str
    label: str
    kind: str
    severity: float


@dataclass
class EvalResult:
    metrics: dict[str, float]
    labels: np.ndarray
    scores: np.ndarray
    paths: list[str]


class NoisyDataset(Dataset):
    def __init__(self, base: Dataset, corruption: Corruption, seed: int) -> None:
        self.base = base
        self.corruption = corruption
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label, path = self.base[index]
        return apply_noise(image, self.corruption, self.seed + index * 1009), label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IDS models under noisy traffic-image perturbations.")
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--dataset", default="all", choices=["all", "unsw", "cic", "toniot"])
    parser.add_argument("--methods", default="all", help="Comma list or all: proposed,if,vae,f-anogan,bigan,mts-dvgan")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--out-root", default="outputs/noise_robustness")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--comparison-root", default="", help="Root containing dataset/method/best_model.pt comparison checkpoints.")
    parser.add_argument("--comparison-base-channels", type=int, default=48)
    parser.add_argument("--proposed-root", default="", help="Root containing dataset/best_model.pt or dataset/full/best_model.pt proposed checkpoints.")
    parser.add_argument("--proposed-base-channels", type=int, default=48)
    parser.add_argument("--score-alpha", type=float, default=0.0)
    parser.add_argument("--score-beta", type=float, default=0.8)
    parser.add_argument("--score-gamma", type=float, default=0.2)
    parser.add_argument("--score-delta", type=float, default=0.0)
    parser.add_argument("--threshold-objective", default="acc", choices=["ba", "f1", "acc", "f1_acc"])
    return parser.parse_args()


def apply_noise(image: torch.Tensor, corruption: Corruption, seed: int) -> torch.Tensor:
    if corruption.kind == "none" or corruption.severity <= 0:
        return image
    generator = torch.Generator().manual_seed(seed)
    x = ((image + 1.0) * 0.5).clamp(0.0, 1.0)
    if corruption.kind == "gaussian":
        x = x + torch.randn(x.shape, generator=generator, dtype=x.dtype) * corruption.severity
    elif corruption.kind == "speckle":
        x = x + x * torch.randn(x.shape, generator=generator, dtype=x.dtype) * corruption.severity
    elif corruption.kind == "saltpepper":
        mask = torch.rand(x.shape, generator=generator) < corruption.severity
        salt = torch.rand(x.shape, generator=generator) < 0.5
        x = torch.where(mask & salt, torch.ones_like(x), x)
        x = torch.where(mask & ~salt, torch.zeros_like(x), x)
    elif corruption.kind == "dropout":
        mask = torch.rand(x.shape, generator=generator) < corruption.severity
        x = torch.where(mask, torch.zeros_like(x), x)
    else:
        raise ValueError(f"Unknown corruption kind: {corruption.kind}")
    return (x.clamp(0.0, 1.0) - 0.5) / 0.5


def make_noisy_bundle(bundle: DataBundle, corruption: Corruption, args: argparse.Namespace) -> DataBundle:
    base_ds = bundle.test.dataset
    noisy_ds = NoisyDataset(base_ds, corruption, args.seed)
    loader_kwargs = {}
    if args.workers > 0:
        loader_kwargs = {"persistent_workers": True, "prefetch_factor": 4}
    test_loader = DataLoader(
        noisy_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )
    return DataBundle(bundle.train, test_loader, bundle.spec, bundle.train_size, bundle.test_size, bundle.input_channels)


def comparison_path(dataset: str, method: str, root: str = "") -> Path:
    if root:
        return Path(root) / dataset / method / "best_model.pt"
    table = {
        ("unsw", "vae"): "outputs/comparison_30_bs256/20260606_181541/unsw/vae/best_model.pt",
        ("unsw", "f-anogan"): "outputs/comparison_30_bs256/20260606_181541/unsw/f-anogan/best_model.pt",
        ("unsw", "bigan"): "outputs/comparison_30_remote/20260606_185549/unsw/bigan/best_model.pt",
        ("unsw", "mts-dvgan"): "outputs/comparison_30_remote_fast/20260606_190703/unsw/mts-dvgan/best_model.pt",
        ("cic", "vae"): "outputs/comparison_30_remote_fast/20260606_191017/cic/vae/best_model.pt",
        ("cic", "f-anogan"): "outputs/comparison_30_remote_fast/20260606_191017/cic/f-anogan/best_model.pt",
        ("cic", "bigan"): "outputs/comparison_30_remote_fast/20260606_191017/cic/bigan/best_model.pt",
        ("cic", "mts-dvgan"): "outputs/comparison_30_remote_fast/20260606_191017/cic/mts-dvgan/best_model.pt",
        ("toniot", "vae"): "outputs/comparison_30_remote_fast/20260606_205742/toniot/vae/best_model.pt",
        ("toniot", "f-anogan"): "outputs/comparison_30_remote_fast/20260606_205742/toniot/f-anogan/best_model.pt",
        ("toniot", "bigan"): "outputs/comparison_30_remote_fast/20260606_205742/toniot/bigan/best_model.pt",
        ("toniot", "mts-dvgan"): "outputs/comparison_30_remote_fast/20260606_205742/toniot/mts-dvgan/best_model.pt",
    }
    return Path(table[(dataset, method)])


def proposed_path(dataset: str, root: str = "") -> Path:
    if root:
        candidates = [
            Path(root) / dataset / "best_model.pt",
            Path(root) / dataset / "full" / "best_model.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    return Path("outputs/ablation_30_remote_fast/20260606_214514") / dataset / "full" / "best_model.pt"


def result_from_rows(metrics: dict[str, float], rows: list[dict[str, float | int | str]]) -> EvalResult:
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.array([float(row["score"]) for row in rows], dtype=np.float64)
    paths = [str(row["path"]) for row in rows]
    return EvalResult(metrics, labels, scores, paths)


def evaluate_proposed(dataset_key: str, bundle: DataBundle, device: torch.device, args: argparse.Namespace) -> EvalResult:
    model_bundle = build_models(
        base_channels=args.proposed_base_channels,
        in_channels=bundle.input_channels,
        device=device,
        ablation=AblationOptions(),
    )
    ckpt = torch.load(proposed_path(dataset_key, args.proposed_root), map_location=device)
    model_bundle.generator.load_state_dict(ckpt["generator"])
    model_bundle.discriminator.load_state_dict(ckpt["discriminator"])
    eval_cfg = TrainConfig(
        device=str(device),
        score_alpha=args.score_alpha,
        score_beta=args.score_beta,
        score_gamma=args.score_gamma,
        score_delta=args.score_delta,
        threshold_objective=args.threshold_objective,
    )
    metrics, rows, _ = evaluate(model_bundle, bundle, eval_cfg, device)
    return result_from_rows(metrics, rows)


def evaluate_comparison(dataset_key: str, method: str, bundle: DataBundle, device: torch.device, args: argparse.Namespace) -> EvalResult:
    ckpt = torch.load(comparison_path(dataset_key, method, args.comparison_root), map_location=device)
    if method == "vae":
        model = VAEModel(base_channels=args.comparison_base_channels, in_channels=bundle.input_channels).to(device)
        model.load_state_dict(ckpt["model"])
        metrics, rows, _ = _eval_vae(model, bundle, device)
    elif method == "f-anogan":
        model = FAnoGANModel(base_channels=args.comparison_base_channels, in_channels=bundle.input_channels).to(device)
        model.load_state_dict(ckpt["model"])
        metrics, rows, _ = _eval_fanogan(model, bundle, device)
    elif method == "bigan":
        model = BiGANModel(base_channels=args.comparison_base_channels, in_channels=bundle.input_channels).to(device)
        model.load_state_dict(ckpt["model"])
        metrics, rows, _ = _eval_bigan(model, bundle, device)
    elif method == "mts-dvgan":
        model = MTSDVGANModel(base_channels=args.comparison_base_channels, in_channels=bundle.input_channels).to(device)
        model.load_state_dict(ckpt["model"])
        metrics, rows, _ = _eval_mtsdvgan(model, bundle, device)
    else:
        raise ValueError(f"Unsupported checkpoint comparison method: {method}")
    return result_from_rows(metrics, rows)


def collect_flat(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, paths = [], [], []
    for image, label, path in loader:
        xs.append(image.flatten(1).numpy())
        ys.append(label.numpy())
        paths.extend(path)
    return np.concatenate(xs), np.concatenate(ys), paths


def fit_if(bundle: DataBundle, seed: int) -> IsolationForest:
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


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int64)
    y = labels.astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    acc = (tp + tn) / max(1, len(y))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    far = fp / max(1, fp + tn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {"TransferAcc": acc, "TransferPrec": prec, "TransferRec": rec, "TransferFAR": far, "TransferF1": f1}


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    return 0.0 if denom == 0 else float(np.sum(ra * rb) / denom)


def append_csv(path: Path, row: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {(row["dataset"], row["method_key"], row["corruption_key"]) for row in csv.DictReader(handle)}


def draw_radar_chart(
    path: Path,
    title: str,
    axes: list[str],
    series: dict[str, list[float]],
    size: tuple[int, int] = (1800, 1350),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = 3
    width, height = size[0] * scale, size[1] * scale
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def s(value: float) -> int:
        return int(round(value * scale))

    def font(size_px: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        names = (
            "arialbd.ttf",
            "Arial Bold.ttf",
            "segoeuib.ttf",
        ) if bold else (
            "arial.ttf",
            "Arial.ttf",
            "segoeui.ttf",
        )
        for name in names:
            try:
                return ImageFont.truetype(name, s(size_px))
            except OSError:
                continue
        return ImageFont.load_default()

    font_title = font(42, bold=True)
    font_axis = font(24, bold=True)
    font_tick = font(19)
    font_legend = font(24, bold=True)
    font_small = font(20)

    colors = {
        "Proposed": (213, 94, 0),
        "IF": (0, 114, 178),
        "VAE": (0, 158, 115),
        "f-AnoGAN": (204, 121, 167),
        "BiGAN": (230, 159, 0),
        "MTS-DVGAN": (86, 80, 161),
    }

    cx, cy = s(690), s(720)
    radius = s(405)
    n = len(axes)
    angles = [(-np.pi / 2) + 2 * np.pi * i / n for i in range(n)]

    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    draw.text((s(70), s(46)), title, fill=(22, 32, 48, 255), font=font_title)
    draw.line((s(70), s(104), s(70) + (title_bbox[2] - title_bbox[0]), s(104)), fill=(213, 94, 0, 210), width=s(4))

    grid_color = (222, 228, 236, 255)
    spoke_color = (232, 236, 242, 255)
    label_color = (40, 50, 68, 255)
    tick_color = (119, 130, 146, 255)

    for level in range(1, 6):
        r = radius * level / 5
        points = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
        draw.line(points + [points[0]], fill=grid_color, width=s(2))
        tick = f"{level / 5:.1f}"
        draw.text((cx + s(12), cy - r - s(10)), tick, fill=tick_color, font=font_tick)

    for axis, angle in zip(axes, angles):
        end = (cx + radius * np.cos(angle), cy + radius * np.sin(angle))
        draw.line((cx, cy, end[0], end[1]), fill=spoke_color, width=s(2))
        lx = cx + s(112) * np.cos(angle) + radius * np.cos(angle)
        ly = cy + s(92) * np.sin(angle) + radius * np.sin(angle)
        bbox = draw.textbbox((0, 0), axis, font=font_axis)
        draw.text((lx - (bbox[2] - bbox[0]) / 2, ly - (bbox[3] - bbox[1]) / 2), axis, fill=label_color, font=font_axis)

    # Draw lower-priority methods first so the proposed method stays visible.
    ordered_names = [name for name in ["IF", "VAE", "f-AnoGAN", "BiGAN", "MTS-DVGAN", "Proposed"] if name in series]
    for name in ordered_names:
        color = colors.get(name, (85, 85, 85))
        rgba = (*color, 255)
        pts = []
        for angle, value in zip(angles, series[name]):
            v = max(0.0, min(1.0, float(value)))
            pts.append((cx + radius * v * np.cos(angle), cy + radius * v * np.sin(angle)))

        overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.polygon(pts, fill=(*color, 34 if name != "Proposed" else 54))
        image.alpha_composite(overlay)
        draw = ImageDraw.Draw(image)
        draw.line(pts + [pts[0]], fill=rgba, width=s(5 if name == "Proposed" else 4), joint="curve")
        for px, py in pts:
            rr = s(8 if name == "Proposed" else 6)
            draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=(255, 255, 255, 255), outline=rgba, width=s(3))

    legend_x = s(1230)
    legend_y = s(265)
    draw.text((legend_x, legend_y - s(70)), "Methods", fill=(22, 32, 48, 255), font=font_legend)
    for idx, name in enumerate([name for name in ["Proposed", "IF", "VAE", "f-AnoGAN", "BiGAN", "MTS-DVGAN"] if name in series]):
        y = legend_y + idx * s(78)
        color = colors.get(name, (85, 85, 85))
        draw.rounded_rectangle((legend_x, y, legend_x + s(52), y + s(18)), radius=s(9), fill=(*color, 255))
        draw.text((legend_x + s(74), y - s(10)), name, fill=(45, 55, 72, 255), font=font_small)

    draw.text((s(70), s(1260)), "Scale: 0.0 to 1.0; higher values indicate better robustness.", fill=(105, 116, 132, 255), font=font_small)

    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    image = image.resize(size, resample).convert("RGB")
    image.save(path, quality=96)


def radar_values(rows: list[dict[str, str | float]], dataset: str | None = None) -> dict[str, list[float]]:
    selected = [r for r in rows if dataset is None or r["dataset"] == dataset]
    by_method: dict[str, list[dict[str, str | float]]] = {}
    for row in selected:
        by_method.setdefault(str(row["method"]), []).append(row)
    result = {}
    for method, items in by_method.items():
        by_corruption = {str(i["corruption_key"]): i for i in items}
        clean_auc = float(by_corruption["clean"]["AUC"])
        noisy = [i for i in items if i["corruption_key"] != "clean"]
        mean_noisy_auc = float(np.mean([float(i["AUC"]) for i in noisy]))
        worst_noisy_auc = float(np.min([float(i["AUC"]) for i in noisy]))
        mean_noisy_f1 = float(np.mean([float(i["F1"]) for i in noisy]))
        mean_transfer_f1 = float(np.mean([float(i["TransferF1"]) for i in noisy]))
        stability = float(np.mean([float(i["RankCorrelation"]) for i in noisy]))
        result[method] = [
            clean_auc,
            mean_noisy_auc,
            worst_noisy_auc,
            mean_noisy_f1,
            mean_transfer_f1,
            max(0.0, stability),
        ]
    order = ["Proposed", "IF", "VAE", "f-AnoGAN", "BiGAN", "MTS-DVGAN"]
    return {name: result[name] for name in order if name in result}


def write_validation_summary(path: Path, rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    grouped: dict[tuple[str, str], list[dict[str, str | float]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)
    summary = []
    for (dataset, method), items in grouped.items():
        clean = next(i for i in items if i["corruption_key"] == "clean")
        noisy = [i for i in items if i["corruption_key"] != "clean"]
        clean_auc = float(clean["AUC"])
        clean_f1 = float(clean["F1"])
        mean_noisy_auc = float(np.mean([float(i["AUC"]) for i in noisy]))
        worst_noisy_auc = float(np.min([float(i["AUC"]) for i in noisy]))
        mean_noisy_f1 = float(np.mean([float(i["F1"]) for i in noisy]))
        mean_transfer_f1 = float(np.mean([float(i["TransferF1"]) for i in noisy]))
        mean_rank = float(np.mean([float(i["RankCorrelation"]) for i in noisy]))
        summary.append(
            {
                "dataset": dataset,
                "method": method,
                "clean_auc": clean_auc,
                "clean_f1": clean_f1,
                "mean_noisy_auc": mean_noisy_auc,
                "worst_noisy_auc": worst_noisy_auc,
                "auc_retention": mean_noisy_auc / max(clean_auc, 1e-12),
                "auc_drop": clean_auc - mean_noisy_auc,
                "mean_noisy_f1": mean_noisy_f1,
                "mean_transfer_f1": mean_transfer_f1,
                "mean_rank_correlation": mean_rank,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary[0].keys()) if summary else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    specs = load_dataset_specs(args.config)
    datasets = list(specs) if args.dataset == "all" else [args.dataset]
    methods = list(METHOD_LABELS) if args.methods == "all" else [m.strip() for m in args.methods.split(",") if m.strip()]
    corruptions = [Corruption(*item) for item in CORRUPTIONS]
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    summary_path = out_root / "noise_robustness_summary.csv"
    completed = load_existing_keys(summary_path) if args.skip_existing else set()
    all_rows: list[dict[str, str | float]] = []
    clean_cache: dict[tuple[str, str], EvalResult] = {}

    for dataset_key in datasets:
        spec: DatasetSpec = specs[dataset_key]
        print(f"Loading {spec.name} ...")
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
        if_model = fit_if(bundle, args.seed) if "if" in methods else None
        for method in methods:
            method_label = METHOD_LABELS[method]
            for corruption in corruptions:
                if (dataset_key, method, corruption.key) in completed:
                    continue
                print(f"Evaluating {spec.name} / {method_label} / {corruption.label}")
                noisy_bundle = make_noisy_bundle(bundle, corruption, args)
                torch.manual_seed(args.seed)
                if method == "proposed":
                    result = evaluate_proposed(dataset_key, noisy_bundle, device, args)
                elif method == "if":
                    assert if_model is not None
                    result = evaluate_if(if_model, noisy_bundle)
                else:
                    result = evaluate_comparison(dataset_key, method, noisy_bundle, device, args)
                key = (dataset_key, method)
                if corruption.key == "clean":
                    clean_cache[key] = result
                    transfer = metrics_at_threshold(result.labels, result.scores, result.metrics["Threshold"])
                    rank_corr = 1.0
                else:
                    clean = clean_cache.get(key)
                    if clean is None:
                        raise RuntimeError(f"Clean result must be evaluated before noisy conditions for {dataset_key}/{method}")
                    transfer = metrics_at_threshold(result.labels, result.scores, clean.metrics["Threshold"])
                    rank_corr = rank_correlation(clean.scores, result.scores)
                row = {
                    "dataset": dataset_key,
                    "dataset_name": spec.name,
                    "method_key": method,
                    "method": method_label,
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
                all_rows.append(row)

    if not all_rows and summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
    validation_rows = write_validation_summary(out_root / "robustness_validation_summary.csv", all_rows)
    axes = ["Clean AUC", "Mean Noisy AUC", "Worst Noisy AUC", "Mean Noisy F1", "Transfer F1", "Rank Stability"]
    draw_radar_chart(out_root / "radar_overall.png", "Overall Noise Robustness Validation", axes, radar_values(all_rows))
    for dataset_key in datasets:
        draw_radar_chart(
            out_root / f"radar_{dataset_key}.png",
            f"{specs[dataset_key].name} Noise Robustness Validation",
            axes,
            radar_values(all_rows, dataset_key),
        )
    print(f"Wrote noise robustness summary: {summary_path}")
    print(f"Wrote validation summary: {out_root / 'robustness_validation_summary.csv'}")
    print(f"Validation rows: {len(validation_rows)}")


if __name__ == "__main__":
    main()
