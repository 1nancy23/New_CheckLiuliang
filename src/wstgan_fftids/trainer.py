from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import DataBundle
from .metrics import best_threshold_metrics, minmax, roc_points, save_json, save_scores
from .models import AblationOptions, ModelBundle, build_models
from .visualize import save_line_plot, save_metrics_bar, save_reconstruction_grid, save_roc_plot


@dataclass
class TrainConfig:
    ablation_name: str = "full"
    epochs: int = 30
    lr: float = 2e-4
    beta1: float = 0.5
    w_adv: float = 1.0
    w_con: float = 50.0
    w_lat: float = 1.0
    w_freq: float = 0.5
    score_alpha: float = 0.35
    score_beta: float = 0.35
    score_gamma: float = 0.20
    score_delta: float = 0.10
    base_channels: int = 32
    device: str = "cuda"
    seed: int = 3407
    lr_policy: str = "lambda"
    lr_decay_start: int = 15
    warmup_epochs: int = 0
    min_lr_ratio: float = 0.05
    eval_every: int = 5
    summary_path: str = "outputs/summary_metrics.csv"
    use_fft_prior: bool = True
    use_temporal: bool = True
    use_st_fusion: bool = True
    use_cffm: bool = True
    fft_low_cutoff: float = 0.18
    fft_mid_cutoff: float = 0.36
    temporal_hidden_ratio: float = 0.5
    cffm_bottleneck_ratio: float = 1.0
    adv_loss: str = "bce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    threshold_objective: str = "ba"
    selection_metric: str = "AUC"
    grad_clip: float = 0.0
    ema_decay: float = 0.0
    ema_start_epoch: int = 1
    pretrained_path: str = ""
    pretrained_strict: bool = True


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def choose_device(name: str) -> torch.device:
    if name != "cpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _append_summary(path: Path, row: dict[str, str | float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _make_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> torch.optim.lr_scheduler.LRScheduler:
    if cfg.lr_policy == "warmup_cosine":
        warmup = max(0, min(cfg.warmup_epochs, cfg.epochs - 1))
        min_ratio = min(max(cfg.min_lr_ratio, 0.0), 1.0)

        def rule(epoch_index: int) -> float:
            epoch = epoch_index + 1
            if warmup > 0 and epoch <= warmup:
                return max(1e-6, epoch / warmup)
            progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
            cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
            return min_ratio + (1.0 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)
    if cfg.lr_policy == "lambda":
        decay_start = max(1, min(cfg.lr_decay_start, cfg.epochs))

        def rule(epoch_index: int) -> float:
            epoch = epoch_index + 1
            if epoch <= decay_start:
                return 1.0
            span = max(1, cfg.epochs - decay_start)
            return max(0.0, 1.0 - (epoch - decay_start) / span)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)
    if cfg.lr_policy == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0.0)
    if cfg.lr_policy == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5)
    raise ValueError(f"Unsupported lr policy: {cfg.lr_policy}")


def _append_history(path: Path, row: dict[str, float | int | str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class BinaryFocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, prob, 1.0 - prob)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return loss.mean()


def _init_ema(models: list[nn.Module]) -> list[dict[str, torch.Tensor]]:
    return [{name: param.detach().clone() for name, param in model.state_dict().items()} for model in models]


def _update_ema(models: list[nn.Module], shadows: list[dict[str, torch.Tensor]], decay: float) -> None:
    with torch.no_grad():
        for model, shadow in zip(models, shadows):
            for name, value in model.state_dict().items():
                if value.dtype.is_floating_point:
                    shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
                else:
                    shadow[name].copy_(value)


def _swap_ema(models: list[nn.Module], shadows: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    backups = _init_ema(models)
    for model, shadow in zip(models, shadows):
        model.load_state_dict(shadow, strict=True)
    return backups


def _selection_value(metrics: dict[str, float], cfg: TrainConfig) -> float:
    if cfg.selection_metric == "F1_Acc":
        return 0.5 * metrics["F1"] + 0.5 * metrics["Acc"]
    if cfg.selection_metric not in metrics:
        raise ValueError(f"Unsupported selection metric: {cfg.selection_metric}")
    return float(metrics[cfg.selection_metric])


def _load_compatible_state_dict(
    module: nn.Module,
    pretrained_state: dict[str, torch.Tensor],
) -> tuple[torch.nn.modules.module._IncompatibleKeys, int, int]:
    current_state = module.state_dict()
    compatible = {}
    skipped_shape = 0
    skipped_unknown = 0
    for key, value in pretrained_state.items():
        if key not in current_state:
            skipped_unknown += 1
            continue
        if current_state[key].shape != value.shape:
            skipped_shape += 1
            continue
        compatible[key] = value
    result = module.load_state_dict(compatible, strict=False)
    return result, skipped_shape, skipped_unknown


def train_one_dataset(bundle: DataBundle, cfg: TrainConfig, out_dir: Path) -> dict[str, float]:
    seed_everything(cfg.seed)
    device = choose_device(cfg.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation = AblationOptions(
        use_fft_prior=cfg.use_fft_prior,
        use_temporal=cfg.use_temporal,
        use_st_fusion=cfg.use_st_fusion,
        use_cffm=cfg.use_cffm,
        fft_low_cutoff=cfg.fft_low_cutoff,
        fft_mid_cutoff=cfg.fft_mid_cutoff,
        temporal_hidden_ratio=cfg.temporal_hidden_ratio,
        cffm_bottleneck_ratio=cfg.cffm_bottleneck_ratio,
    )
    models: ModelBundle = build_models(
        in_channels=bundle.input_channels,
        base_channels=cfg.base_channels,
        device=device,
        ablation=ablation,
    )
    net_g = models.generator
    net_d = models.discriminator
    if cfg.pretrained_path:
        checkpoint = torch.load(cfg.pretrained_path, map_location=device)
        if "generator" not in checkpoint or "discriminator" not in checkpoint:
            raise KeyError(f"Pretrained checkpoint must contain generator and discriminator: {cfg.pretrained_path}")
        mode = "strict" if cfg.pretrained_strict else "non-strict"
        if cfg.pretrained_strict:
            g_result = net_g.load_state_dict(checkpoint["generator"], strict=True)
            d_result = net_d.load_state_dict(checkpoint["discriminator"], strict=True)
            g_skipped_shape = 0
            g_skipped_unknown = 0
            d_skipped_shape = 0
            d_skipped_unknown = 0
        else:
            g_result, g_skipped_shape, g_skipped_unknown = _load_compatible_state_dict(net_g, checkpoint["generator"])
            d_result, d_skipped_shape, d_skipped_unknown = _load_compatible_state_dict(net_d, checkpoint["discriminator"])
        print(f"Loaded pretrained checkpoint ({mode}): {cfg.pretrained_path}")
        if not cfg.pretrained_strict:
            print(
                "Pretrained load report: "
                f"G missing={len(g_result.missing_keys)} unexpected={len(g_result.unexpected_keys)}; "
                f"G skipped_shape={g_skipped_shape} skipped_unknown={g_skipped_unknown}; "
                f"D missing={len(d_result.missing_keys)} unexpected={len(d_result.unexpected_keys)}; "
                f"D skipped_shape={d_skipped_shape} skipped_unknown={d_skipped_unknown}"
            )
    opt_g = torch.optim.Adam(net_g.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=cfg.lr * 0.25, betas=(cfg.beta1, 0.999))
    scheduler_g = _make_scheduler(opt_g, cfg)
    scheduler_d = _make_scheduler(opt_d, cfg)
    if cfg.adv_loss == "focal":
        adv_criterion: nn.Module = BinaryFocalWithLogitsLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    elif cfg.adv_loss == "bce":
        adv_criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported adversarial loss: {cfg.adv_loss}")
    loss_log: dict[str, list[float]] = {"g": [], "d": [], "con": [], "lat": [], "freq": []}
    ema_shadows = _init_ema([net_g, net_d]) if cfg.ema_decay > 0.0 else None

    best_metrics: dict[str, float] | None = None
    best_selection = -float("inf")
    best_score_rows: list[dict[str, float | int | str]] = []
    best_roc_data: tuple[np.ndarray, np.ndarray, float] | None = None
    first_real = None
    first_fake = None
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        net_g.train()
        net_d.train()
        epoch_loss = {key: 0.0 for key in loss_log}
        batches = 0
        for real, _, _ in bundle.train:
            real = real.to(device, non_blocking=True)
            batches += 1

            with torch.no_grad():
                fake_for_d, _, _ = net_g(real)
            real_logits, _ = net_d(real)
            fake_logits, _ = net_d(fake_for_d.detach())
            real_target = torch.ones_like(real_logits)
            fake_target = torch.zeros_like(fake_logits)
            d_loss = adv_criterion(real_logits, real_target) + adv_criterion(fake_logits, fake_target)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(net_d.parameters(), cfg.grad_clip)
            opt_d.step()

            fake, _, _ = net_g(real)
            fake_logits, fake_feat = net_d(fake)
            with torch.no_grad():
                _, real_feat = net_d(real)
            adv_loss = adv_criterion(fake_logits, torch.ones_like(fake_logits))
            con_loss = F.l1_loss(fake, real)
            lat_loss = F.mse_loss(fake_feat, real_feat.detach())
            freq_loss = net_g.frequency_consistency(real, fake)
            g_loss = (
                cfg.w_adv * adv_loss
                + cfg.w_con * con_loss
                + cfg.w_lat * lat_loss
                + cfg.w_freq * freq_loss
            )
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(net_g.parameters(), cfg.grad_clip)
            opt_g.step()
            if ema_shadows is not None and epoch >= cfg.ema_start_epoch:
                _update_ema([net_g, net_d], ema_shadows, cfg.ema_decay)

            epoch_loss["g"] += float(g_loss.detach().cpu())
            epoch_loss["d"] += float(d_loss.detach().cpu())
            epoch_loss["con"] += float(con_loss.detach().cpu())
            epoch_loss["lat"] += float(lat_loss.detach().cpu())
            epoch_loss["freq"] += float(freq_loss.detach().cpu())

            if first_real is None:
                first_real = real[:8].detach().cpu()
                first_fake = fake[:8].detach().cpu()

        for key in loss_log:
            loss_log[key].append(epoch_loss[key] / max(1, batches))

        should_eval = epoch == 1 or epoch == cfg.epochs or epoch % max(1, cfg.eval_every) == 0
        metrics = None
        score_rows = []
        roc_data = None
        if should_eval:
            ema_backup = None
            if ema_shadows is not None and epoch >= cfg.ema_start_epoch:
                ema_backup = _swap_ema([net_g, net_d], ema_shadows)
            metrics, score_rows, roc_data = evaluate(models, bundle, cfg, device)
            if ema_backup is not None:
                _swap_ema([net_g, net_d], ema_backup)
            metrics["epoch"] = float(epoch)
            metrics["SelectionMetric"] = cfg.selection_metric
            selection_value = _selection_value(metrics, cfg)
            if best_metrics is None or selection_value > best_selection:
                best_selection = selection_value
                best_metrics = metrics
                best_score_rows = score_rows
                best_roc_data = roc_data
                if ema_shadows is not None and epoch >= cfg.ema_start_epoch:
                    ema_backup = _swap_ema([net_g, net_d], ema_shadows)
                    torch.save({"generator": net_g.state_dict(), "discriminator": net_d.state_dict(), "metrics": metrics}, out_dir / "best_model.pt")
                    _swap_ema([net_g, net_d], ema_backup)
                else:
                    torch.save({"generator": net_g.state_dict(), "discriminator": net_d.state_dict(), "metrics": metrics}, out_dir / "best_model.pt")

        current_lr_g = float(opt_g.param_groups[0]["lr"])
        current_lr_d = float(opt_d.param_groups[0]["lr"])
        torch.save(
            {
                "epoch": epoch,
                "generator": net_g.state_dict(),
                "discriminator": net_d.state_dict(),
                "optimizer_g": opt_g.state_dict(),
                "optimizer_d": opt_d.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                "best_metrics": best_metrics,
            },
            out_dir / "latest_model.pt",
        )
        _append_history(
            out_dir / "training_history.csv",
            {
                "epoch": epoch,
                "lr_g": current_lr_g,
                "lr_d": current_lr_d,
                "g": loss_log["g"][-1],
                "d": loss_log["d"][-1],
                "con": loss_log["con"][-1],
                "lat": loss_log["lat"][-1],
                "freq": loss_log["freq"][-1],
                "Acc": "" if metrics is None else metrics["Acc"],
                "F1": "" if metrics is None else metrics["F1"],
                "AUC": "" if metrics is None else metrics["AUC"],
                "Selection": "" if metrics is None else _selection_value(metrics, cfg),
            },
        )
        print(
            f"[{bundle.spec.key}] epoch {epoch}/{cfg.epochs} "
            f"lr={current_lr_g:.7f} g={loss_log['g'][-1]:.4f} d={loss_log['d'][-1]:.4f} "
            + ("eval=skip" if metrics is None else f"auc={metrics['AUC']:.4f} f1={metrics['F1']:.4f}")
        )
        scheduler_g.step()
        scheduler_d.step()

    assert best_metrics is not None
    assert best_roc_data is not None
    elapsed = time.time() - start
    best_metrics["seconds"] = float(elapsed)
    best_metrics["train_size"] = float(bundle.train_size)
    best_metrics["test_size"] = float(bundle.test_size)
    best_metrics["variant"] = cfg.ablation_name
    save_json(out_dir / "metrics.json", best_metrics)
    save_scores(out_dir / "scores.csv", best_score_rows)
    fpr, tpr, roc_auc = best_roc_data
    save_line_plot(out_dir / "loss_curve.png", loss_log, f"{bundle.spec.name} training losses")
    save_roc_plot(out_dir / "roc_curve.png", fpr, tpr, roc_auc)
    save_metrics_bar(out_dir / "metrics_bar.png", best_metrics)
    if first_real is not None and first_fake is not None:
        save_reconstruction_grid(out_dir / "reconstruction_grid.png", first_real, first_fake)

    if cfg.summary_path:
        _append_summary(
            Path(cfg.summary_path),
            {
                "dataset": bundle.spec.key,
                "name": bundle.spec.name,
                "out_dir": str(out_dir),
                "epoch": int(best_metrics["epoch"]),
                "Threshold": best_metrics["Threshold"],
                "Acc": best_metrics["Acc"],
                "Prec": best_metrics["Prec"],
                "Rec": best_metrics["Rec"],
                "FAR": best_metrics["FAR"],
                "F1": best_metrics["F1"],
                "AUC": best_metrics["AUC"],
                "seconds": best_metrics["seconds"],
                "train_size": int(best_metrics["train_size"]),
                "test_size": int(best_metrics["test_size"]),
                "input_channels": int(bundle.input_channels),
            },
        )
    return best_metrics


def evaluate(
    models: ModelBundle,
    bundle: DataBundle,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float | int | str]], tuple[np.ndarray, np.ndarray, float]]:
    net_g = models.generator
    net_d = models.discriminator
    net_g.eval()
    net_d.eval()
    labels = []
    paths = []
    d_scores = []
    rec_scores = []
    lat_scores = []
    freq_scores = []

    with torch.no_grad():
        for real, label, path in bundle.test:
            real = real.to(device, non_blocking=True)
            fake, _, _ = net_g(real)
            real_logits, real_feat = net_d(real)
            _, fake_feat = net_d(fake)
            rec = torch.mean(torch.abs(real - fake).flatten(1), dim=1)
            lat = torch.mean((real_feat - fake_feat).pow(2).flatten(1), dim=1)
            freq = net_g.frequency_error(real, fake)
            disc = 1.0 - torch.sigmoid(real_logits)

            labels.append(label.numpy())
            paths.extend(path)
            d_scores.append(disc.cpu().numpy())
            rec_scores.append(rec.cpu().numpy())
            lat_scores.append(lat.cpu().numpy())
            freq_scores.append(freq.cpu().numpy())

    y = np.concatenate(labels)
    disc = minmax(np.concatenate(d_scores))
    rec = minmax(np.concatenate(rec_scores))
    lat = minmax(np.concatenate(lat_scores))
    freq = minmax(np.concatenate(freq_scores))
    score = minmax(cfg.score_alpha * disc + cfg.score_beta * rec + cfg.score_gamma * lat + cfg.score_delta * freq)
    metrics = best_threshold_metrics(y, score, objective=cfg.threshold_objective)
    fpr, tpr, roc_auc = roc_points(y, score)

    rows = []
    for idx, path in enumerate(paths):
        rows.append(
            {
                "path": path,
                "label": int(y[idx]),
                "score": float(score[idx]),
                "disc_score": float(disc[idx]),
                "reconstruction_score": float(rec[idx]),
                "latent_score": float(lat[idx]),
                "fft_score": float(freq[idx]),
            }
        )
    return metrics, rows, (fpr, tpr, roc_auc)
