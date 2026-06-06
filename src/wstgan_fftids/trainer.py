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
from .metrics import best_balanced_accuracy, minmax, roc_points, save_json, save_scores
from .models import ModelBundle, build_models
from .visualize import save_line_plot, save_metrics_bar, save_reconstruction_grid, save_roc_plot


@dataclass
class TrainConfig:
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
    eval_every: int = 5


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


def train_one_dataset(bundle: DataBundle, cfg: TrainConfig, out_dir: Path) -> dict[str, float]:
    seed_everything(cfg.seed)
    device = choose_device(cfg.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    models: ModelBundle = build_models(base_channels=cfg.base_channels, device=device)
    net_g = models.generator
    net_d = models.discriminator
    opt_g = torch.optim.Adam(net_g.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=cfg.lr * 0.25, betas=(cfg.beta1, 0.999))
    scheduler_g = _make_scheduler(opt_g, cfg)
    scheduler_d = _make_scheduler(opt_d, cfg)
    bce = nn.BCEWithLogitsLoss()
    loss_log: dict[str, list[float]] = {"g": [], "d": [], "con": [], "lat": [], "freq": []}

    best_metrics: dict[str, float] | None = None
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
            d_loss = bce(real_logits, real_target) + bce(fake_logits, fake_target)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            fake, _, _ = net_g(real)
            fake_logits, fake_feat = net_d(fake)
            with torch.no_grad():
                _, real_feat = net_d(real)
            adv_loss = bce(fake_logits, torch.ones_like(fake_logits))
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
            opt_g.step()

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
            metrics, score_rows, roc_data = evaluate(models, bundle, cfg, device)
            metrics["epoch"] = float(epoch)
            if best_metrics is None or metrics["AUC"] > best_metrics["AUC"]:
                best_metrics = metrics
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
    elapsed = time.time() - start
    best_metrics["seconds"] = float(elapsed)
    best_metrics["train_size"] = float(bundle.train_size)
    best_metrics["test_size"] = float(bundle.test_size)
    save_json(out_dir / "metrics.json", best_metrics)
    save_scores(out_dir / "scores.csv", score_rows)
    fpr, tpr, roc_auc = roc_data
    save_line_plot(out_dir / "loss_curve.png", loss_log, f"{bundle.spec.name} training losses")
    save_roc_plot(out_dir / "roc_curve.png", fpr, tpr, roc_auc)
    save_metrics_bar(out_dir / "metrics_bar.png", best_metrics)
    if first_real is not None and first_fake is not None:
        save_reconstruction_grid(out_dir / "reconstruction_grid.png", first_real, first_fake)

    _append_summary(
        Path("outputs") / "summary_metrics.csv",
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
    metrics = best_balanced_accuracy(y, score)
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
