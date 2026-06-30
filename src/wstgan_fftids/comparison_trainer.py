from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest

from .comparison_models import BiGANModel, FAnoGANModel, MTSDVGANModel, VAEModel, kl_loss
from .data import DataBundle
from .metrics import best_balanced_accuracy, minmax, roc_points, save_json, save_scores
from .trainer import choose_device, seed_everything
from .visualize import save_line_plot, save_metrics_bar, save_reconstruction_grid, save_roc_plot


METHODS = ("if", "vae", "f-anogan", "bigan", "mts-dvgan")
METHOD_NAMES = {
    "if": "IF",
    "vae": "VAE",
    "f-anogan": "f-AnoGAN",
    "bigan": "BiGAN",
    "mts-dvgan": "MTS-DVGAN",
}


@dataclass
class ComparisonConfig:
    epochs: int = 30
    lr: float = 2e-4
    beta1: float = 0.5
    batch_size: int = 256
    base_channels: int = 32
    latent_dim: int = 64
    input_channels: int = 3
    device: str = "cuda"
    seed: int = 3407
    lr_decay_start: int = 15
    eval_every: int = 5
    if_estimators: int = 100
    if_max_samples: int = 10000


def _scheduler(optimizer: torch.optim.Optimizer, cfg: ComparisonConfig) -> torch.optim.lr_scheduler.LambdaLR:
    decay_start = max(1, min(cfg.lr_decay_start, cfg.epochs))

    def rule(epoch_index: int) -> float:
        epoch = epoch_index + 1
        if epoch <= decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / max(1, cfg.epochs - decay_start))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)


def _append_csv(path: Path, row: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _collect_flat(loader, device: torch.device | str = "cpu") -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, paths = [], [], []
    for image, label, path in loader:
        xs.append(image.flatten(1).numpy())
        ys.append(label.numpy())
        paths.extend(path)
    return np.concatenate(xs), np.concatenate(ys), paths


def train_if(bundle: DataBundle, cfg: ComparisonConfig, out_dir: Path) -> dict[str, float]:
    seed_everything(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    x_train, _, _ = _collect_flat(bundle.train)
    x_test, labels, paths = _collect_flat(bundle.test)
    model = IsolationForest(
        n_estimators=cfg.if_estimators,
        max_samples=min(cfg.if_max_samples, len(x_train)),
        contamination="auto",
        random_state=cfg.seed,
        n_jobs=-1,
    )
    model.fit(x_train)
    scores = minmax(-model.decision_function(x_test))
    metrics = best_balanced_accuracy(labels, scores)
    metrics["epoch"] = 0.0
    metrics["seconds"] = float(time.time() - start)
    metrics["train_size"] = float(bundle.train_size)
    metrics["test_size"] = float(bundle.test_size)
    rows = [{"path": p, "label": int(labels[i]), "score": float(scores[i])} for i, p in enumerate(paths)]
    fpr, tpr, auc = roc_points(labels, scores)
    save_json(out_dir / "metrics.json", metrics)
    save_scores(out_dir / "scores.csv", rows)
    save_roc_plot(out_dir / "roc_curve.png", fpr, tpr, auc)
    save_metrics_bar(out_dir / "metrics_bar.png", metrics)
    return metrics


def train_vae(bundle: DataBundle, cfg: ComparisonConfig, out_dir: Path) -> dict[str, float]:
    device = choose_device(cfg.device)
    model = VAEModel(cfg.latent_dim, cfg.base_channels, in_channels=cfg.input_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    sched = _scheduler(opt, cfg)
    return _train_reconstruction_model("vae", model, opt, sched, bundle, cfg, out_dir)


def train_fanogan(bundle: DataBundle, cfg: ComparisonConfig, out_dir: Path) -> dict[str, float]:
    device = choose_device(cfg.device)
    model = FAnoGANModel(cfg.latent_dim, cfg.base_channels, in_channels=cfg.input_channels).to(device)
    opt_g = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.lr * 0.25, betas=(cfg.beta1, 0.999))
    sched_g = _scheduler(opt_g, cfg)
    sched_d = _scheduler(opt_d, cfg)
    return _train_gan_reconstruction_model("f-anogan", model, opt_g, opt_d, sched_g, sched_d, bundle, cfg, out_dir)


def train_bigan(bundle: DataBundle, cfg: ComparisonConfig, out_dir: Path) -> dict[str, float]:
    device = choose_device(cfg.device)
    model = BiGANModel(cfg.latent_dim, cfg.base_channels, in_channels=cfg.input_channels).to(device)
    opt_g = torch.optim.Adam(list(model.encoder.parameters()) + list(model.generator.parameters()), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.lr * 0.25, betas=(cfg.beta1, 0.999))
    sched_g = _scheduler(opt_g, cfg)
    sched_d = _scheduler(opt_d, cfg)
    return _train_bigan_model(model, opt_g, opt_d, sched_g, sched_d, bundle, cfg, out_dir)


def train_mtsdvgan(bundle: DataBundle, cfg: ComparisonConfig, out_dir: Path) -> dict[str, float]:
    device = choose_device(cfg.device)
    model = MTSDVGANModel(cfg.latent_dim, cfg.base_channels, in_channels=cfg.input_channels).to(device)
    enc_params = list(model.local_encoder.parameters()) + list(model.global_encoder.parameters()) + list(model.decoder.parameters())
    opt_g = torch.optim.Adam(enc_params, lr=cfg.lr, betas=(cfg.beta1, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.lr * 0.25, betas=(cfg.beta1, 0.999))
    sched_g = _scheduler(opt_g, cfg)
    sched_d = _scheduler(opt_d, cfg)
    return _train_mtsdvgan_model(model, opt_g, opt_d, sched_g, sched_d, bundle, cfg, out_dir)


def _train_reconstruction_model(method: str, model, opt, sched, bundle: DataBundle, cfg: ComparisonConfig, out_dir: Path) -> dict[str, float]:
    seed_everything(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.device)
    history = {"loss": [], "rec": [], "kl": []}
    best_metrics = None
    first_real = None
    first_fake = None
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "rec": 0.0, "kl": 0.0}
        batches = 0
        for real, _, _ in bundle.train:
            real = real.to(device, non_blocking=True)
            fake, mu, logvar = model(real)
            rec = F.l1_loss(fake, real)
            kl = kl_loss(mu, logvar)
            loss = rec + 0.01 * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sums["loss"] += float(loss.detach().cpu())
            sums["rec"] += float(rec.detach().cpu())
            sums["kl"] += float(kl.detach().cpu())
            batches += 1
            if first_real is None:
                first_real = real[:8].detach().cpu()
                first_fake = fake[:8].detach().cpu()
        for key in history:
            history[key].append(sums[key] / max(1, batches))
        metrics = None
        score_rows = []
        roc_data = None
        if epoch == 1 or epoch == cfg.epochs or epoch % cfg.eval_every == 0:
            metrics, score_rows, roc_data = _eval_vae(model, bundle, device)
            metrics["epoch"] = float(epoch)
            if best_metrics is None or metrics["AUC"] > best_metrics["AUC"]:
                best_metrics = metrics
                torch.save({"model": model.state_dict(), "metrics": metrics}, out_dir / "best_model.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict()}, out_dir / "latest_model.pt")
        _append_csv(out_dir / "training_history.csv", {"epoch": epoch, "lr": opt.param_groups[0]["lr"], **{k: history[k][-1] for k in history}, "Acc": "" if metrics is None else metrics["Acc"], "F1": "" if metrics is None else metrics["F1"], "AUC": "" if metrics is None else metrics["AUC"]})
        print(f"[{bundle.spec.key}][{method}] epoch {epoch}/{cfg.epochs} loss={history['loss'][-1]:.4f} " + ("eval=skip" if metrics is None else f"auc={metrics['AUC']:.4f} f1={metrics['F1']:.4f}"))
        sched.step()
    assert best_metrics is not None
    _finalize(out_dir, best_metrics, score_rows, roc_data, history, first_real, first_fake, bundle, start)
    return best_metrics


def _train_gan_reconstruction_model(method: str, model: FAnoGANModel, opt_g, opt_d, sched_g, sched_d, bundle, cfg, out_dir):
    seed_everything(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.device)
    bce = torch.nn.BCEWithLogitsLoss()
    history = {"g": [], "d": [], "rec": [], "lat": []}
    best_metrics = None
    first_real = None
    first_fake = None
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sums = {k: 0.0 for k in history}
        batches = 0
        for real, _, _ in bundle.train:
            real = real.to(device, non_blocking=True)
            fake, _ = model(real)
            real_logits, _ = model.discriminator(real)
            fake_logits, _ = model.discriminator(fake.detach())
            d_loss = bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            fake, _ = model(real)
            fake_logits, fake_feat = model.discriminator(fake)
            with torch.no_grad():
                _, real_feat = model.discriminator(real)
            rec = F.l1_loss(fake, real)
            lat = F.mse_loss(fake_feat, real_feat.detach())
            g_loss = 50.0 * rec + lat + bce(fake_logits, torch.ones_like(fake_logits))
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            for key, value in (("g", g_loss), ("d", d_loss), ("rec", rec), ("lat", lat)):
                sums[key] += float(value.detach().cpu())
            batches += 1
            if first_real is None:
                first_real = real[:8].detach().cpu()
                first_fake = fake[:8].detach().cpu()
        for key in history:
            history[key].append(sums[key] / max(1, batches))
        metrics = None
        score_rows = []
        roc_data = None
        if epoch == 1 or epoch == cfg.epochs or epoch % cfg.eval_every == 0:
            metrics, score_rows, roc_data = _eval_fanogan(model, bundle, device)
            metrics["epoch"] = float(epoch)
            if best_metrics is None or metrics["AUC"] > best_metrics["AUC"]:
                best_metrics = metrics
                torch.save({"model": model.state_dict(), "metrics": metrics}, out_dir / "best_model.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict()}, out_dir / "latest_model.pt")
        _append_csv(out_dir / "training_history.csv", {"epoch": epoch, "lr_g": opt_g.param_groups[0]["lr"], "lr_d": opt_d.param_groups[0]["lr"], **{k: history[k][-1] for k in history}, "Acc": "" if metrics is None else metrics["Acc"], "F1": "" if metrics is None else metrics["F1"], "AUC": "" if metrics is None else metrics["AUC"]})
        print(f"[{bundle.spec.key}][{method}] epoch {epoch}/{cfg.epochs} g={history['g'][-1]:.4f} d={history['d'][-1]:.4f} " + ("eval=skip" if metrics is None else f"auc={metrics['AUC']:.4f} f1={metrics['F1']:.4f}"))
        sched_g.step()
        sched_d.step()
    assert best_metrics is not None
    _finalize(out_dir, best_metrics, score_rows, roc_data, history, first_real, first_fake, bundle, start)
    return best_metrics


def _train_bigan_model(model: BiGANModel, opt_g, opt_d, sched_g, sched_d, bundle, cfg, out_dir):
    seed_everything(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.device)
    bce = torch.nn.BCEWithLogitsLoss()
    history = {"g": [], "d": [], "rec": []}
    best_metrics = None
    first_real = None
    first_fake = None
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sums = {k: 0.0 for k in history}
        batches = 0
        for real, _, _ in bundle.train:
            real = real.to(device, non_blocking=True)
            z_real = model.encoder(real)
            z_noise = torch.randn(real.size(0), model.latent_dim, device=device)
            fake = model.generator(z_noise)
            d_real, _ = model.discriminator(real, z_real.detach())
            d_fake, _ = model.discriminator(fake.detach(), z_noise)
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            recon, z_enc = model.reconstruct(real)
            z_noise = torch.randn(real.size(0), model.latent_dim, device=device)
            fake = model.generator(z_noise)
            d_recon, _ = model.discriminator(recon, z_enc)
            d_fake, _ = model.discriminator(fake, z_noise)
            rec = F.l1_loss(recon, real)
            g_loss = 50.0 * rec + bce(d_recon, torch.ones_like(d_recon)) + bce(d_fake, torch.ones_like(d_fake))
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            for key, value in (("g", g_loss), ("d", d_loss), ("rec", rec)):
                sums[key] += float(value.detach().cpu())
            batches += 1
            if first_real is None:
                first_real = real[:8].detach().cpu()
                first_fake = recon[:8].detach().cpu()
        for key in history:
            history[key].append(sums[key] / max(1, batches))
        metrics = None
        score_rows = []
        roc_data = None
        if epoch == 1 or epoch == cfg.epochs or epoch % cfg.eval_every == 0:
            metrics, score_rows, roc_data = _eval_bigan(model, bundle, device)
            metrics["epoch"] = float(epoch)
            if best_metrics is None or metrics["AUC"] > best_metrics["AUC"]:
                best_metrics = metrics
                torch.save({"model": model.state_dict(), "metrics": metrics}, out_dir / "best_model.pt")
        torch.save({"epoch": epoch, "model": model.state_dict()}, out_dir / "latest_model.pt")
        _append_csv(out_dir / "training_history.csv", {"epoch": epoch, "lr_g": opt_g.param_groups[0]["lr"], "lr_d": opt_d.param_groups[0]["lr"], **{k: history[k][-1] for k in history}, "Acc": "" if metrics is None else metrics["Acc"], "F1": "" if metrics is None else metrics["F1"], "AUC": "" if metrics is None else metrics["AUC"]})
        print(f"[{bundle.spec.key}][bigan] epoch {epoch}/{cfg.epochs} g={history['g'][-1]:.4f} d={history['d'][-1]:.4f} " + ("eval=skip" if metrics is None else f"auc={metrics['AUC']:.4f} f1={metrics['F1']:.4f}"))
        sched_g.step()
        sched_d.step()
    assert best_metrics is not None
    _finalize(out_dir, best_metrics, score_rows, roc_data, history, first_real, first_fake, bundle, start)
    return best_metrics


def _train_mtsdvgan_model(model: MTSDVGANModel, opt_g, opt_d, sched_g, sched_d, bundle, cfg, out_dir):
    seed_everything(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.device)
    bce = torch.nn.BCEWithLogitsLoss()
    history = {"g": [], "d": [], "rec": [], "kl": []}
    best_metrics = None
    first_real = None
    first_fake = None
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sums = {k: 0.0 for k in history}
        batches = 0
        for real, _, _ in bundle.train:
            real = real.to(device, non_blocking=True)
            fake, *_ = model(real)
            real_logits, _ = model.discriminator(real)
            fake_logits, _ = model.discriminator(fake.detach())
            d_loss = bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            fake, mu_l, log_l, mu_g, log_g = model(real)
            fake_logits, _ = model.discriminator(fake)
            rec = F.l1_loss(fake, real)
            kl = kl_loss(mu_l, log_l) + kl_loss(mu_g, log_g)
            g_loss = 50.0 * rec + 0.01 * kl + bce(fake_logits, torch.ones_like(fake_logits))
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            for key, value in (("g", g_loss), ("d", d_loss), ("rec", rec), ("kl", kl)):
                sums[key] += float(value.detach().cpu())
            batches += 1
            if first_real is None:
                first_real = real[:8].detach().cpu()
                first_fake = fake[:8].detach().cpu()
        for key in history:
            history[key].append(sums[key] / max(1, batches))
        metrics = None
        score_rows = []
        roc_data = None
        if epoch == 1 or epoch == cfg.epochs or epoch % cfg.eval_every == 0:
            metrics, score_rows, roc_data = _eval_mtsdvgan(model, bundle, device)
            metrics["epoch"] = float(epoch)
            if best_metrics is None or metrics["AUC"] > best_metrics["AUC"]:
                best_metrics = metrics
                torch.save({"model": model.state_dict(), "metrics": metrics}, out_dir / "best_model.pt")
        torch.save({"epoch": epoch, "model": model.state_dict()}, out_dir / "latest_model.pt")
        _append_csv(out_dir / "training_history.csv", {"epoch": epoch, "lr_g": opt_g.param_groups[0]["lr"], "lr_d": opt_d.param_groups[0]["lr"], **{k: history[k][-1] for k in history}, "Acc": "" if metrics is None else metrics["Acc"], "F1": "" if metrics is None else metrics["F1"], "AUC": "" if metrics is None else metrics["AUC"]})
        print(f"[{bundle.spec.key}][mts-dvgan] epoch {epoch}/{cfg.epochs} g={history['g'][-1]:.4f} d={history['d'][-1]:.4f} " + ("eval=skip" if metrics is None else f"auc={metrics['AUC']:.4f} f1={metrics['F1']:.4f}"))
        sched_g.step()
        sched_d.step()
    assert best_metrics is not None
    _finalize(out_dir, best_metrics, score_rows, roc_data, history, first_real, first_fake, bundle, start)
    return best_metrics


def _eval_vae(model, bundle, device):
    model.eval()
    labels, paths, rec_scores, kl_scores = [], [], [], []
    with torch.no_grad():
        for real, label, path in bundle.test:
            real = real.to(device, non_blocking=True)
            fake, mu, logvar = model(real)
            rec = torch.mean(torch.abs(real - fake).flatten(1), dim=1)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            labels.append(label.numpy())
            paths.extend(path)
            rec_scores.append(rec.cpu().numpy())
            kl_scores.append(kl.cpu().numpy())
    return _score_result(paths, labels, {"rec": rec_scores, "kl": kl_scores}, {"rec": 0.8, "kl": 0.2})


def _eval_fanogan(model, bundle, device):
    model.eval()
    labels, paths, rec_scores, lat_scores, disc_scores = [], [], [], [], []
    with torch.no_grad():
        for real, label, path in bundle.test:
            real = real.to(device, non_blocking=True)
            fake, _ = model(real)
            real_logits, real_feat = model.discriminator(real)
            _, fake_feat = model.discriminator(fake)
            labels.append(label.numpy())
            paths.extend(path)
            rec_scores.append(torch.mean(torch.abs(real - fake).flatten(1), dim=1).cpu().numpy())
            lat_scores.append(torch.mean((real_feat - fake_feat).pow(2).flatten(1), dim=1).cpu().numpy())
            disc_scores.append((1.0 - torch.sigmoid(real_logits)).cpu().numpy())
    return _score_result(paths, labels, {"rec": rec_scores, "lat": lat_scores, "disc": disc_scores}, {"rec": 0.5, "lat": 0.3, "disc": 0.2})


def _eval_bigan(model: BiGANModel, bundle, device):
    model.eval()
    labels, paths, rec_scores, disc_scores = [], [], [], []
    with torch.no_grad():
        for real, label, path in bundle.test:
            real = real.to(device, non_blocking=True)
            fake, z = model.reconstruct(real)
            logits, _ = model.discriminator(real, z)
            labels.append(label.numpy())
            paths.extend(path)
            rec_scores.append(torch.mean(torch.abs(real - fake).flatten(1), dim=1).cpu().numpy())
            disc_scores.append((1.0 - torch.sigmoid(logits)).cpu().numpy())
    return _score_result(paths, labels, {"rec": rec_scores, "disc": disc_scores}, {"rec": 0.7, "disc": 0.3})


def _eval_mtsdvgan(model, bundle, device):
    model.eval()
    labels, paths, rec_scores, kl_scores, disc_scores = [], [], [], [], []
    with torch.no_grad():
        for real, label, path in bundle.test:
            real = real.to(device, non_blocking=True)
            fake, mu_l, log_l, mu_g, log_g = model(real)
            logits, _ = model.discriminator(real)
            kl = -0.5 * torch.mean(1.0 + log_l - mu_l.pow(2) - log_l.exp(), dim=1)
            kl = kl - 0.5 * torch.mean(1.0 + log_g - mu_g.pow(2) - log_g.exp(), dim=1)
            labels.append(label.numpy())
            paths.extend(path)
            rec_scores.append(torch.mean(torch.abs(real - fake).flatten(1), dim=1).cpu().numpy())
            kl_scores.append(kl.cpu().numpy())
            disc_scores.append((1.0 - torch.sigmoid(logits)).cpu().numpy())
    return _score_result(paths, labels, {"rec": rec_scores, "kl": kl_scores, "disc": disc_scores}, {"rec": 0.5, "kl": 0.2, "disc": 0.3})


def _score_result(paths, labels, components, weights):
    y = np.concatenate(labels)
    normalized = {name: minmax(np.concatenate(values)) for name, values in components.items()}
    score = np.zeros_like(y, dtype=np.float64)
    for name, weight in weights.items():
        score += weight * normalized[name]
    score = minmax(score)
    metrics = best_balanced_accuracy(y, score)
    fpr, tpr, roc_auc = roc_points(y, score)
    rows = []
    for idx, path in enumerate(paths):
        row = {"path": path, "label": int(y[idx]), "score": float(score[idx])}
        row.update({f"{name}_score": float(values[idx]) for name, values in normalized.items()})
        rows.append(row)
    return metrics, rows, (fpr, tpr, roc_auc)


def _finalize(out_dir, best_metrics, score_rows, roc_data, history, first_real, first_fake, bundle, start_time):
    best_metrics["seconds"] = float(time.time() - start_time)
    best_metrics["train_size"] = float(bundle.train_size)
    best_metrics["test_size"] = float(bundle.test_size)
    save_json(out_dir / "metrics.json", best_metrics)
    save_scores(out_dir / "scores.csv", score_rows)
    if roc_data is not None:
        save_roc_plot(out_dir / "roc_curve.png", roc_data[0], roc_data[1], roc_data[2])
    save_line_plot(out_dir / "loss_curve.png", history, f"{bundle.spec.name} comparison training")
    save_metrics_bar(out_dir / "metrics_bar.png", best_metrics)
    if first_real is not None and first_fake is not None:
        save_reconstruction_grid(out_dir / "reconstruction_grid.png", first_real, first_fake)


def train_comparison_method(method: str, bundle: DataBundle, cfg: ComparisonConfig, out_dir: Path) -> dict[str, float]:
    if method == "if":
        return train_if(bundle, cfg, out_dir)
    if method == "vae":
        return train_vae(bundle, cfg, out_dir)
    if method == "f-anogan":
        return train_fanogan(bundle, cfg, out_dir)
    if method == "bigan":
        return train_bigan(bundle, cfg, out_dir)
    if method == "mts-dvgan":
        return train_mtsdvgan(bundle, cfg, out_dir)
    raise ValueError(f"Unknown comparison method: {method}")
