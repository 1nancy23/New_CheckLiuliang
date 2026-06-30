from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wstgan_fftids.data import load_dataset_specs, make_dataloaders
from wstgan_fftids.trainer import TrainConfig, train_one_dataset


@dataclass(frozen=True)
class Variant:
    key: str
    family: str
    parameter: str
    value: str
    description: str
    overrides: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-factor-at-a-time parameter study for the proposed IDS model.")
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--dataset", default="first", help="Dataset key, or 'first' for the first configured dataset.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", default="outputs/parameter_study")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--keep-per-run-plots", action="store_true")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--pretrained-root", default="", help="Root directory containing <dataset>/best_model.pt checkpoints.")
    return parser.parse_args()


def pretrained_for_dataset(root: str, dataset_key: str) -> str:
    if not root:
        return ""
    path = Path(root) / dataset_key / "best_model.pt"
    return str(path) if path.exists() else ""


def variants() -> list[Variant]:
    base = {
        "lr": 2e-4,
        "lr_policy": "warmup_cosine",
        "lr_decay_start": 15,
        "warmup_epochs": 4,
        "min_lr_ratio": 0.02,
        "base_channels": 48,
        "beta1": 0.5,
        "fft_low_cutoff": 0.18,
        "fft_mid_cutoff": 0.36,
        "temporal_hidden_ratio": 0.5,
        "cffm_bottleneck_ratio": 1.0,
        "use_fft_prior": True,
        "use_temporal": True,
        "use_st_fusion": True,
        "use_cffm": True,
        "w_adv": 1.0,
        "w_con": 50.0,
        "w_lat": 1.0,
        "w_freq": 0.5,
        "score_alpha": 0.0,
        "score_beta": 0.80,
        "score_gamma": 0.20,
        "score_delta": 0.0,
        "adv_loss": "focal",
        "focal_alpha": 0.35,
        "focal_gamma": 2.0,
        "threshold_objective": "acc",
        "selection_metric": "Acc",
        "grad_clip": 1.0,
        "ema_decay": 0.999,
        "ema_start_epoch": 3,
    }

    rows = [
        Variant("default", "Reference", "Default setting", "default", "Default proposed model configuration.", {}),
        Variant("capacity_c16", "Structure capacity", "base_channels", "16", "Smaller generator/discriminator width.", {"base_channels": 16}),
        Variant("capacity_c24", "Structure capacity", "base_channels", "24", "Moderately smaller network width.", {"base_channels": 24}),
        Variant("capacity_c48", "Structure capacity", "base_channels", "48", "Larger network width.", {"base_channels": 48}),
        Variant("fft_narrow", "Frequency definition", "fft_cutoffs", "0.12/0.30", "Narrower low/mid FFT band split.", {"fft_low_cutoff": 0.12, "fft_mid_cutoff": 0.30}),
        Variant("fft_default_plus", "Frequency definition", "fft_cutoffs", "0.18/0.42", "Wider middle FFT band while preserving the low band.", {"fft_low_cutoff": 0.18, "fft_mid_cutoff": 0.42}),
        Variant("fft_wide_low", "Frequency definition", "fft_cutoffs", "0.24/0.42", "Wider low-frequency region and narrower high-frequency region.", {"fft_low_cutoff": 0.24, "fft_mid_cutoff": 0.42}),
        Variant("temporal_h025", "Temporal structure", "temporal_hidden_ratio", "0.25", "Smaller GRU hidden state in the temporal branch.", {"temporal_hidden_ratio": 0.25}),
        Variant("temporal_h075", "Temporal structure", "temporal_hidden_ratio", "0.75", "Larger GRU hidden state in the temporal branch.", {"temporal_hidden_ratio": 0.75}),
        Variant("temporal_h100", "Temporal structure", "temporal_hidden_ratio", "1.00", "Full-channel GRU hidden state in the temporal branch.", {"temporal_hidden_ratio": 1.0}),
        Variant("cffm_b050", "Fusion structure", "cffm_bottleneck_ratio", "0.50", "More compact CFFM decoder skip fusion.", {"cffm_bottleneck_ratio": 0.5}),
        Variant("cffm_b150", "Fusion structure", "cffm_bottleneck_ratio", "1.50", "Wider CFFM decoder skip fusion.", {"cffm_bottleneck_ratio": 1.5}),
        Variant("no_fft_prior", "Structural switch", "use_fft_prior", "False", "Disable the learnable FFT band prior branch.", {"use_fft_prior": False}),
        Variant("no_temporal", "Structural switch", "use_temporal", "False", "Disable the temporal GRU branch inside spectral-spatial fusion.", {"use_temporal": False}),
        Variant("no_st_fusion", "Structural switch", "use_st_fusion", "False", "Bypass the spectral-spatial-temporal fusion block.", {"use_st_fusion": False}),
        Variant("no_cffm", "Structural switch", "use_cffm", "False", "Remove CFFM decoder skip fusion.", {"use_cffm": False}),
        Variant("adv_0.5", "Training loss", "w_adv", "0.5", "Lower adversarial loss weight.", {"w_adv": 0.5}),
        Variant("adv_2.0", "Training loss", "w_adv", "2.0", "Higher adversarial loss weight.", {"w_adv": 2.0}),
        Variant("recon_25", "Training loss", "w_con", "25", "Lower reconstruction loss weight.", {"w_con": 25.0}),
        Variant("recon_75", "Training loss", "w_con", "75", "Higher reconstruction loss weight.", {"w_con": 75.0}),
        Variant("latent_0.5", "Training loss", "w_lat", "0.5", "Lower latent feature consistency weight.", {"w_lat": 0.5}),
        Variant("latent_2.0", "Training loss", "w_lat", "2.0", "Higher latent feature consistency weight.", {"w_lat": 2.0}),
        Variant("freq_0.0", "Training loss", "w_freq", "0.0", "No frequency consistency loss.", {"w_freq": 0.0}),
        Variant("freq_0.25", "Training loss", "w_freq", "0.25", "Lower frequency consistency loss.", {"w_freq": 0.25}),
        Variant("freq_1.0", "Training loss", "w_freq", "1.0", "Higher frequency consistency loss.", {"w_freq": 1.0}),
        Variant(
            "score_disc",
            "Anomaly score",
            "score_weights",
            "disc-heavy",
            "More emphasis on discriminator confidence.",
            {"score_alpha": 0.55, "score_beta": 0.25, "score_gamma": 0.15, "score_delta": 0.05},
        ),
        Variant(
            "score_recon",
            "Anomaly score",
            "score_weights",
            "recon-heavy",
            "More emphasis on reconstruction error.",
            {"score_alpha": 0.20, "score_beta": 0.55, "score_gamma": 0.15, "score_delta": 0.10},
        ),
        Variant(
            "score_balanced",
            "Anomaly score",
            "score_weights",
            "balanced",
            "Equal contribution from discriminator, reconstruction, latent, and frequency scores.",
            {"score_alpha": 0.25, "score_beta": 0.25, "score_gamma": 0.25, "score_delta": 0.25},
        ),
        Variant(
            "score_freq",
            "Anomaly score",
            "score_weights",
            "freq-aware",
            "More emphasis on frequency-domain reconstruction error.",
            {"score_alpha": 0.25, "score_beta": 0.30, "score_gamma": 0.15, "score_delta": 0.30},
        ),
    ]
    for row in rows:
        merged = dict(base)
        merged.update(row.overrides)
        object.__setattr__(row, "overrides", merged)
    return rows


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cleanup_variant_dir(path: Path, keep_checkpoints: bool, keep_plots: bool) -> None:
    if not keep_checkpoints:
        for item in path.glob("*.pt"):
            item.unlink(missing_ok=True)
    if not keep_plots:
        for name in ("loss_curve.png", "roc_curve.png", "metrics_bar.png", "reconstruction_grid.png"):
            (path / name).unlink(missing_ok=True)


def best_by_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    families = sorted({str(row["family"]) for row in rows})
    for family in families:
        items = [row for row in rows if row["family"] == family]
        best = max(items, key=lambda row: float(row["AUC"]))
        result.append(
            {
                "family": family,
                "best_variant": best["variant"],
                "parameter": best["parameter"],
                "value": best["value"],
                "Acc": best["Acc"],
                "F1": best["F1"],
                "AUC": best["AUC"],
                "FAR": best["FAR"],
                "seconds": best["seconds"],
            }
        )
    return result


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "arialbd.ttf",
        "segoeuib.ttf",
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ) if bold else (
        "arial.ttf",
        "segoeui.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )
    for item in candidates:
        try:
            return ImageFont.truetype(item, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill: tuple[int, int, int], fnt: ImageFont.ImageFont) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, fill=fill, font=fnt)


def save_group_bars(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1900, 1450
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = font(44, True)
    small = font(20)
    label = font(24, True)
    draw.text((70, 46), "Top Parameter Variants on UNSW-NB15", fill=(22, 32, 48), font=title)
    draw.line((70, 110, 825, 110), fill=(213, 94, 0), width=5)

    ranked = sorted(rows, key=lambda r: float(r["AUC"]), reverse=True)[:18]
    plot = (560, 180, 1620, 1240)
    metrics = [("AUC", (0, 114, 178)), ("F1", (0, 158, 115)), ("1-FAR", (213, 94, 0))]
    x_min, x_max = 0.75, 1.00
    for i in range(6):
        value = x_min + (x_max - x_min) * i / 5
        x = plot[0] + (plot[2] - plot[0]) * (value - x_min) / (x_max - x_min)
        draw.line((x, plot[1], x, plot[3]), fill=(232, 236, 242), width=1)
        draw.text((x - 24, plot[3] + 24), f"{value:.2f}", fill=(105, 116, 132), font=small)
    draw.text((plot[0], plot[3] + 65), "Metric value (magnified from 0.75 to 1.00)", fill=(105, 116, 132), font=small)

    row_h = (plot[3] - plot[1]) / len(ranked)
    bar_h = 12
    for idx, row in enumerate(ranked):
        y0 = plot[1] + idx * row_h
        name = str(row["variant"]).replace("_", " ")
        detail = f"{row['family']} | {row['parameter']}={row['value']}"
        draw.text((70, y0 + 4), name, fill=(22, 32, 48), font=label if idx < 5 else small)
        draw.text((70, y0 + 34), detail[:38], fill=(105, 116, 132), font=small)
        values = {"AUC": float(row["AUC"]), "F1": float(row["F1"]), "1-FAR": 1.0 - float(row["FAR"])}
        for m_idx, (metric, color) in enumerate(metrics):
            y = y0 + 8 + m_idx * 18
            value = max(x_min, min(x_max, values[metric]))
            x = plot[0] + (plot[2] - plot[0]) * (value - x_min) / (x_max - x_min)
            draw.rounded_rectangle((plot[0], y, x, y + bar_h), radius=6, fill=color)
            draw.text((x + 10, y - 5), f"{values[metric]:.3f}", fill=(45, 55, 72), font=small)

    legend_x = 1660
    draw.text((legend_x, 210), "Metrics", fill=(22, 32, 48), font=label)
    for idx, (metric, color) in enumerate(metrics):
        y = 265 + idx * 58
        draw.rounded_rectangle((legend_x, y, legend_x + 48, y + 18), radius=9, fill=color)
        draw.text((legend_x + 66, y - 7), metric, fill=(45, 55, 72), font=small)
    draw.text((70, 1340), "Top variants are sorted by AUC. 1-FAR is shown so that all bars follow higher-is-better semantics.", fill=(105, 116, 132), font=small)
    image.save(path, quality=96)


def save_tradeoff(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1800, 1080
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = font(42, True)
    small = font(20)
    label = font(22, True)
    draw.text((70, 46), "Accuracy-Time Trade-off", fill=(22, 32, 48), font=title)
    draw.line((70, 106, 560, 106), fill=(0, 114, 178), width=4)

    plot = (125, 170, 1140, 850)
    draw.rectangle(plot, outline=(226, 232, 240), width=2)
    seconds = np.array([float(r["seconds"]) for r in rows])
    aucs = np.array([float(r["AUC"]) for r in rows])
    performance = np.array(
        [
            (float(r["AUC"]) + float(r["F1"]) + (1.0 - float(r["FAR"]))) / 3.0
            for r in rows
        ],
        dtype=np.float64,
    )
    perf_min, perf_max = float(performance.min()), float(performance.max())
    perf_span = max(1e-9, perf_max - perf_min)
    x_min, x_max = float(seconds.min()), float(seconds.max())
    y_min = max(0.0, float(aucs.min()) - 0.03)
    y_max = min(1.0, float(aucs.max()) + 0.03)
    if x_min == x_max:
        x_max += 1.0
    if y_min == y_max:
        y_max += 0.01

    for i in range(6):
        x = plot[0] + (plot[2] - plot[0]) * i / 5
        y = plot[3] - (plot[3] - plot[1]) * i / 5
        draw.line((x, plot[1], x, plot[3]), fill=(236, 240, 245), width=1)
        draw.line((plot[0], y, plot[2], y), fill=(236, 240, 245), width=1)
        draw.text((x - 30, plot[3] + 22), f"{x_min + (x_max - x_min) * i / 5:.0f}", fill=(105, 116, 132), font=small)
        draw.text((58, y - 12), f"{y_min + (y_max - y_min) * i / 5:.2f}", fill=(105, 116, 132), font=small)

    families = [
        "Reference",
        "Structure capacity",
        "Frequency definition",
        "Temporal structure",
        "Fusion structure",
        "Structural switch",
        "Training loss",
        "Anomaly score",
    ]
    palette = {
        "Reference": (213, 94, 0),
        "Structure capacity": (0, 114, 178),
        "Frequency definition": (86, 80, 161),
        "Temporal structure": (0, 158, 115),
        "Fusion structure": (204, 121, 167),
        "Structural switch": (230, 159, 0),
        "Training loss": (0, 158, 115),
        "Anomaly score": (204, 121, 167),
    }
    for row, perf in zip(rows, performance):
        x = plot[0] + (plot[2] - plot[0]) * (float(row["seconds"]) - x_min) / (x_max - x_min)
        y = plot[3] - (plot[3] - plot[1]) * (float(row["AUC"]) - y_min) / (y_max - y_min)
        radius = 10 + 24 * (float(perf) - perf_min) / perf_span
        color = palette.get(str(row["family"]), (80, 80, 80))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color,), outline=(255, 255, 255), width=3)
        if row["variant"] in {"default"} or float(row["AUC"]) >= float(aucs.max()) - 1e-12:
            text = str(row["variant"])
            box = draw.textbbox((0, 0), text, font=small)
            text_w = box[2] - box[0]
            if x > plot[2] - 90:
                draw.text((x - text_w - 16, y - 18), text, fill=(22, 32, 48), font=small)
            else:
                draw.text((x + 14, y - 18), text, fill=(22, 32, 48), font=small)

    draw_text_center(draw, ((plot[0] + plot[2]) / 2, 930), "Training time (seconds)", (45, 55, 72), label)
    draw.text((22, 142), "AUC", fill=(45, 55, 72), font=label)
    legend_x = 1275
    draw.text((legend_x, 190), "Families", fill=(22, 32, 48), font=label)
    for idx, family in enumerate(families):
        y = 245 + idx * 54
        draw.ellipse((legend_x, y, legend_x + 20, y + 20), fill=palette[family])
        draw.text((legend_x + 38, y - 8), family, fill=(45, 55, 72), font=small)
    draw.text((legend_x, 700), "Point size tracks composite performance:", fill=(105, 116, 132), font=small)
    draw.text((legend_x, 730), "mean(AUC, F1, 1-FAR).", fill=(105, 116, 132), font=small)

    sample_y = 810
    for idx, value in enumerate((perf_min, (perf_min + perf_max) / 2, perf_max)):
        radius = 10 + 24 * (value - perf_min) / perf_span
        x = legend_x + idx * 95 + 25
        draw.ellipse((x - radius, sample_y - radius, x + radius, sample_y + radius), fill=(0, 114, 178), outline=(255, 255, 255), width=3)
        draw_text_center(draw, (x, sample_y + 48), f"{value:.3f}", (105, 116, 132), small)
    image.save(path, quality=96)


def save_parallel_coordinates(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1700, 1850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = font(42, True)
    small = font(19)
    label = font(22, True)
    draw.text((70, 46), "Full Parameter Study Metric Heatmap", fill=(22, 32, 48), font=title)
    draw.line((70, 106, 780, 106), fill=(0, 158, 115), width=5)

    rows = sorted(rows, key=lambda r: (str(r["family"]), -float(r["AUC"])))
    seconds = np.array([float(r["seconds"]) for r in rows])
    sec_min, sec_max = float(seconds.min()), float(seconds.max())
    denom = max(1e-9, sec_max - sec_min)
    metrics = [
        ("Acc", "Accuracy"),
        ("F1", "F1"),
        ("AUC", "AUC"),
        ("FAR_inv", "1-FAR"),
        ("speed", "Speed"),
    ]
    values: list[dict[str, Any]] = []
    for row in rows:
        values.append(
            {
                "variant": row["variant"],
                "family": row["family"],
                "Acc": float(row["Acc"]),
                "F1": float(row["F1"]),
                "AUC": float(row["AUC"]),
                "FAR_inv": 1.0 - float(row["FAR"]),
                "speed": 1.0 - (float(row["seconds"]) - sec_min) / denom,
            }
        )

    def cell_color(value: float) -> tuple[int, int, int]:
        value = max(0.0, min(1.0, value))
        lo = np.array([239, 246, 255], dtype=np.float64)
        hi = np.array([0, 114, 178], dtype=np.float64)
        rgb = lo * (1.0 - value) + hi * value
        return tuple(int(x) for x in rgb)

    left, top = 540, 190
    cell_w, cell_h = 185, 45
    for col, (_, name) in enumerate(metrics):
        x = left + col * cell_w
        draw.rounded_rectangle((x, top - 58, x + cell_w - 10, top - 18), radius=8, fill=(22, 32, 48))
        draw_text_center(draw, (x + (cell_w - 10) / 2, top - 38), name, (255, 255, 255), small)

    last_family = None
    for idx, item in enumerate(values):
        y = top + idx * cell_h
        if item["family"] != last_family:
            draw.text((70, y + 11), str(item["family"]), fill=(22, 32, 48), font=label)
            last_family = item["family"]
        draw.text((335, y + 11), str(item["variant"]).replace("_", " "), fill=(45, 55, 72), font=small)
        for col, (key, _) in enumerate(metrics):
            x = left + col * cell_w
            val = float(item[key])
            draw.rounded_rectangle((x, y, x + cell_w - 10, y + cell_h - 8), radius=7, fill=cell_color(val))
            txt = f"{val:.3f}"
            text_fill = (255, 255, 255) if val >= 0.72 else (38, 50, 68)
            draw_text_center(draw, (x + (cell_w - 10) / 2, y + (cell_h - 8) / 2), txt, text_fill, small)

    legend_x, legend_y = 1505, 230
    draw.text((legend_x, legend_y - 52), "Color scale", fill=(22, 32, 48), font=label)
    for i in range(100):
        val = i / 99
        draw.rectangle((legend_x, legend_y + i * 4, legend_x + 42, legend_y + i * 4 + 4), fill=cell_color(1.0 - val))
    draw.text((legend_x + 60, legend_y - 4), "1.0", fill=(105, 116, 132), font=small)
    draw.text((legend_x + 60, legend_y + 385), "0.0", fill=(105, 116, 132), font=small)
    draw.text((70, 1720), "Speed is normalized as inverted training time; all columns use higher-is-better semantics.", fill=(105, 116, 132), font=small)
    image.save(path, quality=96)


def write_report(path: Path, rows: list[dict[str, Any]], dataset_name: str) -> None:
    ranked = sorted(rows, key=lambda row: float(row["AUC"]), reverse=True)
    best = ranked[0]
    default = next(row for row in rows if row["variant"] == "default")
    lines = [
        f"# Parameter Study Report: {dataset_name}",
        "",
        "This study varies one controllable parameter at a time around the default proposed model configuration.",
        "All runs use the same train/test split, normal-only training, and mixed normal/abnormal testing.",
        "",
        "## Best Variant",
        "",
        f"- Variant: `{best['variant']}`",
        f"- Family: {best['family']}",
        f"- Changed parameter: {best['parameter']} = {best['value']}",
        f"- AUC: {float(best['AUC']):.4f}",
        f"- F1: {float(best['F1']):.4f}",
        f"- FAR: {float(best['FAR']):.4f}",
        "",
        "## Default Reference",
        "",
        f"- AUC: {float(default['AUC']):.4f}",
        f"- F1: {float(default['F1']):.4f}",
        f"- FAR: {float(default['FAR']):.4f}",
        "",
        "## Top 10 Variants",
        "",
        "| Rank | Variant | Family | Parameter | Value | Acc | F1 | AUC | FAR | Seconds |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked[:10], start=1):
        lines.append(
            f"| {rank} | {row['variant']} | {row['family']} | {row['parameter']} | {row['value']} | "
            f"{float(row['Acc']):.4f} | {float(row['F1']):.4f} | {float(row['AUC']):.4f} | "
            f"{float(row['FAR']):.4f} | {float(row['seconds']):.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    specs = load_dataset_specs(args.config)
    dataset_key = next(iter(specs)) if args.dataset == "first" else args.dataset
    if dataset_key not in specs:
        raise KeyError(f"Unknown dataset key: {dataset_key}")
    spec = specs[dataset_key]
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    selected = variants()
    if args.max_variants is not None:
        selected = selected[: args.max_variants]
    (out_root / "variant_plan.json").write_text(json.dumps([asdict(v) for v in selected], indent=2), encoding="utf-8")

    print(f"Loading dataset {spec.name}: {spec.root}", flush=True)
    bundle = make_dataloaders(
        spec,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        seed=args.seed,
        cache_images=args.cache_images,
    )
    print(f"Dataset sizes: train={bundle.train_size}, test={bundle.test_size}", flush=True)
    pretrained_path = pretrained_for_dataset(args.pretrained_root, dataset_key)
    if args.pretrained_root:
        if pretrained_path:
            print(f"Using non-strict pretrained checkpoint for parameter study: {pretrained_path}", flush=True)
        else:
            print(f"No pretrained checkpoint found for dataset {dataset_key} under {args.pretrained_root}; training from scratch.", flush=True)

    summary_path = out_root / "parameter_study_summary.csv"
    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(selected, start=1):
        variant_dir = out_root / "variants" / variant.key
        print(f"[{index}/{len(selected)}] Training {variant.key}: {variant.description}", flush=True)
        cfg = TrainConfig(
            ablation_name=variant.key,
            epochs=args.epochs,
            lr=float(variant.overrides["lr"]),
            beta1=float(variant.overrides["beta1"]),
            w_adv=float(variant.overrides["w_adv"]),
            w_con=float(variant.overrides["w_con"]),
            w_lat=float(variant.overrides["w_lat"]),
            w_freq=float(variant.overrides["w_freq"]),
            score_alpha=float(variant.overrides["score_alpha"]),
            score_beta=float(variant.overrides["score_beta"]),
            score_gamma=float(variant.overrides["score_gamma"]),
            score_delta=float(variant.overrides["score_delta"]),
            base_channels=int(variant.overrides["base_channels"]),
            device=args.device,
            seed=args.seed,
            lr_policy=str(variant.overrides["lr_policy"]),
            lr_decay_start=int(variant.overrides["lr_decay_start"]),
            warmup_epochs=int(variant.overrides["warmup_epochs"]),
            min_lr_ratio=float(variant.overrides["min_lr_ratio"]),
            eval_every=args.eval_every,
            summary_path="",
            use_fft_prior=bool(variant.overrides["use_fft_prior"]),
            use_temporal=bool(variant.overrides["use_temporal"]),
            use_st_fusion=bool(variant.overrides["use_st_fusion"]),
            use_cffm=bool(variant.overrides["use_cffm"]),
            fft_low_cutoff=float(variant.overrides["fft_low_cutoff"]),
            fft_mid_cutoff=float(variant.overrides["fft_mid_cutoff"]),
            temporal_hidden_ratio=float(variant.overrides["temporal_hidden_ratio"]),
            cffm_bottleneck_ratio=float(variant.overrides["cffm_bottleneck_ratio"]),
            adv_loss=str(variant.overrides["adv_loss"]),
            focal_alpha=float(variant.overrides["focal_alpha"]),
            focal_gamma=float(variant.overrides["focal_gamma"]),
            threshold_objective=str(variant.overrides["threshold_objective"]),
            selection_metric=str(variant.overrides["selection_metric"]),
            grad_clip=float(variant.overrides["grad_clip"]),
            ema_decay=float(variant.overrides["ema_decay"]),
            ema_start_epoch=int(variant.overrides["ema_start_epoch"]),
            pretrained_path=pretrained_path,
            pretrained_strict=False if pretrained_path else True,
        )
        started = time.time()
        metrics = train_one_dataset(bundle, cfg, variant_dir)
        row = {
            "dataset": dataset_key,
            "dataset_name": spec.name,
            "variant": variant.key,
            "family": variant.family,
            "parameter": variant.parameter,
            "value": variant.value,
            "description": variant.description,
            "out_dir": str(variant_dir),
            "epoch": int(metrics["epoch"]),
            "Threshold": metrics["Threshold"],
            "Acc": metrics["Acc"],
            "Prec": metrics["Prec"],
            "Rec": metrics["Rec"],
            "FAR": metrics["FAR"],
            "F1": metrics["F1"],
            "AUC": metrics["AUC"],
            "seconds": metrics["seconds"],
            "wall_seconds": time.time() - started,
            "train_size": int(metrics["train_size"]),
            "test_size": int(metrics["test_size"]),
            **{f"cfg_{key}": value for key, value in variant.overrides.items()},
        }
        rows.append(row)
        append_csv(summary_path, row)
        cleanup_variant_dir(variant_dir, args.keep_checkpoints, args.keep_per_run_plots)

    ranked = sorted(rows, key=lambda row: float(row["AUC"]), reverse=True)
    write_csv(out_root / "parameter_study_ranking.csv", ranked)
    write_csv(out_root / "parameter_study_best_by_family.csv", best_by_family(rows))
    write_report(out_root / "PARAMETER_STUDY_REPORT.md", rows, spec.name)
    save_group_bars(out_root / "parameter_study_group_bars.png", rows)
    save_tradeoff(out_root / "parameter_study_tradeoff.png", rows)
    save_parallel_coordinates(out_root / "parameter_study_metric_heatmap.png", rows)
    shutil.copy2(summary_path, out_root / "parameter_study_summary_final.csv")
    print(f"Wrote parameter study outputs to {out_root}", flush=True)


if __name__ == "__main__":
    main()
