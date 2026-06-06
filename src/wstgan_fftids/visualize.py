from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def _canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    return image, draw


def save_line_plot(
    path: str | Path,
    series: dict[str, list[float]],
    title: str,
    width: int = 900,
    height: int = 480,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = _canvas(width, height)
    margin = 58
    plot = (margin, margin, width - 24, height - margin)
    draw.rectangle(plot, outline=(45, 55, 70), width=1)
    draw.text((margin, 18), title, fill=(20, 30, 45))

    all_values = [v for values in series.values() for v in values]
    if not all_values:
        image.save(path)
        return
    ymin = min(all_values)
    ymax = max(all_values)
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
    colors = [(37, 99, 235), (220, 38, 38), (5, 150, 105), (147, 51, 234), (234, 88, 12)]
    max_len = max(len(values) for values in series.values())
    for idx, (name, values) in enumerate(series.items()):
        if len(values) < 2:
            continue
        color = colors[idx % len(colors)]
        points = []
        for i, value in enumerate(values):
            x = plot[0] + (plot[2] - plot[0]) * i / max(1, max_len - 1)
            y = plot[3] - (plot[3] - plot[1]) * (value - ymin) / (ymax - ymin)
            points.append((x, y))
        draw.line(points, fill=color, width=2)
        draw.text((plot[0] + 10, plot[1] + 12 + idx * 18), name, fill=color)
    draw.text((8, plot[1] - 4), f"{ymax:.3f}", fill=(80, 80, 80))
    draw.text((8, plot[3] - 10), f"{ymin:.3f}", fill=(80, 80, 80))
    image.save(path)


def save_roc_plot(path: str | Path, fpr: np.ndarray, tpr: np.ndarray, auc: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 620, 560
    image, draw = _canvas(width, height)
    margin = 64
    plot = (margin, 48, width - 34, height - margin)
    draw.rectangle(plot, outline=(45, 55, 70), width=1)
    draw.text((margin, 18), f"ROC Curve  AUC={auc:.4f}", fill=(20, 30, 45))
    draw.line([(plot[0], plot[3]), (plot[2], plot[1])], fill=(150, 150, 150), width=1)
    points = []
    for xval, yval in zip(fpr, tpr):
        x = plot[0] + (plot[2] - plot[0]) * float(xval)
        y = plot[3] - (plot[3] - plot[1]) * float(yval)
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill=(37, 99, 235), width=3)
    draw.text((plot[0], plot[3] + 20), "False Positive Rate", fill=(40, 40, 40))
    draw.text((6, plot[1]), "True Positive Rate", fill=(40, 40, 40))
    image.save(path)


def save_metrics_bar(path: str | Path, metrics: dict[str, float]) -> None:
    selected = ["Acc", "Prec", "Rec", "FAR", "F1", "AUC"]
    width, height = 760, 430
    image, draw = _canvas(width, height)
    margin = 58
    base_y = height - margin
    max_h = height - 130
    draw.text((margin, 20), "IDS Metrics", fill=(20, 30, 45))
    bar_w = 72
    gap = 38
    colors = [(37, 99, 235), (5, 150, 105), (147, 51, 234), (220, 38, 38), (234, 88, 12), (14, 116, 144)]
    for idx, name in enumerate(selected):
        value = float(metrics.get(name, 0.0))
        x0 = margin + idx * (bar_w + gap)
        x1 = x0 + bar_w
        y0 = base_y - max_h * max(0.0, min(1.0, value))
        draw.rectangle((x0, y0, x1, base_y), fill=colors[idx % len(colors)])
        draw.text((x0 + 4, y0 - 18), f"{value:.3f}", fill=(20, 30, 45))
        draw.text((x0 + 10, base_y + 12), name, fill=(20, 30, 45))
    draw.line((margin - 10, base_y, width - margin, base_y), fill=(45, 55, 70), width=1)
    image.save(path)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(-1, 1)
    array = ((tensor + 1.0) * 127.5).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def save_reconstruction_grid(path: str | Path, real: torch.Tensor, fake: torch.Tensor, limit: int = 8) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(limit, real.size(0), fake.size(0))
    if count == 0:
        return
    cell_w = real.size(3)
    cell_h = real.size(2)
    gap = 6
    label_h = 18
    canvas = Image.new("RGB", (count * (cell_w + gap) + gap, 2 * cell_h + label_h + 3 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 2), "real", fill=(20, 30, 45))
    draw.text((gap, cell_h + label_h + gap), "recon", fill=(20, 30, 45))
    for i in range(count):
        x = gap + i * (cell_w + gap)
        canvas.paste(tensor_to_pil(real[i]), (x, label_h))
        canvas.paste(tensor_to_pil(fake[i]), (x, cell_h + label_h + 2 * gap))
    scale = max(1, 256 // max(cell_w, cell_h))
    canvas = canvas.resize((canvas.width * scale, canvas.height * scale), Image.Resampling.NEAREST)
    canvas.save(path)

