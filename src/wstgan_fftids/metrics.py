from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_auc_score, roc_curve


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def best_balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), scores]))
    best: dict[str, float] | None = None
    for threshold in thresholds:
        pred = (scores >= threshold).astype(np.int64)
        tp = int(np.sum((labels == 1) & (pred == 1)))
        tn = int(np.sum((labels == 0) & (pred == 0)))
        fp = int(np.sum((labels == 0) & (pred == 1)))
        fn = int(np.sum((labels == 1) & (pred == 0)))
        tpr = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        ba = 0.5 * (tpr + tnr)
        if best is None or ba > best["BA"]:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tpr
            far = fp / (fp + tn) if fp + tn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            acc = (tp + tn) / max(1, tp + tn + fp + fn)
            best = {
                "Threshold": float(threshold),
                "BA": float(ba),
                "Acc": float(acc),
                "Prec": float(precision),
                "Rec": float(recall),
                "FAR": float(far),
                "F1": float(f1),
                "TP": float(tp),
                "TN": float(tn),
                "FP": float(fp),
                "FN": float(fn),
            }
    assert best is not None
    try:
        best["AUC"] = float(roc_auc_score(labels, scores))
    except ValueError:
        best["AUC"] = 0.0
    return best


def roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return fpr, tpr, float(sk_auc(fpr, tpr))
    except ValueError:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.0


def save_scores(path: str | Path, rows: list[dict[str, float | int | str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

