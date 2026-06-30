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


def best_threshold_metrics(labels: np.ndarray, scores: np.ndarray, objective: str = "ba") -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), scores]))
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    pos_prefix = np.concatenate([[0], np.cumsum(sorted_labels == 1)])
    neg_prefix = np.concatenate([[0], np.cumsum(sorted_labels == 0)])
    total_pos = int(pos_prefix[-1])
    total_neg = int(neg_prefix[-1])

    first_positive = np.searchsorted(sorted_scores, thresholds, side="left")
    tp = total_pos - pos_prefix[first_positive]
    fp = total_neg - neg_prefix[first_positive]
    fn = total_pos - tp
    tn = total_neg - fp

    tpr = np.divide(tp, total_pos, out=np.zeros_like(tp, dtype=np.float64), where=total_pos != 0)
    tnr = np.divide(tn, total_neg, out=np.zeros_like(tn, dtype=np.float64), where=total_neg != 0)
    ba = 0.5 * (tpr + tnr)
    precision_all = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) != 0)
    recall_all = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) != 0)
    f1_all = np.divide(
        2.0 * precision_all * recall_all,
        precision_all + recall_all,
        out=np.zeros_like(precision_all, dtype=np.float64),
        where=(precision_all + recall_all) != 0,
    )
    acc_all = np.divide(tp + tn, np.maximum(1, tp + tn + fp + fn))
    if objective == "ba":
        selection = ba
    elif objective == "f1":
        selection = f1_all
    elif objective == "acc":
        selection = acc_all
    elif objective == "f1_acc":
        selection = 0.5 * f1_all + 0.5 * acc_all
    else:
        raise ValueError(f"Unsupported threshold objective: {objective}")
    best_index = int(np.argmax(selection))

    best_tp = int(tp[best_index])
    best_tn = int(tn[best_index])
    best_fp = int(fp[best_index])
    best_fn = int(fn[best_index])
    precision = best_tp / (best_tp + best_fp) if best_tp + best_fp else 0.0
    recall = best_tp / (best_tp + best_fn) if best_tp + best_fn else 0.0
    far = best_fp / (best_fp + best_tn) if best_fp + best_tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (best_tp + best_tn) / max(1, best_tp + best_tn + best_fp + best_fn)
    best = {
        "Threshold": float(thresholds[best_index]),
        "BA": float(ba[best_index]),
        "Acc": float(acc),
        "Prec": float(precision),
        "Rec": float(recall),
        "FAR": float(far),
        "F1": float(f1),
        "SelectionScore": float(selection[best_index]),
        "ThresholdObjective": objective,
        "TP": float(best_tp),
        "TN": float(best_tn),
        "FP": float(best_fp),
        "FN": float(best_fn),
    }
    try:
        best["AUC"] = float(roc_auc_score(labels, scores))
    except ValueError:
        best["AUC"] = 0.0
    return best


def best_balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return best_threshold_metrics(labels, scores, objective="ba")


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
