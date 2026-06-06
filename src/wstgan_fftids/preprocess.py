from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import QuantileTransformer


def _encode_mixed(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if out[column].dtype == "object" or str(out[column].dtype).startswith("string"):
            out[column] = out[column].fillna("NA")
            out[column], _ = pd.factorize(out[column], sort=True)
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out[columns] = out[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _labels_to_binary(values: pd.Series, normal_label: str) -> np.ndarray:
    normal = str(normal_label)
    return np.array([0 if str(v) == normal else 1 for v in values], dtype=np.int64)


def _feature_order(train_normal: pd.DataFrame, columns: list[str]) -> list[str]:
    keep = []
    const = []
    for column in columns:
        values = train_normal[column].to_numpy(dtype=np.float64)
        if np.nanvar(values) > 1e-12:
            keep.append(column)
        else:
            const.append(column)
    if len(keep) < 2:
        return columns
    corr = np.corrcoef(train_normal[keep].to_numpy(dtype=np.float64), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    return [keep[i] for i in order] + const


def _row_to_image(row: np.ndarray, height: int, width: int) -> np.ndarray:
    pixels = height * width
    values = row[:pixels]
    if values.shape[0] < pixels:
        values = np.pad(values, (0, pixels - values.shape[0]), mode="constant", constant_values=0.0)
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    return (values.reshape(height, width) * 255.0).round().astype(np.uint8)


def _save_triplets(
    x: np.ndarray,
    y: np.ndarray,
    out_split: Path,
    prefix: str,
    height: int,
    width: int,
    stride: int,
) -> dict[str, int]:
    for cls in ("normal", "abnormal"):
        (out_split / cls).mkdir(parents=True, exist_ok=True)
    counts = {"normal": 0, "abnormal": 0}
    saved = 0
    for start in range(0, x.shape[0] - 2, stride):
        block = x[start : start + 3]
        labels = y[start : start + 3]
        cls = "abnormal" if int(np.any(labels == 1)) else "normal"
        rgb = np.stack(
            [
                _row_to_image(block[0], height, width),
                _row_to_image(block[1], height, width),
                _row_to_image(block[2], height, width),
            ],
            axis=-1,
        )
        Image.fromarray(rgb).save(out_split / cls / f"{prefix}_{saved:08d}.png")
        counts[cls] += 1
        saved += 1
    return counts


def csv_to_rgb_dataset(
    train_csv: Path,
    test_csv: Path,
    out_root: Path,
    label_col: str,
    height: int,
    width: int,
    drop_cols: list[str],
    normal_label: str = "0",
    stride: int = 3,
    use_quantile: bool = True,
) -> dict[str, Any]:
    train_df = pd.read_csv(train_csv, low_memory=False)
    test_df = pd.read_csv(test_csv, low_memory=False)
    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise KeyError(f"Label column not found: {label_col}")

    feature_cols = [c for c in train_df.columns if c != label_col and c not in drop_cols]
    feature_cols = [c for c in feature_cols if c in test_df.columns and not train_df[c].isna().all()]
    train_df = _encode_mixed(train_df, feature_cols)
    test_df = _encode_mixed(test_df, feature_cols)
    y_train = _labels_to_binary(train_df[label_col], normal_label)
    y_test = _labels_to_binary(test_df[label_col], normal_label)

    train_normal = train_df.loc[y_train == 0, feature_cols]
    if len(train_normal) < 10:
        raise ValueError("Need at least 10 normal training rows to estimate correlation layout.")
    ordered = _feature_order(train_normal, feature_cols)
    pixels = height * width
    if len(ordered) > pixels:
        ordered = ordered[:pixels]

    if use_quantile:
        qt = QuantileTransformer(
            n_quantiles=min(1000, max(10, len(train_df))),
            output_distribution="uniform",
            random_state=3407,
            subsample=int(1e9),
        )
        x_train = qt.fit_transform(train_df[ordered].to_numpy(dtype=np.float64))
        x_test = qt.transform(test_df[ordered].to_numpy(dtype=np.float64))
    else:
        train_values = train_df[ordered].to_numpy(dtype=np.float64)
        test_values = test_df[ordered].to_numpy(dtype=np.float64)
        lo = np.nanmin(train_values, axis=0)
        hi = np.nanmax(train_values, axis=0)
        span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
        x_train = (train_values - lo) / span
        x_test = (test_values - lo) / span

    out_root.mkdir(parents=True, exist_ok=True)
    train_counts = _save_triplets(x_train[y_train == 0], y_train[y_train == 0], out_root / "train", "train", height, width, stride)
    test_counts = _save_triplets(x_test, y_test, out_root / "test", "test", height, width, stride)
    meta = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "out_root": str(out_root),
        "height": height,
        "width": width,
        "features": ordered,
        "train_counts": train_counts,
        "test_counts": test_counts,
    }
    (out_root / "layout_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta

