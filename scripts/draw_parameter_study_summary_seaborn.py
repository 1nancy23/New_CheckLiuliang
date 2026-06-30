from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib import gridspec


FAMILY_PALETTE = {
    "Reference": "#8C564B",
    "Structure capacity": "#2F80ED",
    "Frequency definition": "#7E57C2",
    "Temporal structure": "#00A676",
    "Fusion structure": "#D45087",
    "Structural switch": "#F2A900",
    "Training loss": "#0E7C7B",
    "Anomaly score": "#C43E55",
}

METRIC_PALETTE = {
    "AUC": "#0072B2",
    "F1": "#009E73",
    "1-FAR": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw publication-style parameter-study summary figures.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def prepare(summary: Path) -> pd.DataFrame:
    df = pd.read_csv(summary)
    df["1-FAR"] = 1.0 - df["FAR"]
    df["Composite"] = (df["AUC"] + df["F1"] + df["1-FAR"]) / 3.0
    seconds = df["seconds"].astype(float)
    sec_span = max(1e-9, float(seconds.max() - seconds.min()))
    df["Speed"] = 1.0 - (seconds - seconds.min()) / sec_span
    df["Variant"] = df["variant"].str.replace("_", " ", regex=False)
    df["Setting"] = df["parameter"].astype(str) + "=" + df["value"].astype(str)
    return df


def save_top_bars(df: pd.DataFrame, out: Path) -> None:
    top = df.sort_values("AUC", ascending=False).head(18).copy()
    top["VariantLabel"] = top["Variant"] + "\n" + top["family"] + " | " + top["Setting"]
    order = top["VariantLabel"].tolist()
    long = top.melt(
        id_vars=["VariantLabel", "family"],
        value_vars=["AUC", "F1", "1-FAR"],
        var_name="Metric",
        value_name="Value",
    )

    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
    fig, ax = plt.subplots(figsize=(14.8, 10.6), dpi=320)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFE")

    sns.barplot(
        data=long,
        y="VariantLabel",
        x="Value",
        hue="Metric",
        order=order,
        hue_order=["AUC", "F1", "1-FAR"],
        palette=METRIC_PALETTE,
        orient="h",
        ax=ax,
        width=0.72,
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_title("Top Parameter Variants on UNSW-NB15", loc="left", fontsize=18, fontweight="bold", color="#111827", pad=18)
    ax.text(
        0.0,
        1.01,
        "Top variants are ranked by AUC. 1-FAR is included so all displayed metrics follow higher-is-better semantics.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#64748B",
        va="bottom",
    )
    ax.set_xlim(0.74, 1.0)
    ax.set_xlabel("Metric value", fontsize=12, fontweight="bold", color="#1F2937", labelpad=12)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8.8)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", color="#E5EAF2", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True, bottom=False)

    for container in ax.containers:
        labels = [f"{bar.get_width():.3f}" if bar.get_width() >= 0.80 else "" for bar in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=7.5, color="#374151")

    legend = ax.legend(
        title="Metric",
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        borderpad=0.9,
    )
    legend.get_frame().set_edgecolor("#E5EAF2")
    legend.get_frame().set_facecolor("white")

    fig.subplots_adjust(left=0.31, right=0.98, top=0.90, bottom=0.10)
    fig.savefig(out, dpi=320, facecolor="white")
    plt.close(fig)


def value_family_color(base_color: str, value: float) -> tuple[float, float, float]:
    norm = float(np.clip(value, 0.0, 1.0))
    base = np.array(mcolors.to_rgb(base_color))
    white = np.array(mcolors.to_rgb("#F8FAFC"))
    strength = 0.22 + 0.72 * norm
    return tuple(white * (1.0 - strength) + base * strength)


def readable_text_color(rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.52 else "#1F2937"


def save_metric_heatmap(df: pd.DataFrame, out: Path) -> None:
    df = df.sort_values(["family", "AUC"], ascending=[True, False]).copy()
    family_order = df["family"].drop_duplicates().tolist()
    row_labels = df["Variant"].tolist()
    metric_df = df[["Acc", "F1", "AUC", "1-FAR", "Speed"]].copy()
    metric_df.columns = ["Accuracy", "F1", "AUC", "1-FAR", "Speed"]
    metric_df.index = row_labels

    sns.set_theme(style="white", context="paper", font="DejaVu Sans")
    fig = plt.figure(figsize=(15.6, 15.4), dpi=320, facecolor="white")
    gs = gridspec.GridSpec(
        1,
        3,
        width_ratios=[0.48, 1.0, 0.30],
        wspace=0.05,
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.06,
    )
    strip_ax = fig.add_subplot(gs[0, 0])
    heat_ax = fig.add_subplot(gs[0, 1])
    legend_ax = fig.add_subplot(gs[0, 2])

    n_rows, n_cols = metric_df.shape
    heat_ax.set_xlim(0, n_cols)
    heat_ax.set_ylim(n_rows, 0)
    heat_ax.set_facecolor("#F8FAFC")
    for row_idx, (_, row) in enumerate(df.iterrows()):
        family_color = FAMILY_PALETTE.get(row["family"], "#6B7280")
        for col_idx, metric in enumerate(metric_df.columns):
            value = float(metric_df.iloc[row_idx, col_idx])
            face_color = value_family_color(family_color, value)
            heat_ax.add_patch(
                plt.Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    facecolor=face_color,
                    edgecolor="white",
                    linewidth=1.6,
                )
            )
            heat_ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8.4,
                color=readable_text_color(face_color),
            )
    heat_ax.set_xticks(np.arange(n_cols) + 0.5)
    heat_ax.set_xticklabels(metric_df.columns, rotation=0, fontsize=10)
    heat_ax.set_yticks([])
    heat_ax.tick_params(axis="x", length=0, pad=10)
    heat_ax.tick_params(axis="y", length=0)
    for spine in heat_ax.spines.values():
        spine.set_visible(False)
    heat_ax.set_title("Full Parameter Study Metric Heatmap", loc="left", fontsize=18, fontweight="bold", color="#111827", pad=50)
    heat_ax.text(
        0.0,
        1.03,
        "Cell hue follows the parameter family legend; darker shade indicates a higher metric value.",
        transform=heat_ax.transAxes,
        fontsize=10.5,
        color="#64748B",
        va="bottom",
    )

    strip_ax.set_xlim(0, 1)
    strip_ax.set_ylim(len(df), 0)
    strip_ax.axis("off")
    last_family = None
    for idx, (_, row) in enumerate(df.iterrows()):
        color = FAMILY_PALETTE.get(row["family"], "#6B7280")
        strip_ax.add_patch(
            plt.Rectangle(
                (0.42, idx + 0.05),
                0.035,
                0.90,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                transform=strip_ax.transData,
                clip_on=False,
            )
        )
        if row["family"] != last_family:
            strip_ax.text(0.0, idx + 0.62, row["family"], ha="left", va="center", fontsize=9.0, fontweight="bold", color="#111827")
            last_family = row["family"]
        strip_ax.text(0.50, idx + 0.62, row["Variant"], ha="left", va="center", fontsize=8.8, color="#374151")

    legend_ax.axis("off")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.text(0.0, 0.98, "Parameter family", fontsize=11.5, color="#111827", va="top")
    y = 0.92
    for family in family_order:
        color = FAMILY_PALETTE.get(family, "#6B7280")
        legend_ax.scatter([0.04], [y], s=72, color=color, edgecolors="white", linewidths=0.8)
        legend_ax.text(0.14, y, family, fontsize=9.8, color="#374151", va="center")
        y -= 0.062
    legend_ax.text(
        0.0,
        0.38,
        "Color encoding",
        fontsize=11.5,
        color="#111827",
        va="top",
    )
    legend_ax.text(
        0.0,
        0.33,
        "Hue identifies the row's\nparameter family. Shade\nintensity encodes the metric\nvalue; darker is better.",
        fontsize=9.4,
        color="#64748B",
        va="top",
        linespacing=1.45,
    )
    sample_color = FAMILY_PALETTE.get(family_order[0], "#6B7280") if family_order else "#6B7280"
    for x, value, label in [(0.04, 0.35, "Low"), (0.32, 0.70, "Mid"), (0.60, 0.95, "High")]:
        legend_ax.add_patch(
            plt.Rectangle(
                (x, 0.17),
                0.18,
                0.045,
                facecolor=value_family_color(sample_color, value),
                edgecolor="white",
                linewidth=0.8,
            )
        )
        legend_ax.text(x + 0.09, 0.14, label, ha="center", va="top", fontsize=8.6, color="#64748B")
    legend_ax.text(
        0.0,
        0.09,
        "Metric columns",
        fontsize=11.5,
        color="#111827",
        va="top",
    )
    legend_ax.text(
        0.0,
        0.04,
        "Accuracy, F1, AUC, 1-FAR,\nand normalized speed all use\nhigher-is-better scaling.",
        fontsize=9.4,
        color="#64748B",
        va="top",
        linespacing=1.45,
    )

    fig.savefig(out, dpi=320, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare(args.summary)
    save_top_bars(df, args.out_dir / "parameter_study_group_bars.png")
    save_metric_heatmap(df, args.out_dir / "parameter_study_metric_heatmap.png")


if __name__ == "__main__":
    main()
