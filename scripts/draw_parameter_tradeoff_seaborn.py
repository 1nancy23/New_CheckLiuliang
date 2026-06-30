from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = {
    "Reference": "#D55E00",
    "Structure capacity": "#0072B2",
    "Frequency definition": "#6A51A3",
    "Temporal structure": "#009E73",
    "Fusion structure": "#CC79A7",
    "Structural switch": "#E69F00",
    "Training loss": "#1B9E77",
    "Anomaly score": "#B65C8A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a publication-style parameter-study trade-off plot.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="Accuracy-Time Trade-off on UNSW-NB15")
    return parser.parse_args()


def marker_sizes(values: pd.Series) -> pd.Series:
    lo = float(values.min())
    hi = float(values.max())
    if np.isclose(lo, hi):
        return pd.Series(np.full(len(values), 420.0), index=values.index)
    norm = (values - lo) / (hi - lo)
    return 110.0 + 780.0 * np.power(norm, 1.15)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)
    df["Composite"] = (df["AUC"] + df["F1"] + (1.0 - df["FAR"])) / 3.0
    df["MarkerSize"] = marker_sizes(df["Composite"])
    df["VariantLabel"] = df["variant"].str.replace("_", " ", regex=False)

    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
    fig, ax = plt.subplots(figsize=(16.4, 8.8), dpi=320)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFE")

    default = df.loc[df["variant"].eq("default")].iloc[0]
    ax.axhline(default["AUC"], color="#94A3B8", linestyle=(0, (4, 4)), linewidth=1.0, zorder=0)
    ax.axvline(default["seconds"], color="#CBD5E1", linestyle=(0, (4, 4)), linewidth=1.0, zorder=0)

    for family, group in df.groupby("family", sort=False):
        ax.scatter(
            group["seconds"],
            group["AUC"],
            s=group["MarkerSize"],
            c=PALETTE.get(family, "#4B5563"),
            label=family,
            alpha=0.88,
            edgecolors="white",
            linewidths=1.25,
            zorder=3,
        )

    top = df.sort_values("AUC", ascending=False).head(5)
    labels = set(top["variant"]).union({"default"})
    offsets = {
        "capacity_c48": (-22, 8, "right"),
        "temporal_h100": (18, 19, "left"),
        "recon_25": (18, -22, "left"),
        "temporal_h075": (22, 31, "left"),
        "latent_2.0": (18, -36, "left"),
        "default": (22, -26, "left"),
    }
    for _, row in df[df["variant"].isin(labels)].iterrows():
        dx, dy, align = offsets.get(row["variant"], (10, 8, "left"))
        ax.annotate(
            row["VariantLabel"],
            xy=(row["seconds"], row["AUC"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=9.5,
            color="#111827",
            arrowprops={
                "arrowstyle": "-",
                "color": "#94A3B8",
                "lw": 0.9,
                "shrinkA": 2,
                "shrinkB": 3,
            },
            zorder=5,
        )

    ax.set_title(args.title, loc="left", fontsize=18, fontweight="bold", color="#111827", pad=18)
    ax.text(
        0.0,
        1.01,
        "Circle size encodes composite performance: mean(AUC, F1, 1-FAR). Dashed lines mark the default setting.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#64748B",
        va="bottom",
    )
    ax.set_xlabel("Training time (seconds)", fontsize=12, fontweight="bold", color="#1F2937", labelpad=12)
    ax.set_ylabel("AUC", fontsize=12, fontweight="bold", color="#1F2937", labelpad=12)

    x_pad = (df["seconds"].max() - df["seconds"].min()) * 0.08
    y_pad = (df["AUC"].max() - df["AUC"].min()) * 0.12
    ax.set_xlim(df["seconds"].min() - x_pad, df["seconds"].max() + x_pad * 1.15)
    ax.set_ylim(max(0.0, df["AUC"].min() - y_pad), min(1.0, df["AUC"].max() + y_pad))
    ax.grid(True, color="#E5EAF2", linewidth=0.9)
    sns.despine(ax=ax, left=False, bottom=False)

    fig.subplots_adjust(left=0.075, right=0.74, top=0.86, bottom=0.13)

    legend_ax = fig.add_axes([0.785, 0.16, 0.20, 0.72])
    legend_ax.set_axis_off()
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.text(0.0, 0.98, "Parameter family", fontsize=11.5, color="#111827", va="top")
    y = 0.92
    for family, color in PALETTE.items():
        if family not in set(df["family"]):
            continue
        legend_ax.scatter([0.035], [y], s=72, color=color, edgecolors="white", linewidths=0.8)
        legend_ax.text(0.12, y, family, fontsize=10.5, color="#374151", va="center")
        y -= 0.064

    legend_ax.text(0.0, 0.28, "Composite score", fontsize=11.5, color="#111827", va="top")
    comp_values = [df["Composite"].min(), df["Composite"].median(), df["Composite"].max()]
    y = 0.20
    for value in comp_values:
        size = marker_sizes(pd.Series([value], index=[0])).iloc[0] * 0.28
        legend_ax.scatter([0.05], [y], s=size, color="#0072B2", edgecolors="white", linewidths=0.8, alpha=0.85)
        legend_ax.text(0.18, y, f"{value:.3f}", fontsize=10.5, color="#374151", va="center")
        y -= 0.06

    fig.savefig(args.out, dpi=320, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
