from __future__ import annotations

import argparse
import csv
from pathlib import Path

from noise_robustness import draw_radar_chart, radar_values, write_validation_summary


DATASET_TITLES = {
    "unsw": "UNSW-NB15",
    "cic": "CIC-IDS2017",
    "toniot": "TON_IoT",
}


def read_rows(path: Path) -> list[dict[str, str | float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw one noise robustness radar chart per dataset.")
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--proposed-summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="radar_noise")
    args = parser.parse_args()

    baseline_rows = read_rows(args.baseline_summary)
    proposed_rows = [row for row in read_rows(args.proposed_summary) if row["method_key"] == "proposed"]
    merged = [row for row in baseline_rows if row["method_key"] != "proposed"] + proposed_rows

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "noise_robustness_merged_latest.csv", merged)
    write_validation_summary(args.out_dir / "noise_robustness_validation_latest.csv", merged)

    axes = ["Clean AUC", "Mean Noisy AUC", "Worst Noisy AUC", "Mean Noisy F1", "Transfer F1", "Rank Stability"]
    for dataset_key, title in DATASET_TITLES.items():
        draw_radar_chart(
            args.out_dir / f"{args.prefix}_{dataset_key}.png",
            f"{title} Noise Robustness",
            axes,
            radar_values(merged, dataset_key),
        )


if __name__ == "__main__":
    main()
