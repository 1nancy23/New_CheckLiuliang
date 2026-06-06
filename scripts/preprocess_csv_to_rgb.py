from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wstgan_fftids.preprocess import csv_to_rgb_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert IDS CSV splits into correlation-guided RGB traffic images.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--drop-cols", default="", help="Comma-separated non-feature columns to drop.")
    parser.add_argument("--normal-label", default="0")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--no-quantile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    meta = csv_to_rgb_dataset(
        train_csv=Path(args.train_csv),
        test_csv=Path(args.test_csv),
        out_root=Path(args.out_root),
        label_col=args.label_col,
        height=args.height,
        width=args.width,
        drop_cols=drop_cols,
        normal_label=args.normal_label,
        stride=args.stride,
        use_quantile=not args.no_quantile,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

