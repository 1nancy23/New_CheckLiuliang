from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wstgan_fftids.data import list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 4-channel positional-encoding traffic images from RGB traffic images.")
    parser.add_argument("--src-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--position-weight", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def position_map(height: int, width: int, weight: float) -> tuple[np.ndarray, np.ndarray]:
    pixels = height * width
    phase = np.linspace(0.0, 2.0 * np.pi, pixels, dtype=np.float32)
    pos = np.sin(phase).reshape(height, width) * float(weight)
    if abs(weight) < 1e-12:
        pos_channel = np.full((height, width), 0.5, dtype=np.float32)
    else:
        pos_channel = pos / (2.0 * abs(float(weight))) + 0.5
    return pos, np.clip(pos_channel, 0.0, 1.0)


def convert_one(src_path: Path, src_root: Path, out_root: Path, weight: float) -> None:
    rel = src_path.relative_to(src_root)
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    pos, pos_channel = position_map(rgb.shape[0], rgb.shape[1], weight)
    enhanced = np.clip(rgb + pos[:, :, None], 0.0, 1.0)
    rgba = np.concatenate([enhanced, pos_channel[:, :, None]], axis=2)
    Image.fromarray((rgba * 255.0).round().astype(np.uint8), mode="RGBA").save(out_path)


def main() -> None:
    args = parse_args()
    paths = list_images(args.src_root)
    if not paths:
        raise RuntimeError(f"No images found under {args.src_root}")
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(lambda p: convert_one(p, args.src_root, args.out_root, args.position_weight), paths))
    else:
        for path in paths:
            convert_one(path, args.src_root, args.out_root, args.position_weight)
    meta = {
        "source_root": str(args.src_root),
        "out_root": str(args.out_root),
        "channels": 4,
        "position_encoding": "sinusoidal_0_to_2pi",
        "position_weight": args.position_weight,
        "channel_definition": [
            "rgb_channel_0 + position_weight * sin(position)",
            "rgb_channel_1 + position_weight * sin(position)",
            "rgb_channel_2 + position_weight * sin(position)",
            "position_weight * sin(position), rescaled to image range",
        ],
        "image_count": len(paths),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "layout_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
