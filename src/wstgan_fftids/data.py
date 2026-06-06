from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    root: Path
    image_size: int
    description: str = ""


def load_dataset_specs(config_path: str | Path) -> dict[str, DatasetSpec]:
    config_path = Path(config_path)
    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    specs: dict[str, DatasetSpec] = {}
    for key, item in raw.items():
        specs[key] = DatasetSpec(
            key=key,
            name=item.get("name", key),
            root=Path(item["root"]),
            image_size=int(item.get("image_size", 16)),
            description=item.get("description", ""),
        )
    return specs


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def class_label(path: Path) -> int:
    parent = path.parent.name.lower()
    return 1 if parent in {"abnormal", "attack", "anomaly", "malicious"} else 0


class TrafficImageDataset(Dataset):
    """Image dataset with fixed IDS labels: normal=0, abnormal=1."""

    def __init__(
        self,
        split_root: str | Path,
        image_size: int = 16,
        train_normal_only: bool = False,
        max_samples: int | None = None,
        seed: int = 3407,
    ) -> None:
        self.split_root = Path(split_root)
        paths = list_images(self.split_root)
        if train_normal_only:
            paths = [p for p in paths if class_label(p) == 0]
        if max_samples is not None and len(paths) > max_samples:
            rng = random.Random(seed)
            normal = [p for p in paths if class_label(p) == 0]
            abnormal = [p for p in paths if class_label(p) == 1]
            if normal and abnormal:
                half = max_samples // 2
                selected = rng.sample(normal, min(len(normal), max_samples - min(len(abnormal), half)))
                selected += rng.sample(abnormal, min(len(abnormal), max_samples - len(selected)))
                while len(selected) < max_samples and len(selected) < len(paths):
                    p = rng.choice(paths)
                    if p not in selected:
                        selected.append(p)
                paths = sorted(selected)
            else:
                paths = sorted(rng.sample(paths, max_samples))
        if not paths:
            raise RuntimeError(f"No images found under {self.split_root}")

        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        label = torch.tensor(class_label(path), dtype=torch.long)
        return tensor, label, str(path)


@dataclass
class DataBundle:
    train: DataLoader
    test: DataLoader
    spec: DatasetSpec
    train_size: int
    test_size: int


def make_dataloaders(
    spec: DatasetSpec,
    batch_size: int,
    workers: int,
    image_size: int | None = None,
    max_train: int | None = None,
    max_test: int | None = None,
    seed: int = 3407,
) -> DataBundle:
    size = image_size or spec.image_size
    train_ds = TrafficImageDataset(
        spec.root / "train",
        image_size=size,
        train_normal_only=True,
        max_samples=max_train,
        seed=seed,
    )
    test_ds = TrafficImageDataset(
        spec.root / "test",
        image_size=size,
        train_normal_only=False,
        max_samples=max_test,
        seed=seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(train_ds) >= batch_size else False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return DataBundle(train_loader, test_loader, spec, len(train_ds), len(test_ds))


def count_split(root: str | Path) -> dict[str, int]:
    root = Path(root)
    counts = {}
    for split in ("train", "test"):
        for cls in ("normal", "abnormal"):
            counts[f"{split}_{cls}"] = len(list_images(root / split / cls))
    return counts


def write_manifest(specs: Iterable[DatasetSpec], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        counts = count_split(spec.root)
        rows.append(
            {
                "key": spec.key,
                "name": spec.name,
                "root": str(spec.root),
                "image_size": spec.image_size,
                **counts,
            }
        )
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
