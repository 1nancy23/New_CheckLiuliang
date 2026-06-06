from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wstgan_fftids.data import load_dataset_specs, write_manifest


def main() -> None:
    specs = load_dataset_specs(ROOT / "configs" / "datasets.json")
    missing = []
    for key, spec in specs.items():
        if not spec.root.exists():
            missing.append(f"{key}: {spec.root}")
    if missing:
        raise FileNotFoundError("Missing dataset roots:\n" + "\n".join(missing))

    manifest = ROOT / "data" / "datasets_manifest.json"
    write_manifest(specs.values(), manifest)
    print(f"Wrote dataset manifest: {manifest}")
    for spec in specs.values():
        print(f"{spec.key}: {spec.root}")


if __name__ == "__main__":
    main()

