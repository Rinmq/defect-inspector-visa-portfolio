"""Read-only audit for the official VisA PCB1 two-class high-shot CSV.

This script does not create or modify dataset files. It checks CSV records,
file paths, image/mask integrity, expected label counts, and exact duplicates.

Usage:
  python tools/audit_visa_pcb1.py
  python tools/audit_visa_pcb1.py --csv data/official/2cls_highshot.csv \
      --data-root data/raw/VisA
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    ("train", "normal"): 602,
    ("train", "anomaly"): 60,
    ("test", "normal"): 402,
    ("test", "anomaly"): 40,
}
LABEL_IDS = {"normal": 0, "anomaly": 1}
REQUIRED_COLUMNS = {"object", "split", "label", "image", "mask"}


def safe_path(value: str, root: Path, expected_prefix: tuple[str, ...]) -> Path:
    """Resolve an official POSIX-relative path within the local data root."""
    if not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid path: {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or p.parts[: len(expected_prefix)] != expected_prefix:
        raise ValueError(f"Path outside expected subtree: {value!r}")
    resolved_root = root.resolve()
    resolved_path = root.joinpath(*p.parts).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Path escapes data root: {value!r}")
    return resolved_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(image_path: Path, mask_path: Path | None) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image missing: {image_path}")
    with Image.open(image_path) as image:
        image.load()
        image_size = image.size
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"Invalid image dimensions: {image_path}")
    if mask_path is None:
        return
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask missing: {mask_path}")
    with Image.open(mask_path) as mask:
        if mask.size != image_size:
            raise ValueError(f"Image/mask size mismatch: {image_path} / {mask_path}")
        if mask.convert("L").getbbox() is None:
            raise ValueError(f"Anomaly mask contains no nonzero pixels: {mask_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit VisA PCB1 high-shot data")
    parser.add_argument("--csv", type=Path,
                        default=PROJECT_ROOT / "data/official/2cls_highshot.csv")
    parser.add_argument("--data-root", type=Path,
                        default=PROJECT_ROOT / "data/raw/VisA")
    parser.add_argument("--no-hash", action="store_true",
                        help="Skip SHA256 duplicate-content check (faster)")
    args = parser.parse_args()

    csv_file = args.csv.resolve()
    data_root = args.data_root.resolve()
    if not csv_file.is_file():
        parser.error(f"CSV not found: {csv_file}")
    if not (data_root / "pcb1/Data/Images").is_dir():
        parser.error(f"VisA pcb1 images not found: {data_root / 'pcb1/Data/Images'}")

    counts: Counter[tuple[str, str]] = Counter()
    seen_paths: set[str] = set()
    seen_digests: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    pcb1_count = 0

    with csv_file.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            parser.error(f"Unexpected CSV header: {reader.fieldnames}")
        for lineno, record in enumerate(reader, start=2):
            if record.get("object") != "pcb1":
                continue
            pcb1_count += 1
            try:
                split = record["split"]
                label = record["label"]
                if split not in {"train", "test"}:
                    raise ValueError(f"Invalid split: {split!r}")
                if label not in LABEL_IDS:
                    raise ValueError(f"Invalid label: {label!r}")

                image_name = record["image"]
                image_prefix = ("pcb1", "Data", "Images",
                                "Normal" if label == "normal" else "Anomaly")
                image_path = safe_path(image_name, data_root, image_prefix)
                if image_name in seen_paths:
                    raise ValueError(f"Duplicate image record: {image_name}")
                seen_paths.add(image_name)

                mask_name = record["mask"]
                if label == "normal":
                    if mask_name:
                        raise ValueError(f"Normal image unexpectedly has mask: {mask_name}")
                    mask_path = None
                else:
                    mask_path = safe_path(
                        mask_name, data_root, ("pcb1", "Data", "Masks", "Anomaly")
                    )

                validate_image(image_path, mask_path)
                counts[split, label] += 1

                if not args.no_hash:
                    digest = sha256_file(image_path)
                    if digest in seen_digests:
                        previous_split, previous_path = seen_digests[digest]
                        # Any exact duplicate is worth reporting, even within a split.
                        raise ValueError(
                            "Byte-identical images: "
                            f"{previous_split}/{previous_path} and {split}/{image_name}"
                        )
                    seen_digests[digest] = (split, image_name)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                errors.append(f"CSV line {lineno}: {exc}")

    print("VisA PCB1 audit")
    print(f"CSV records (pcb1): {pcb1_count}")
    for (split, label), expected in EXPECTED.items():
        actual = counts[split, label]
        print(f"{split:5} {label:7}: {actual:4} / expected {expected}")
        if actual != expected:
            errors.append(f"Count mismatch for {split}/{label}: {actual} != {expected}")
    if pcb1_count != 1104:
        errors.append(f"PCB1 CSV rows: {pcb1_count} != 1104")
    print(f"Image SHA256 check: {'skipped' if args.no_hash else 'performed'}")
    print(f"Issues: {len(errors)}")
    for message in errors[:30]:
        print("  ERROR:", message)
    if len(errors) > 30:
        print(f"  ... and {len(errors) - 30} more issues")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
