"""Create a reproducible VisA PCB1 train/val/test manifest.

Reads Amazon Science's official 2-class high-shot CSV, preserves its Test
partition, and stratifies ONLY the official Train partition into Train/Val.
Does not modify source images, source CSV, or KSDD2 experiments.

Usage (from the repository root):
    python tools/prepare_visa_splits.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    ("train", "normal"): 602,
    ("train", "anomaly"): 60,
    ("test", "normal"): 402,
    ("test", "anomaly"): 40,
}
CLASS_MAP = {"normal": ("good", 0), "anomaly": ("defect", 1)}
SEED = 42
VAL_RATIO = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_relative_path(value: str, data_root: Path, prefix: tuple[str, ...]) -> str:
    if not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid path: {value!r}")
    relative = PurePosixPath(value)
    if (relative.is_absolute() or ".." in relative.parts
            or relative.parts[:len(prefix)] != prefix):
        raise ValueError(f"Unexpected path: {value!r}")
    candidate = data_root.joinpath(*relative.parts).resolve()
    if not candidate.is_relative_to(data_root.resolve()) or not candidate.is_file():
        raise FileNotFoundError(f"Missing or unsafe file: {value}")
    return relative.as_posix()


def counts_for(rows: list[dict]) -> dict[str, int]:
    counts = Counter(record["label"] for record in rows)
    return {
        "total": len(rows),
        "good": counts["good"],
        "defect": counts["defect"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare VisA PCB1 high-shot split manifest")
    parser.add_argument("--csv", type=Path, default=ROOT / "data/official/2cls_highshot.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/VisA")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/splits/visa_pcb1_highshot_v1.json")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    data_root = args.data_root.resolve()
    output_path = args.output.resolve()
    if not csv_path.is_file():
        parser.error(f"CSV not found: {csv_path}")
    if not (data_root / "pcb1/Data/Images").is_dir():
        parser.error(f"PCB1 images not found: {data_root}")
    if output_path.exists():
        parser.error(f"Refusing to overwrite existing manifest: {output_path}")

    source_rows: dict[str, list[dict]] = {"train": [], "test": []}
    observed: Counter[tuple[str, str]] = Counter()
    seen_images: set[str] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["object", "split", "label", "image", "mask"]:
            raise ValueError(f"Unexpected CSV fields: {reader.fieldnames}")

        for line_number, source in enumerate(reader, start=2):
            if source["object"] != "pcb1":
                continue
            split, label = source["split"], source["label"]
            if split not in source_rows or label not in CLASS_MAP:
                raise ValueError(f"Invalid split or label at CSV line {line_number}")
            image = checked_relative_path(
                source["image"], data_root,
                ("pcb1", "Data", "Images", "Normal" if label == "normal" else "Anomaly"),
            )
            if image in seen_images:
                raise ValueError(f"Duplicate image across CSV records: {image}")
            seen_images.add(image)

            if label == "normal":
                if source["mask"]:
                    raise ValueError(f"Normal image unexpectedly has a mask: {image}")
                mask = None
            else:
                mask = checked_relative_path(
                    source["mask"], data_root,
                    ("pcb1", "Data", "Masks", "Anomaly"),
                )

            display_label, label_id = CLASS_MAP[label]
            source_rows[split].append({
                "image": image,
                "mask": mask,
                "label": display_label,
                "label_id": label_id,
            })
            observed[split, label] += 1

    for key, expected in EXPECTED.items():
        if observed[key] != expected:
            raise ValueError(f"Unexpected PCB1 CSV count {key}: {observed[key]} != {expected}")
    if sum(observed.values()) != 1104:
        raise ValueError("Expected exactly 1104 PCB1 records")

    original_train = source_rows["train"]
    train, val = train_test_split(
        original_train,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=[row["label_id"] for row in original_train],
    )
    splits = {
        "train": sorted(train, key=lambda row: row["image"]),
        "val": sorted(val, key=lambda row: row["image"]),
        "test": sorted(source_rows["test"], key=lambda row: row["image"]),
    }
    paths = {name: {r["image"] for r in rows} for name, rows in splits.items()}
    if any(paths[a] & paths[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Image path overlap across final splits")
    if paths["train"] | paths["val"] != {r["image"] for r in original_train}:
        raise ValueError("Train/val no longer equals original official Train")
    if paths["test"] != {r["image"] for r in source_rows["test"]}:
        raise ValueError("Official Test was changed")

    manifest = {
        "schema_version": 1,
        "dataset": "VisA",
        "object": "pcb1",
        "official_split": "2cls_highshot.csv",
        "official_csv_sha256": sha256_file(csv_path),
        "seed": SEED,
        "validation_ratio": VAL_RATIO,
        "split_method": "stratified_split_of_official_train_only",
        "class_to_id": {"good": 0, "defect": 1},
        "defect_class_id": 1,
        "official_test_preserved": True,
        "related_image_groups_verified": False,
        "grouping_limitation": (
            "Exact duplicate content was checked during the earlier data audit, "
            "but visually similar images / shared product identity have not been ruled out."
        ),
        "counts": {name: counts_for(rows) for name, rows in splits.items()},
        "splits": splits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(f"Official CSV SHA256: {manifest['official_csv_sha256']}")
    for name, counts in manifest["counts"].items():
        print(f"{name:5}: good={counts['good']:3}, defect={counts['defect']:2}, total={counts['total']:3}")
    print("Official Test unchanged: True")
    print("Split path overlap: 0")
    print("Related-image group independence: NOT VERIFIED")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
