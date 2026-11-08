"""VisA PCB1 two-class dataset based on a frozen split manifest.

The class uses the same (image, label_id) / .samples interface as
KSDD2Dataset.  It does not modify images, masks, or split files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/splits/visa_pcb1_highshot_v1.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/raw/VisA"
CLASS_TO_ID = {"good": 0, "defect": 1}
SPLITS = ("train", "val", "test")


def _relative_parts(value: object, expected_prefix: tuple[str, ...]) -> tuple[str, ...]:
    """Validate a POSIX path from the manifest before joining the data root."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid relative path: {value!r}")
    if value.startswith("/") or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    parts = tuple(value.split("/"))
    if (len(parts) != len(expected_prefix) + 1
            or parts[:len(expected_prefix)] != expected_prefix
            or any(part in ("", ".", "..") for part in parts)):
        raise ValueError(f"Unexpected VisA path: {value!r}")
    return parts


class VisADataset(Dataset):
    """PCB1 image-level good/defect classification.

    A mask is validated for anomalous images but is not returned because
    the current model is an image classifier, not a segmentation model.
    """

    def __init__(
        self,
        split: str,
        transform: Callable[[Image.Image], Any] | None = None,
        manifest_path: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown VisA split: {split!r}")

        self.split = split
        self.transform = transform
        self.data_root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
        manifest_file = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST

        with manifest_file.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        if (manifest.get("schema_version") != 1
                or manifest.get("dataset") != "VisA"
                or manifest.get("object") != "pcb1"
                or manifest.get("official_split") != "2cls_highshot.csv"
                or manifest.get("class_to_id") != CLASS_TO_ID
                or manifest.get("defect_class_id") != 1
                or manifest.get("official_test_preserved") is not True):
            raise ValueError("Unexpected VisA PCB1 manifest configuration")

        split_records = manifest.get("splits")
        split_counts = manifest.get("counts")
        if not isinstance(split_records, dict) or not isinstance(split_counts, dict):
            raise ValueError("Manifest is missing splits/counts")

        self.samples: list[tuple[Path, int]] = []
        self.records: list[dict] = []
        seen_images: set[str] = set()
        seen_masks: set[str] = set()

        for split_name in SPLITS:
            records = split_records.get(split_name)
            counts = split_counts.get(split_name)
            if not isinstance(records, list) or not isinstance(counts, dict):
                raise ValueError(f"Missing manifest records/counts for {split_name}")

            class_counts = {"good": 0, "defect": 0}
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError(f"Invalid {split_name} record: {record!r}")

                label = record.get("label")
                label_id = record.get("label_id")
                if label not in CLASS_TO_ID or type(label_id) is not int or label_id != CLASS_TO_ID[label]:
                    raise ValueError(f"Invalid label in {split_name}: {record!r}")

                folder = "Normal" if label == "good" else "Anomaly"
                image_parts = _relative_parts(
                    record.get("image"),
                    ("pcb1", "Data", "Images", folder),
                )
                image_relative = "/".join(image_parts)
                if image_relative in seen_images:
                    raise ValueError(f"Duplicate/cross-split image: {image_relative}")
                seen_images.add(image_relative)

                image_path = self.data_root.joinpath(*image_parts)
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                mask = record.get("mask")
                if label == "good":
                    if mask not in (None, ""):
                        raise ValueError(f"Normal image must not have a mask: {image_relative}")
                else:
                    mask_parts = _relative_parts(
                        mask,
                        ("pcb1", "Data", "Masks", "Anomaly"),
                    )
                    mask_relative = "/".join(mask_parts)
                    if mask_relative in seen_masks:
                        raise ValueError(f"Duplicate anomaly mask: {mask_relative}")
                    seen_masks.add(mask_relative)
                    mask_path = self.data_root.joinpath(*mask_parts)
                    if not mask_path.is_file():
                        raise FileNotFoundError(f"Mask not found: {mask_path}")
                    if Path(image_parts[-1]).stem != Path(mask_parts[-1]).stem:
                        raise ValueError(f"Image/mask filename mismatch: {image_relative}")

                class_counts[label] += 1
                if split_name == split:
                    self.samples.append((image_path, label_id))
                    self.records.append(record)

            if (counts.get("total") != len(records)
                    or counts.get("good") != class_counts["good"]
                    or counts.get("defect") != class_counts["defect"]):
                raise ValueError(f"Manifest count mismatch in {split_name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image_path, label_id = self.samples[index]
        with Image.open(image_path) as file:
            image = file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_id
