"""Tests for VisA PCB1 dataset manifest validation and DataLoader contract."""
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from defect_inspector.visa_data import VisADataset


@pytest.fixture
def sample_visa(tmp_path):
    data_root = tmp_path / "VisA"

    def add_image(label, filename):
        category = "Normal" if label == "good" else "Anomaly"
        path = data_root / "pcb1" / "Data" / "Images" / category / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (18, 12), (100, 120, 140)).save(path)
        mask = ""
        if label == "defect":
            mask_path = data_root / "pcb1" / "Data" / "Masks" / "Anomaly" / (Path(filename).stem + ".png")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (18, 12), 255).save(mask_path)
            mask = mask_path.relative_to(data_root).as_posix()
        return {
            "image": path.relative_to(data_root).as_posix(),
            "mask": mask,
            "label": label,
            "label_id": int(label == "defect"),
        }

    splits = {
        "train": [add_image("good", "001.JPG"), add_image("defect", "002.JPG"), add_image("good", "003.JPG")],
        "val": [add_image("good", "004.JPG")],
        "test": [add_image("defect", "005.JPG")],
    }
    manifest = {
        "schema_version": 1,
        "dataset": "VisA",
        "object": "pcb1",
        "official_split": "2cls_highshot.csv",
        "class_to_id": {"good": 0, "defect": 1},
        "defect_class_id": 1,
        "official_test_preserved": True,
        "counts": {
            split: {
                "total": len(records),
                "good": sum(record["label"] == "good" for record in records),
                "defect": sum(record["label"] == "defect" for record in records),
            }
            for split, records in splits.items()
        },
        "splits": splits,
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    return data_root, manifest_file, manifest


def test_dataset_and_last_batch(sample_visa):
    data_root, manifest_file, _ = sample_visa
    dataset = VisADataset(
        "train",
        transform=lambda image: torch.ones((3, 16, 16), dtype=torch.float32),
        manifest_path=manifest_file,
        data_root=data_root,
    )
    assert len(dataset) == 3
    assert [label for _, label in dataset.samples] == [0, 1, 0]
    assert dataset[0][0].shape == (3, 16, 16)
    batches = list(DataLoader(dataset, batch_size=2, shuffle=False))
    assert len(batches) == 2
    assert batches[0][0].shape == (2, 3, 16, 16)
    assert batches[1][0].shape == (1, 3, 16, 16)
    assert batches[1][1].tolist() == [0]


def test_duplicate_across_splits_rejected(sample_visa):
    data_root, manifest_file, manifest = sample_visa
    manifest["splits"]["val"][0] = manifest["splits"]["train"][0].copy()
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate/cross-split"):
        VisADataset("train", manifest_path=manifest_file, data_root=data_root)


def test_missing_anomaly_mask_rejected(sample_visa):
    data_root, manifest_file, _ = sample_visa
    path = data_root / "pcb1" / "Data" / "Masks" / "Anomaly" / "002.png"
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Mask not found"):
        VisADataset("train", manifest_path=manifest_file, data_root=data_root)


def test_unsafe_relative_path_rejected(sample_visa):
    data_root, manifest_file, manifest = sample_visa
    manifest["splits"]["train"][0]["image"] = "pcb1/Data/Images/Normal/../001.JPG"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected VisA path"):
        VisADataset("train", manifest_path=manifest_file, data_root=data_root)
