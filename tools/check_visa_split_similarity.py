"""Screen VisA PCB1 splits for visually similar images; read-only.

This is a near-duplicate screening aid, NOT a proof of group independence.
Outputs stay under data/raw/VisA/_review (ignored by Git).

Run from the Defect Inspector 2 project root:
    python tools/check_visa_split_similarity.py
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def load_rgb(data_root: Path, relative: str) -> np.ndarray:
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != "pcb1":
        raise ValueError(f"Unsafe image path: {relative!r}")
    image_path = data_root.joinpath(*p.parts).resolve()
    if not image_path.is_relative_to(data_root.resolve()):
        raise ValueError(f"Path escapes data root: {relative!r}")
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to decode image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def perceptual_hash(rgb: np.ndarray) -> int:
    """63-bit pHash using an 8x8 low-frequency DCT block (excluding DC)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    reduced = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(reduced.astype(np.float32))[:8, :8].reshape(-1)
    median = float(np.median(dct[1:]))
    result = 0
    for coefficient in dct[1:]:
        result = (result << 1) | int(coefficient > median)
    return result


def safe_rows(manifest: dict, split: str) -> list[dict]:
    rows = manifest["splits"][split]
    if not isinstance(rows, list):
        raise ValueError(f"Invalid split records: {split}")
    return rows


def create_review_sheet(rows: list[dict], data_root: Path, dest: Path) -> None:
    thumb_w, thumb_h = 300, 180
    row_h, outer = 228, 8
    sheet = Image.new("RGB", (thumb_w * 2 + 3 * outer, max(1, len(rows)) * row_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, pair in enumerate(rows):
        y = index * row_h
        title = (f"{pair['split_a']} / {pair['split_b']}   "
                 f"pHash distance={pair['phash_distance']}   "
                 f"labels={pair['label_a']}/{pair['label_b']}")
        draw.text((outer, y + 4), title, fill="black")
        for col, (key, label_key) in enumerate((("image_a", "label_a"), ("image_b", "label_b"))):
            rel = pair[key]
            rgb = load_rgb(data_root, rel)
            img = Image.fromarray(rgb)
            img = ImageOps.contain(img, (thumb_w, thumb_h))
            x = outer + col * (thumb_w + outer)
            sheet.paste(img, (x + (thumb_w - img.width) // 2, y + 26 + (thumb_h - img.height) // 2))
            draw.text((x, y + 209), f"{pair[label_key]}: {Path(rel).name}", fill="black")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser(description="VisA split near-duplicate screening")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/splits/visa_pcb1_highshot_v1.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/VisA")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/VisA/_review")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "VisA" or manifest.get("object") != "pcb1":
        raise ValueError("Expected VisA PCB1 manifest")
    if manifest.get("related_image_groups_verified") is True:
        print("Notice: manifest claims groups verified; this check cannot establish that.")

    records: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        records[split] = []
        for record in safe_rows(manifest, split):
            rgb = load_rgb(data_root, record["image"])
            records[split].append({
                "image": record["image"],
                "label": record["label"],
                "hash": perceptual_hash(rgb),
            })
        print(f"Loaded {split}: {len(records[split])} images", flush=True)

    report = {
        "dataset": "VisA", "object": "pcb1",
        "method": "63-bit grayscale DCT pHash; screening only, not identity verification",
        "limitations": [
            "Low pHash distance indicates similar global appearance, not necessarily the same physical board.",
            "Small defects may not affect pHash. High distance does not rule out related images.",
            "This script cannot establish product/capture-sequence group independence.",
        ],
        "comparisons": {},
    }
    selected: list[dict] = []
    for split_a, split_b in PAIRS:
        close4, close8, num_pairs = 0, 0, 0
        candidates: list[tuple[int, str, str, str, str]] = []
        for a in records[split_a]:
            for b in records[split_b]:
                distance = (a["hash"] ^ b["hash"]).bit_count()
                num_pairs += 1
                close4 += distance <= 4
                close8 += distance <= 8
                candidates.append((distance, a["image"], b["image"], a["label"], b["label"]))
        closest = heapq.nsmallest(6, candidates)
        name = f"{split_a}_vs_{split_b}"
        items = [
            {
                "split_a": split_a, "split_b": split_b,
                "image_a": pa, "image_b": pb,
                "label_a": la, "label_b": lb,
                "phash_distance": distance,
            }
            for distance, pa, pb, la, lb in closest
        ]
        report["comparisons"][name] = {
            "image_pairs_compared": num_pairs,
            "pairs_distance_le_4": close4,
            "pairs_distance_le_8": close8,
            "closest_pairs": items,
        }
        selected.extend(items)
        print(f"{name}: pairs={num_pairs}, distance<=4: {close4}, "
              f"distance<=8: {close8}, closest={closest[0][0] if closest else 'N/A'}", flush=True)

    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(data_root):
        raise ValueError("Output directory must be inside data-root to avoid Git publication")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "split_similarity_report.json"
    output_image = output_dir / "split_similarity_review.jpg"
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    create_review_sheet(selected, data_root, output_image)
    print("Review report:", output_json)
    print("Review image: ", output_image)
    print("Important: This is a screening result, NOT proof of group independence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
