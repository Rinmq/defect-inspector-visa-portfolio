"""Train a VisA PCB1 ResNet18 using the existing training/validation helpers.

Training helpers are provided by defect_inspector.training.
Official test data is not loaded or evaluated here.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from defect_inspector.model import build_model
from defect_inspector.preprocessing import (
    EvalPreprocessor,
    PreprocessingConfig,
    TrainPreprocessor,
)
from defect_inspector.training import train_epoch, validate, sha256_file, save_json
from defect_inspector.visa_data import DEFAULT_MANIFEST, VisADataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
THRESHOLD = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet18 training on VisA PCB1 high-shot")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Check one real GPU batch and model forward pass, without saving or training",
    )
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.height, args.width) <= 0 or not (0 < args.lr < 1):
        parser.error("epochs, batch-size, height, width must be positive; 0 < lr < 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available; check CUDA-enabled PyTorch")
    if not DEFAULT_MANIFEST.is_file():
        raise FileNotFoundError(DEFAULT_MANIFEST)

    # Dataset never reads official Test during training.
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    if (manifest.get("dataset") != "VisA"
            or manifest.get("object") != "pcb1"
            or manifest.get("official_split") != "2cls_highshot.csv"
            or manifest.get("official_test_preserved") is not True):
        raise ValueError("Unexpected VisA split manifest")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = PreprocessingConfig(height=args.height, width=args.width)
    train_dataset = VisADataset(
        "train",
        transform=TrainPreprocessor(config, horizontal_flip_probability=0.0),
    )
    val_dataset = VisADataset("val", transform=EvalPreprocessor(config))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VisA PCB1: train={len(train_dataset)}, val={len(val_dataset)}", flush=True)
    print(f"Input: [N, 3, {args.height}, {args.width}]", flush=True)

    model = build_model(pretrained=True).to(device)
    if args.check_only:
        model.eval()
        images, labels = next(iter(train_loader))
        if images.shape != (len(labels), 3, args.height, args.width):
            raise ValueError(f"Unexpected input shape {tuple(images.shape)}")
        with torch.inference_mode():
            logits = model(images.to(device))
        if logits.shape != (len(labels), 2) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Bad model logits in check-only run")
        print(f"GPU forward check: PASS (batch={len(labels)})", flush=True)
        return

    run_id = datetime.now().strftime("resnet18_visa_pcb1_%Y%m%d_%H%M%S_%f")
    run_dir = PROJECT_ROOT / "artifacts" / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    configuration = {
        "seed": SEED,
        "dataset": "VisA",
        "object": "pcb1",
        "official_split": "2cls_highshot.csv",
        "split_manifest": "data/splits/visa_pcb1_highshot_v1.json",
        "split_manifest_sha256": sha256_file(DEFAULT_MANIFEST),
        "official_csv_sha256": manifest["official_csv_sha256"],
        "official_test_used_during_training": False,
        "related_image_groups_verified": False,
        "architecture": "resnet18",
        "initial_weights": "IMAGENET1K_V1",
        "fine_tuning": "all_layers",
        "loss": "CrossEntropyLoss (unweighted)",
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "train_horizontal_flip_probability": 0.0,
        "model_selection": "highest validation average_precision (ties: earlier epoch)",
        "defect_threshold": THRESHOLD,
        "input": config.to_metadata(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    save_json(run_dir / "experiment_config.json", configuration)
    print(f"Run directory: {run_dir}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_ap = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        metrics, labels, scores = validate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val": metrics})
        save_json(run_dir / "history.json", history)
        print(
            f"Epoch {epoch}/{args.epochs}: loss={train_loss:.4f}, "
            f"val_AP={metrics['average_precision']:.4f}, "
            f"val_F1={metrics['defect_f1']:.4f}, "
            f"FN={metrics['confusion_matrix']['fn']}, "
            f"FP={metrics['confusion_matrix']['fp']}",
            flush=True,
        )

        if metrics["average_precision"] > best_ap:
            best_ap = metrics["average_precision"]
            torch.save(
                {
                    "model_state_dict": {
                        name: param.detach().cpu()
                        for name, param in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "val_ap": best_ap,
                },
                run_dir / "best_model.pth",
            )
            save_json(
                run_dir / "model_metadata.json",
                {
                    "schema_version": 1,
                    "architecture": "resnet18",
                    "dataset": "VisA",
                    "object": "pcb1",
                    "classes": {"0": "good", "1": "defect"},
                    "defect_class_id": 1,
                    "input": config.to_metadata(),
                    "output_type": "logits",
                    "defect_threshold": THRESHOLD,
                    "selected_epoch": epoch,
                    "selection_metric": "validation_average_precision",
                    "validation_metrics": metrics,
                },
            )
            save_json(
                run_dir / "best_validation_predictions.json",
                [
                    {
                        "image": image_path.relative_to(val_dataset.data_root).as_posix(),
                        "true_label": label,
                        "predicted_label": int(score >= THRESHOLD),
                        "defect_score": score,
                        "correct": (int(score >= THRESHOLD) == label),
                    }
                    for (image_path, _), label, score in zip(
                        val_dataset.samples, labels, scores, strict=True
                    )
                ],
            )
            print(f"Saved best checkpoint: epoch={epoch} AP={best_ap:.4f}", flush=True)

    print(f"Training complete. Best val AP: {best_ap:.4f}", flush=True)
    print(f"Artifacts: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
