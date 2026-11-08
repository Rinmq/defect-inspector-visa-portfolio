"""Shared VisA training and validation helpers."""

import hashlib
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def save_json(path: Path, value: dict | list) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(
                f"Non-finite training loss at epoch {epoch}"
            )

        loss.backward()
        optimizer.step()

        count = labels.size(0)
        total_examples += count
        total_loss += loss.item() * count

        if batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"  epoch={epoch} "
                f"batch={batch_index}/{len(loader)} "
                f"train_loss={total_loss / total_examples:.4f}",
                flush=True,
            )

    return total_loss / total_examples

@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_examples = 0
    true_labels = []
    defect_scores = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Non-finite validation loss")

        count = labels.size(0)
        total_examples += count
        total_loss += loss.item() * count

        true_labels.extend(labels.cpu().tolist())
        defect_scores.extend(
            torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        )

    predicted = [
        int(score >= THRESHOLD)
        for score in defect_scores
    ]

    tn, fp, fn, tp = (
        int(value)
        for value in confusion_matrix(
            true_labels, predicted, labels=[0, 1]
        ).ravel()
    )

    metrics = {
        "loss": total_loss / total_examples,
        "average_precision": float(
            average_precision_score(true_labels, defect_scores)
        ),
        "accuracy": float(
            accuracy_score(true_labels, predicted)
        ),
        "defect_precision": float(
            precision_score(
                true_labels, predicted, zero_division=0
            )
        ),
        "defect_recall": float(
            recall_score(
                true_labels, predicted, zero_division=0
            )
        ),
        "defect_f1": float(
            f1_score(true_labels, predicted, zero_division=0)
        ),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "threshold": THRESHOLD,
    }

    return metrics, true_labels, defect_scores
