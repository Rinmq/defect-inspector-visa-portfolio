import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from defect_inspector.gui import InferenceEngine
from defect_inspector.visa_data import VisADataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()

    metadata = json.loads(
        (run_dir / "model_metadata.json")
        .read_text(encoding="utf-8")
    )

    if metadata.get("dataset") != "VisA":
        raise ValueError("Expected VisA dataset")

    if metadata.get("object") != "pcb1":
        raise ValueError("Expected pcb1")

    if float(metadata["defect_threshold"]) != 0.5:
        raise ValueError("Expected frozen threshold 0.5")

    json_path = run_dir / "test_evaluation.json"
    csv_path = run_dir / "test_predictions.csv"

    if json_path.exists() or csv_path.exists():
        raise FileExistsError(
            "Test evaluation already exists; refusing overwrite"
        )

    dataset = VisADataset("test")

    if len(dataset) != 442:
        raise ValueError("Unexpected test size")

    labels = [label for _, label in dataset.samples]

    if (labels.count(0), labels.count(1)) != (402, 40):
        raise ValueError("Unexpected test class counts")

    engine = InferenceEngine(run_dir)

    y_true = []
    y_pred = []
    y_score = []
    predictions = []

    for index, (image_path, label) in enumerate(
        dataset.samples, start=1
    ):
        result = engine.predict(image_path)

        score = float(result["defect_score"])
        predicted = int(result["is_defect"])

        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid score: {image_path}")

        y_true.append(label)
        y_pred.append(predicted)
        y_score.append(score)

        predictions.append({
            "image": image_path.relative_to(
                dataset.data_root
            ).as_posix(),
            "true_label": label,
            "predicted_label": predicted,
            "defect_score": score,
            "correct": int(label == predicted),
        })

        if index % 50 == 0 or index == len(dataset):
            print(f"Processed: {index}/{len(dataset)}", flush=True)

    tn, fp, fn, tp = (
        int(x) for x in confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
    )

    report = {
        "dataset": "VisA",
        "object": "pcb1",
        "split": "official_test",
        "total": len(dataset),
        "good": 402,
        "defect": 40,
        "selected_epoch": metadata["selected_epoch"],
        "threshold": engine.threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "average_precision": float(
            average_precision_score(y_true, y_score)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(predictions[0])
        )
        writer.writeheader()
        writer.writerows(predictions)

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== VisA PCB1 OFFICIAL TEST ===")
    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nEvaluation complete: PASS")


if __name__ == "__main__":
    main()
