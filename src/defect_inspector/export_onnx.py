
"""Export the best checkpoint and compare PyTorch with ONNX Runtime on CPU."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from defect_inspector.visa_data import VisADataset
from defect_inspector.model import build_model
from defect_inspector.preprocessing import (
    EvalPreprocessor,
    PreprocessingConfig,
)

ABS_TOL = 1e-4
REL_TOL = 1e-3
OPSET = 18


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def defect_score(logits: np.ndarray, defect_id: int) -> float:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    probabilities = values / values.sum(axis=1, keepdims=True)
    return float(probabilities[0, defect_id])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint_path = run_dir / "best_model.pth"
    metadata_path = run_dir / "model_metadata.json"
    onnx_path = run_dir / "model.onnx"
    report_path = run_dir / "onnx_export_report.json"

    if onnx_path.exists():
        raise FileExistsError(
            f"ONNX file already exists: {onnx_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    config = PreprocessingConfig.from_metadata(metadata)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["epoch"] != metadata["selected_epoch"]:
        raise ValueError("Checkpoint and metadata epochs differ")

    model = build_model(pretrained=False).cpu()
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    if metadata.get("dataset") != "VisA" or metadata.get("object") != "pcb1":
        raise ValueError("Only VisA PCB1 models are supported")

    dataset = VisADataset(
        "val",
        transform=EvalPreprocessor(config),
    )

    example, _ = dataset[0]
    example = example.unsqueeze(0).contiguous()

    print("Selected epoch:", checkpoint["epoch"], flush=True)
    print("Selected validation AP:", checkpoint["val_ap"], flush=True)
    print("Exporting ONNX...", flush=True)

    with tempfile.TemporaryDirectory(prefix=".onnx_export_", dir=run_dir) as temp_dir:
        staged_onnx = Path(temp_dir) / "model.onnx"

        torch.onnx.export(
            model,
            (example,),
            str(staged_onnx),
            input_names=["images"],
            output_names=["logits"],
            opset_version=OPSET,
            dynamo=True,
            external_data=False,
        )

        onnx.checker.check_model(str(staged_onnx))
        print("ONNX structural check: PASS", flush=True)

        session = ort.InferenceSession(
            str(staged_onnx),
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name

        defect_id = int(metadata["defect_class_id"])
        threshold = float(metadata["defect_threshold"])

        sample_indices = [
            next(
                index
                for index, (_, label) in enumerate(dataset.samples)
                if label == class_id
            )
            for class_id in (0, 1)
        ]

        comparisons = []

        for index in sample_indices:
            image, true_label = dataset[index]
            tensor = image.unsqueeze(0).contiguous()

            with torch.inference_mode():
                pytorch_logits = model(tensor).numpy()

            onnx_logits = np.asarray(
                session.run(None, {input_name: tensor.numpy()})[0]
            )

            if pytorch_logits.shape != onnx_logits.shape:
                raise ValueError("Output shapes differ")

            error = float(
                np.max(np.abs(pytorch_logits - onnx_logits))
            )
            close = bool(
                np.allclose(
                    pytorch_logits,
                    onnx_logits,
                    atol=ABS_TOL,
                    rtol=REL_TOL,
                )
            )

            pytorch_score = defect_score(
                pytorch_logits, defect_id
            )
            onnx_score = defect_score(
                onnx_logits, defect_id
            )

            same_decision = (
                (pytorch_score >= threshold)
                == (onnx_score >= threshold)
            )

            result = {
                "image": dataset.samples[index][0].name,
                "true_label": true_label,
                "pytorch_logits": pytorch_logits.tolist(),
                "onnx_logits": onnx_logits.tolist(),
                "max_absolute_error": error,
                "allclose": close,
                "decision_match": bool(same_decision),
            }
            comparisons.append(result)

            print(
                f"Image={result['image']} "
                f"max_abs_error={error:.8f} "
                f"allclose={close} "
                f"decision_match={same_decision}",
                flush=True,
            )

        passed = all(
            item["allclose"] and item["decision_match"]
            for item in comparisons
        )

        report = {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_sha256": sha256_file(staged_onnx),
            "opset": OPSET,
            "execution_provider": "CPUExecutionProvider",
            "batch_size": 1,
            "absolute_tolerance": ABS_TOL,
            "relative_tolerance": REL_TOL,
            "comparisons": comparisons,
            "passed": passed,
        }

        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if not passed:
            raise RuntimeError(
                f"Parity check failed. See {report_path}"
            )

        metadata["onnx"] = {
            "file": onnx_path.name,
            "sha256": report["onnx_sha256"],
            "opset": OPSET,
            "batch_size": 1,
        }

        staged_metadata = Path(temp_dir) / "model_metadata.json"
        staged_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if onnx_path.exists():
            raise FileExistsError(f"ONNX file already exists: {onnx_path}")

        staged_onnx.rename(onnx_path)

        try:
            staged_metadata.replace(metadata_path)
        except BaseException:
            onnx_path.unlink(missing_ok=True)
            raise

    print("PyTorch vs ONNX Runtime: PASS")
    print("Saved:", onnx_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
