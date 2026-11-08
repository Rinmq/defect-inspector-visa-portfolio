
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class InferenceEngine:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()

        metadata_path = self.run_dir / "model_metadata.json"
        model_path = self.run_dir / "model.onnx"

        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)

        expected_hash = metadata.get("onnx", {}).get("sha256")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in expected_hash)
        ):
            raise ValueError("Missing or invalid ONNX SHA-256 in metadata")

        with model_path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()

        if actual_hash != expected_hash.lower():
            raise ValueError("ONNX SHA-256 mismatch: model and metadata differ")

        self.config = metadata["input"]
        self.threshold = float(metadata["defect_threshold"])

        expected = {
            "color_space": "RGB",
            "layout": "NCHW",
            "dtype": "float32",
            "resize_method": "stretch",
            "interpolation": "bilinear",
        }

        for key, value in expected.items():
            if self.config[key] != value:
                raise ValueError(
                    f"Unsupported preprocessing: {key}"
                )

        if metadata["output_type"] != "logits":
            raise ValueError("Expected logits output")

        if int(metadata["defect_class_id"]) != 1:
            raise ValueError("Expected defect class ID = 1")

        self.width = int(self.config["width"])
        self.height = int(self.config["height"])

        self.scale = np.float32(
            self.config["pixel_scale"]
        )

        self.mean = np.asarray(
            self.config["mean"],
            dtype=np.float32,
        )

        self.std = np.asarray(
            self.config["std"],
            dtype=np.float32,
        )

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid input dimensions")

        if not np.isfinite(self.scale):
            raise ValueError("Invalid pixel scale")

        if (
            self.mean.shape != (3,)
            or self.std.shape != (3,)
            or not np.all(np.isfinite(self.mean))
            or not np.all(np.isfinite(self.std))
            or np.any(self.std <= 0)
        ):
            raise ValueError("Invalid normalization parameters")

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Invalid threshold")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        if (
            len(self.session.get_inputs()) != 1
            or len(self.session.get_outputs()) != 1
        ):
            raise ValueError("Unexpected ONNX interface")

        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path: Path):
        start = time.perf_counter()

        image_path = image_path.resolve()

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        # 日本語を含むWindowsパスにも対応する読み込み方法
        image_bytes = np.fromfile(
            str(image_path),
            dtype=np.uint8,
        )

        bgr = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR,
        )

        if bgr is None:
            raise ValueError(
                f"Cannot decode image: {image_path}"
            )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        resized = cv2.resize(
            rgb,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )

        image = resized.astype(np.float32)
        image *= self.scale
        image = (image - self.mean) / self.std

        tensor = np.transpose(
            image,
            (2, 0, 1),
        )[None, ...]

        tensor = np.ascontiguousarray(
            tensor,
            dtype=np.float32,
        )

        output = self.session.run(
            None,
            {self.input_name: tensor},
        )[0]

        logits = np.asarray(
            output,
            dtype=np.float32,
        )

        if logits.shape != (1, 2):
            raise ValueError(
                f"Unexpected output shape: {logits.shape}"
            )

        if not np.all(np.isfinite(logits)):
            raise ValueError("Non-finite logits")

        values = logits[0].astype(np.float64)
        values -= np.max(values)

        probabilities = np.exp(values)
        probabilities /= probabilities.sum()

        defect_score = float(probabilities[1])
        is_defect = defect_score >= self.threshold

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return {
            "rgb": rgb,
            "defect_score": defect_score,
            "is_defect": is_defect,
            "logits": logits[0].tolist(),
            "elapsed_ms": elapsed_ms,
        }


class InspectorWindow(QMainWindow):
    def __init__(self, engine: InferenceEngine):
        super().__init__()

        self.engine = engine
        self.original_pixmap = None

        self.setWindowTitle("Defect Inspector")
        self.resize(920, 760)
        self.setMinimumSize(640, 550)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("DEFECT INSPECTOR")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel(
            "ResNet18  |  ONNX Runtime CPU  |  "
            "正常・欠陥の画像分類"
        )
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        self.preview = QLabel(
            "画像を選択してください"
        )
        self.preview.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.preview.setMinimumHeight(310)
        self.preview.setObjectName("preview")
        layout.addWidget(self.preview, stretch=1)

        self.path_label = QLabel(
            "画像：未選択"
        )
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.select_button = QPushButton(
            "画像を選択して推論"
        )
        self.select_button.setMinimumHeight(46)
        self.select_button.clicked.connect(
            self.choose_image
        )
        layout.addWidget(self.select_button)

        result_row = QHBoxLayout()

        self.result_label = QLabel("判定待ち")
        self.result_label.setObjectName("result")
        result_row.addWidget(self.result_label)

        result_row.addStretch()

        self.score_label = QLabel("--")
        self.score_label.setObjectName("score")
        result_row.addWidget(self.score_label)

        layout.addLayout(result_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMinimumHeight(18)
        layout.addWidget(self.progress)

        self.details = QLabel(
            "欠陥スコア：--  |  閾値："
            f"{self.engine.threshold * 100:.1f}%"
        )
        self.details.setWordWrap(True)
        layout.addWidget(self.details)

        self.model_label = QLabel(
            f"Model: {self.engine.run_dir.name}"
        )
        self.model_label.setObjectName("subtitle")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #111827;
                color: #e5e7eb;
                font-family: "Yu Gothic UI";
                font-size: 14px;
            }

            QLabel#title {
                font-size: 25px;
                font-weight: bold;
                color: #f9fafb;
            }

            QLabel#subtitle {
                color: #9ca3af;
                font-size: 12px;
            }

            QLabel#preview {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 10px;
                color: #9ca3af;
            }

            QLabel#result {
                font-size: 24px;
                font-weight: bold;
            }

            QLabel#score {
                font-size: 23px;
                font-weight: bold;
            }

            QPushButton {
                background: #2563eb;
                border: none;
                border-radius: 8px;
                color: white;
                font-size: 15px;
                font-weight: bold;
                padding: 10px;
            }

            QPushButton:hover {
                background: #1d4ed8;
            }

            QPushButton:disabled {
                background: #4b5563;
            }

            QProgressBar {
                background: #374151;
                border: none;
                border-radius: 7px;
            }

            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 7px;
            }
        """)

    def choose_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "検査する画像を選択",
            str(Path.cwd()),
            "画像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )

        if filename:
            self.predict_file(Path(filename))

    def predict_file(self, image_path: Path):
        self.select_button.setEnabled(False)
        self.result_label.setText("推論中...")
        QApplication.processEvents()

        try:
            result = self.engine.predict(image_path)

            rgb = np.ascontiguousarray(result["rgb"])

            height, width, _ = rgb.shape

            qt_image = QImage(
                rgb.data,
                width,
                height,
                rgb.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()

            self.original_pixmap = QPixmap.fromImage(
                qt_image
            )

            self.update_preview()

            score = result["defect_score"]
            is_defect = result["is_defect"]

            self.result_label.setText(
                "欠陥（defect）"
                if is_defect
                else "正常（good）"
            )

            self.result_label.setStyleSheet(
                "color: #f87171;"
                if is_defect
                else "color: #4ade80;"
            )

            self.score_label.setText(
                f"{score * 100:.2f}%"
            )

            self.progress.setValue(
                round(score * 1000)
            )

            self.details.setText(
                f"欠陥スコア：{score * 100:.4f}%"
                f"  |  閾値："
                f"{self.engine.threshold * 100:.1f}%"
                f"  |  処理時間："
                f"{result['elapsed_ms']:.1f} ms"
                f"\nLogits: "
                f"{result['logits'][0]:.6f}, "
                f"{result['logits'][1]:.6f}"
            )

            self.path_label.setText(
                f"画像：{image_path.resolve()}"
            )

        except Exception as error:
            self.result_label.setText("推論失敗")

            QMessageBox.critical(
                self,
                "Defect Inspector Error",
                str(error),
            )

        finally:
            self.select_button.setEnabled(True)

    def update_preview(self):
        if self.original_pixmap is None:
            return

        size = self.preview.contentsRect().size()

        scaled = self.original_pixmap.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.preview.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Directory containing model.onnx and metadata",
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image to classify on startup",
    )

    args = parser.parse_args()

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")

    try:
        engine = InferenceEngine(args.run)
    except Exception as error:
        print(f"Model initialization failed: {error}")
        return 1

    window = InspectorWindow(engine)
    window.show()

    if args.image is not None:
        QTimer.singleShot(
            0,
            lambda: window.predict_file(args.image),
        )

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
