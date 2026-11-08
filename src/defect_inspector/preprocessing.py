
"""Shared image preprocessing specification for training and inference."""

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


@dataclass(frozen=True)
class PreprocessingConfig:
    height: int = 608
    width: int = 224

    color_space: str = "RGB"
    layout: str = "NCHW"
    dtype: str = "float32"

    resize_method: str = "stretch"
    interpolation: str = "bilinear"
    pixel_scale: float = 1.0 / 255.0

    mean: tuple[float, float, float] = (
        0.485, 0.456, 0.406
    )
    std: tuple[float, float, float] = (
        0.229, 0.224, 0.225
    )

    def __post_init__(self):
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Invalid input size")

        if (
            self.color_space != "RGB"
            or self.layout != "NCHW"
            or self.dtype != "float32"
            or self.resize_method != "stretch"
            or self.interpolation != "bilinear"
        ):
            raise ValueError("Unsupported preprocessing specification")

        if not np.isclose(self.pixel_scale, 1.0 / 255.0):
            raise ValueError("Unsupported pixel scale")

        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("Expected 3-channel normalization")

        if not all(value > 0 for value in self.std):
            raise ValueError("Standard deviations must be positive")

    def to_metadata(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "color_space": self.color_space,
            "layout": self.layout,
            "dtype": self.dtype,
            "resize_method": self.resize_method,
            "interpolation": self.interpolation,
            "pixel_scale": self.pixel_scale,
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> "PreprocessingConfig":
        config = metadata["input"]

        return cls(
            height=config["height"],
            width=config["width"],
            color_space=config["color_space"],
            layout=config["layout"],
            dtype=config["dtype"],
            resize_method=config["resize_method"],
            interpolation=config["interpolation"],
            pixel_scale=config["pixel_scale"],
            mean=tuple(config["mean"]),
            std=tuple(config["std"]),
        )


def preprocess_image(
    image: Image.Image,
    config: PreprocessingConfig,
) -> torch.Tensor:
    """Return a normalized CHW float32 CPU tensor."""

    rgb_image = image.convert("RGB")
    pixels = np.asarray(rgb_image, dtype=np.uint8)

    resized = cv2.resize(
        pixels,
        (config.width, config.height),
        interpolation=cv2.INTER_LINEAR,
    )

    scaled = (
        resized.astype(np.float32)
        * np.float32(config.pixel_scale)
    )

    mean = np.asarray(config.mean, dtype=np.float32)
    std = np.asarray(config.std, dtype=np.float32)

    normalized = (scaled - mean) / std

    chw = np.transpose(normalized, (2, 0, 1))
    chw = np.ascontiguousarray(chw, dtype=np.float32)

    return torch.from_numpy(chw)


class EvalPreprocessor:
    """Deterministic preprocessing for val/test/inference."""

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
    ):
        self.config = config or PreprocessingConfig()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return preprocess_image(image, self.config)


class TrainPreprocessor:
    """Training preprocessing with optional random augmentation."""

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        horizontal_flip_probability: float = 0.0,
    ):
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("Invalid flip probability")

        self.config = config or PreprocessingConfig()
        self.horizontal_flip_probability = horizontal_flip_probability

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if torch.rand(()).item() < self.horizontal_flip_probability:
            image = ImageOps.mirror(image)

        return preprocess_image(image, self.config)
