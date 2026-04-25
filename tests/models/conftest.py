# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Shared configuration and helpers for model correctness tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# ---------------------------------------------------------------------------
# Ground-truth expectations
# ---------------------------------------------------------------------------
DETECTOR_EXPECTED: list[dict[str, Any]] = [
    {
        "image": "data/horse.jpg",
        "expected_classes": [17],  # COCO horse
        "min_detections": 1,
        "conf_thres": 0.3,
    },
]

CLASSIFIER_EXPECTED: list[dict[str, Any]] = [
    {
        "image": "data/horse.jpg",
        "expected_top_k_classes": [339],  # ImageNet "sorrel"
        "top_k": 5,
    },
]

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
# Each entry: (model_class_name, model_name_for_download, imgsz_or_None)
# Class names are strings resolved at runtime to avoid import-time dependency
# on TensorRT.
DETECTOR_MODELS: dict[str, tuple[str, str, int | None]] = {
    "yolov10": ("YOLOv10", "yolov10n", None),
    "yolov8": ("YOLOv8", "yolov8n", None),
    "yolov11": ("YOLOv11", "yolov11n", None),
    "rtdetrv1": ("RTDETRv1", "rtdetrv1_r18", None),
    "rtdetrv3": ("RTDETRv3", "rtdetrv3_r18", None),
    "dfine": ("DFINE", "dfine_n", None),
    "rfdetr": ("RFDETR", "rfdetr_n", 384),
}

CLASSIFIER_MODELS: dict[str, tuple[str, str, int | None]] = {
    "resnet18": ("ResNet", "resnet18", None),
    "efficientnet_b0": ("EfficientNet", "efficientnet_b0", None),
    "mobilenet_v3_small": ("MobileNetV3", "mobilenet_v3_small", None),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_model_class(class_name: str) -> type:
    """Resolve a model class name to the actual class from trtutils.models."""
    import trtutils.models as models

    cls = getattr(models, class_name, None)
    if cls is None:
        msg = f"Unknown model class: {class_name}"
        raise ValueError(msg)
    return cls


def build_model_engine(
    model_class_name: str,
    model_name: str,
    imgsz: int | None,
    cache_dir: Path | None = None,
) -> Path:
    """
    Download ONNX (if needed) and build a TRT engine (if needed).

    Engines are cached under ``data/engines/<model_name>/``.

    Parameters
    ----------
    model_class_name : str
        Name of the model class in ``trtutils.models``.
    model_name : str
        The model variant to download (e.g. ``"yolov10n"``).
    imgsz : int | None
        Image size override; ``None`` uses the class default.
    cache_dir : Path | None
        Override for the cache root. Defaults to ``data/engines/``.

    Returns
    -------
    Path
        Path to the compiled TensorRT engine.

    """
    model_class = _resolve_model_class(model_class_name)

    if cache_dir is None:
        cache_dir = DATA_DIR / "engines"

    model_dir = cache_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    onnx_dir = DATA_DIR / model_name
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"{model_name}.onnx"
    engine_path = model_dir / f"{model_name}.engine"

    # Return early if engine already exists
    if engine_path.exists():
        return engine_path

    # Download ONNX if it doesn't exist
    if not onnx_path.exists():
        download_kwargs: dict[str, Any] = {
            "model": model_name,
            "output": onnx_path,
        }
        if imgsz is not None:
            download_kwargs["imgsz"] = imgsz
        model_class.download(**download_kwargs)

    # Build engine
    build_kwargs: dict[str, Any] = {
        "onnx": onnx_path,
        "output": engine_path,
        "opt_level": 1,
        "verbose": False,
    }
    if imgsz is not None:
        build_kwargs["imgsz"] = imgsz
    model_class.build(**build_kwargs)

    return engine_path
