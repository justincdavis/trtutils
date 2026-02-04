# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
# MIT License
"""Utility functions for managing benchmark models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).parent.parent


def _get_model_class(model_name: str) -> Any:
    """Get the model class for a given model name."""
    from trtutils.models import (
        DEIM,
        DFINE,
        RFDETR,
        YOLO3,
        YOLO5,
        YOLO7,
        YOLO8,
        YOLO9,
        YOLO10,
        YOLO11,
        YOLO12,
        YOLO13,
        YOLOX,
        DEIMv2,
        RTDETRv1,
        RTDETRv2,
        RTDETRv3,
    )

    # Map model name prefixes to classes
    # Order matters - more specific matches first
    model_mapping: list[tuple[str, Any]] = [
        ("yolov13", YOLO13),
        ("yolov12", YOLO12),
        ("yolov11", YOLO11),
        ("yolov10", YOLO10),
        ("yolov9", YOLO9),
        ("yolov8", YOLO8),
        ("yolov7", YOLO7),
        ("yolov5", YOLO5),
        ("yolov3", YOLO3),
        ("yolox", YOLOX),
        ("rtdetrv3", RTDETRv3),
        ("rtdetrv2", RTDETRv2),
        ("rtdetrv1", RTDETRv1),
        ("dfine", DFINE),
        ("deimv2", DEIMv2),
        # deim_ prefix models (e.g., deim_dfine_n, deim_rtdetrv2_r18)
        # are DEIM variants and should use DEIM build settings
        ("deim_", DEIM),
        ("deim", DEIM),
        ("rfdetr", RFDETR),
    ]

    model_lower = model_name.lower()
    for prefix, model_class in model_mapping:
        if model_lower.startswith(prefix):
            return model_class

    err_msg = f"Unknown model type: {model_name}"
    raise ValueError(err_msg)


def ensure_model_available(
    model_name: str,
    imgsz: int,
    model_to_dir: dict[str, str],
    opset: int = 17,
    *,
    auto_download: bool = True,
) -> Path:
    """Ensure a model is available, downloading it if necessary."""
    from trtutils.download import download

    if model_name not in model_to_dir:
        err_msg = f"Unknown model: {model_name}. Not found in model directory mapping."
        raise ValueError(err_msg)
    
    model_dir = REPO_DIR / "data" / model_to_dir[model_name]
    model_path = model_dir / f"{model_name}_{imgsz}.onnx"
    
    if model_path.exists():
        return model_path
    
    if not auto_download:
        err_msg = f"Model not found: {model_path}"
        raise FileNotFoundError(err_msg)
    
    # Auto-download
    print(f"Downloading {model_name} @ {imgsz}x{imgsz}...")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    download(
        model=model_name,
        output=model_path,
        opset=opset,
        imgsz=imgsz,
        accept=True,
        verbose=False,
    )
    return model_path


def build_model(
    onnx: Path,
    output: Path,
    imgsz: int,
    opt_level: int = 3,
) -> None:
    """Build a model from an ONNX file using the appropriate model class."""
    # Extract model name from the onnx filename (e.g., "yolov10n_640" -> "yolov10n")
    model_name = onnx.stem.rsplit("_", 1)[0]

    # Get the appropriate model class and call its build method
    model_class = _get_model_class(model_name)
    model_class.build(
        onnx=onnx,
        output=output,
        imgsz=imgsz,
        opt_level=opt_level,
    )
