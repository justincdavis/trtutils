# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
# MIT License
"""Utility functions for managing benchmark models."""

from __future__ import annotations

from pathlib import Path

from trtutils.download import download

REPO_DIR = Path(__file__).parent.parent


def get_model_dir(model_name: str) -> Path:
    """Get the directory for a model based on its name."""
    if "yolov10" in model_name:
        model_type = "yolov10"
    elif "yolov9" in model_name:
        model_type = "yolov9"
    elif "yolov8" in model_name:
        model_type = "yolov8"
    elif "yolov11" in model_name:
        model_type = "yolov11"
    elif "yolov12" in model_name:
        model_type = "yolov12"
    elif "yolov13" in model_name:
        model_type = "yolov13"
    elif "yolov7" in model_name:
        model_type = "yolov7"
    elif "yolox" in model_name:
        model_type = "yolox"
    else:
        raise ValueError(f"Unknown model type for {model_name}")
    return REPO_DIR / "data" / model_type


def ensure_model_available(
    model_name: str,
    imgsz: int,
    opset: int = 17,
    *,
    auto_download: bool = True,
) -> Path:
    """Ensure a model is available, downloading it if necessary."""
    model_dir = get_model_dir(model_name)
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
