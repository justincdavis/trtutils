# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Shared constants and configuration for benchmark scripts."""

from __future__ import annotations

import json
from pathlib import Path

# Paths
REPO_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_PATH = str((REPO_DIR / "data" / "horse.jpg").resolve())
SAHI_IMAGE_PATH = str((REPO_DIR / "data" / "cars.jpeg").resolve())

# Framework lists
MODEL_FRAMEWORKS = [
    "ultralytics(torch)",
    "ultralytics(trt)",
    "trtutils",
    "trtutils(graph)",
    "tensorrt",
    "tensorrt(graph)",
]

BATCH_FRAMEWORKS = [
    "trtutils",
    "trtutils(graph)",
    "ultralytics(trt)",
    "ultralytics(torch)",
]

ULTRALYTICS_MODELS = [
    "yolov13n",
    "yolov13s",
    "yolov13m",
    "yolov12n",
    "yolov12s",
    "yolov12m",
    "yolov11n",
    "yolov11s",
    "yolov11m",
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov8n",
    "yolov8s",
    "yolov8m",
]

# Plot constants
IMAGE_SIZES = [160, 320, 480, 640, 800, 960, 1120, 1280]

MODEL_FAMILIES = [
    "yolov13",
    "yolov12",
    "yolov11",
    "yolov10",
    "yolov9",
    "yolov8",
    "yolov7",
    "yolox",
]


def _load_model_info() -> tuple[list[str], list[str], dict[str, str]]:
    """Load model info from JSON files, return (dirs, names, name->dir mapping)."""
    model_info_dir = Path(__file__).resolve().parent.parent / "info" / "model_info"
    if not model_info_dir.exists():
        err_msg = f"Model info directory not found: {model_info_dir}"
        raise FileNotFoundError(err_msg)

    model_dirs: list[str] = []
    all_model_names: list[str] = []
    model_to_dir: dict[str, str] = {}

    for json_file in sorted(model_info_dir.glob("*.json")):
        model_dir = json_file.stem
        model_dirs.append(model_dir)
        with json_file.open("r") as f:
            model_data = json.load(f)
            for model_name in model_data:
                all_model_names.append(model_name)
                model_to_dir[model_name] = model_dir

    return model_dirs, all_model_names, model_to_dir


def _load_model_imgsizes() -> dict[str, list[int]]:
    """Load model image sizes from JSON config."""
    imgsz_file = Path(__file__).resolve().parent.parent / "info" / "model_imgsz.json"
    if not imgsz_file.exists():
        err_msg = f"Model image size config not found: {imgsz_file}"
        raise FileNotFoundError(err_msg)
    with imgsz_file.open("r") as f:
        return json.load(f)


MODEL_DIRS, MODEL_NAMES, MODEL_TO_DIR = _load_model_info()
MODEL_TO_IMGSIZES = _load_model_imgsizes()
