# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Shared helpers for the webcam demos: model lookup, asset pipeline, render loop."""

from __future__ import annotations

import time
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Callable, TypeVar

import cv2ext

from trtutils.models import (
    DEIM,
    DFINE,
    RFDETR,
    VGG,
    YOLOX,
    AlexNet,
    ConvNeXt,
    DEIMv2,
    DenseNet,
    DepthAnythingV2,
    EfficientNet,
    EfficientNetV2,
    GoogLeNet,
    Inception,
    MaxViT,
    MNASNet,
    MobileNetV2,
    MobileNetV3,
    RegNet,
    ResNet,
    ResNeXt,
    RTDETRv1,
    RTDETRv2,
    RTDETRv3,
    ShuffleNetV2,
    SqueezeNet,
    SwinTransformer,
    SwinTransformerV2,
    ViT,
    WideResNet,
    YOLOv3,
    YOLOv5,
    YOLOv7,
    YOLOv8,
    YOLOv9,
    YOLOv10,
    YOLOv11,
    YOLOv12,
    YOLOv13,
    YOLOv26,
)
from trtutils.models._utils import get_valid_models

if TYPE_CHECKING:
    import numpy as np

    from trtutils.models._model import Model

T = TypeVar("T")

DATA_DIR = Path(__file__).parent / "data"
FPS_BUFFER_SIZE = 30

# Detectors: each alias has a unique _model_type, so iterating builds a clean lookup.
_DETECTOR_ALIASES: list[type[Model]] = [
    YOLOv3,
    YOLOv5,
    YOLOv7,
    YOLOv8,
    YOLOv9,
    YOLOv10,
    YOLOv11,
    YOLOv12,
    YOLOv13,
    YOLOv26,
    YOLOX,
    RTDETRv1,
    RTDETRv2,
    RTDETRv3,
    DFINE,
    DEIM,
    DEIMv2,
    RFDETR,
]

# Torchvision classifiers all share _model_type="torchvision_classifier", so we
# need explicit name-prefix → alias-class mapping. Order matters: longer prefixes
# must match first (efficientnet_v2 before efficientnet, swin_v2 before swin,
# wide_resnet before resnet, etc.). Tuples are searched in order.
_CLASSIFIER_PREFIX_TO_CLS: list[tuple[str, type[Model]]] = [
    ("efficientnet_v2", EfficientNetV2),
    ("efficientnet_", EfficientNet),
    ("swin_v2", SwinTransformerV2),
    ("swin_", SwinTransformer),
    ("mobilenet_v2", MobileNetV2),
    ("mobilenet_v3", MobileNetV3),
    ("wide_resnet", WideResNet),
    ("resnext", ResNeXt),
    ("resnet", ResNet),
    ("convnext_", ConvNeXt),
    ("densenet", DenseNet),
    ("regnet_", RegNet),
    ("shufflenet_", ShuffleNetV2),
    ("squeezenet", SqueezeNet),
    ("vgg", VGG),
    ("maxvit_", MaxViT),
    ("vit_", ViT),
    ("mnasnet", MNASNet),
    ("googlenet", GoogLeNet),
    ("inception_", Inception),
    ("alexnet", AlexNet),
]

_DEPTH_ALIASES: list[type[Model]] = [DepthAnythingV2]


def _build_detector_lookup() -> dict[str, type[Model]]:
    out: dict[str, type[Model]] = {}
    for cls in _DETECTOR_ALIASES:
        for name in get_valid_models(cls._model_type):
            out[name] = cls
    return out


def _build_classifier_lookup() -> dict[str, type[Model]]:
    out: dict[str, type[Model]] = {}
    for name in get_valid_models("torchvision_classifier"):
        for prefix, cls in _CLASSIFIER_PREFIX_TO_CLS:
            if name.startswith(prefix):
                out[name] = cls
                break
        else:
            err_msg = f"No alias class registered for classifier model {name!r}"
            raise RuntimeError(err_msg)
    return out


def _build_depth_lookup() -> dict[str, type[Model]]:
    out: dict[str, type[Model]] = {}
    for cls in _DEPTH_ALIASES:
        for name in get_valid_models(cls._model_type):
            out[name] = cls
    return out


DETECTOR_LOOKUP: dict[str, type[Model]] = _build_detector_lookup()
CLASSIFIER_LOOKUP: dict[str, type[Model]] = _build_classifier_lookup()
DEPTH_LOOKUP: dict[str, type[Model]] = _build_depth_lookup()


def resolve(name: str, lookup: dict[str, type[Model]], task: str) -> type[Model]:
    """
    Resolve a model name to its alias class.

    Parameters
    ----------
    name : str
        The model name (e.g. ``"yolov10n"``, ``"resnet18"``).
    lookup : dict[str, type[Model]]
        The task-specific lookup table.
    task : str
        Human-readable task name used in error messages.

    Returns
    -------
    type[Model]
        The alias class for the given model name.

    Raises
    ------
    ValueError
        If the model name is not registered for the task.

    """
    cls = lookup.get(name)
    if cls is None:
        valid = ", ".join(sorted(lookup))
        err_msg = f"Unknown {task} model {name!r}. Valid names: {valid}"
        raise ValueError(err_msg)
    return cls


def _expected_imgsz(cls: type[Model], model: str) -> int:
    """Compute the imgsz `cls.download()` would pick for this model name."""
    imgsz = cls._default_imgsz
    if cls._model_imgszs is not None:
        for substring, size in cls._model_imgszs.items():
            if substring in model:
                imgsz = size
                break
    return imgsz


def prepare_engine(
    name: str,
    lookup: dict[str, type[Model]],
    task: str,
    *,
    verbose: bool = False,
) -> tuple[type[Model], Path]:
    """
    Resolve a model name, download its ONNX, and build a TensorRT engine.

    ONNX and engine files are cached under ``demos/webcam/data/`` keyed by
    model name. Either is re-created only if missing.

    Parameters
    ----------
    name : str
        The model name (e.g. ``"yolov10n"``).
    lookup : dict[str, type[Model]]
        The task-specific lookup table.
    task : str
        Human-readable task name used in error messages.
    verbose : bool
        Forward verbose flag to download() and build().

    Returns
    -------
    tuple[type[Model], Path]
        The resolved alias class and the path to the built engine.

    """
    cls = resolve(name, lookup, task)
    imgsz = _expected_imgsz(cls, name)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = DATA_DIR / f"{name}.onnx"
    engine_path = DATA_DIR / f"{name}.engine"

    if not onnx_path.exists():
        print(f"[webcam] downloading {name} -> {onnx_path}")
        cls.download(name, onnx_path, imgsz=imgsz, verbose=verbose)
    if not engine_path.exists():
        print(f"[webcam] building engine {name} -> {engine_path}")
        cls.build(onnx_path, engine_path, imgsz=imgsz, verbose=verbose)

    return cls, engine_path


def webcam_loop(
    source: int | str,
    window_title: str,
    infer: Callable[[np.ndarray], T],
    render: Callable[[np.ndarray, T, float], np.ndarray],
) -> None:
    """
    Run a generic webcam display loop.

    Parameters
    ----------
    source : int | str
        OpenCV video source (camera index or filepath).
    window_title : str
        Title of the cv2ext.Display window.
    infer : Callable[[np.ndarray], T]
        Inference function: takes a frame, returns prediction.
    render : Callable[[np.ndarray, T, float], np.ndarray]
        Render function: takes (frame, prediction, fps) and returns the
        canvas to display.

    """
    fps_buffer = [1.0] * FPS_BUFFER_SIZE
    display = cv2ext.Display(window_title)
    try:
        for fid, frame in cv2ext.IterableVideo(source, buffersize=3, use_thread=True):
            if display.stopped:
                break

            t0 = time.time()
            prediction = infer(frame)
            t1 = time.time()
            fps = 1000.0 / max((t1 - t0) * 1000.0, 1e-6)
            fps_buffer[fid % FPS_BUFFER_SIZE] = fps
            avg_fps = mean(fps_buffer)

            canvas = render(frame, prediction, avg_fps)
            display.update(canvas)
    finally:
        display.stop()
