# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

from trtutils.models import YOLO

from ..common import (
    DLA_ENGINES,
    GPU_ENGINES,
    NUM_ITERS,
    build_model_engine,
    run_model_test,
    run_multiple_models_test,
    test_pagelocked_performance,
    validate_model_results,
    validate_swapping_preproc,
)
from ..paths import GROUND_TRUTHS, IMAGE_PATHS, YOLO_ENGINE_PATHS, YOLO_ONNX_PATHS

# Re-export for backward compatibility
__all__ = [
    "DLA_ENGINES",
    "GPU_ENGINES",
    "NUM_ITERS",
    "build_yolo",
    "yolo_run",
    "yolo_run_multiple",
    "yolo_results",
    "yolo_swapping_preproc_results",
    "yolo_pagelocked_perf",
]

YOLOV9_VERSION = 9


def build_yolo(version: int, *, use_dla: bool | None = None) -> Path:
    """
    Build a YOLO engine.

    Parameters
    ----------
    version : int
        The YOLO version to use.
    use_dla : bool, optional
        Whether or not to build the engine using DLA.

    Returns
    -------
    Path
        The location of the compiled engine.

    """
    onnx_path = YOLO_ONNX_PATHS[version]
    engine_path = YOLO_ENGINE_PATHS[version]
    timing_cache_path = Path(__file__).parent / "timing.cache"

    return build_model_engine(
        onnx_path,
        engine_path,
        timing_cache_path,
        use_dla=use_dla,
        requires_static_shapes=(version == YOLOV9_VERSION),
    )


def yolo_run(
    version: int, preprocessor: str = "cpu", *, use_dla: bool | None = None
) -> None:
    """Check if a YOLO engine will run."""
    engine_path = build_yolo(version, use_dla=use_dla)

    scale = (0, 1) if version != 0 else (0, 255)
    yolo = YOLO(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        input_range=scale,
        preprocessor=preprocessor,
        no_warn=True,
    )

    run_model_test(yolo, NUM_ITERS)
    del yolo


def yolo_run_multiple(
    version: int,
    preprocessor: str = "cpu",
    count: int = 4,
    *,
    use_dla: bool | None = None,
) -> None:
    """Check if multiple YOLO engines can run at once."""
    engine_path = build_yolo(version, use_dla=use_dla)

    scale = (0, 1) if version != 0 else (0, 255)
    yolos = [
        YOLO(
            engine_path,
            conf_thres=0.25,
            warmup=False,
            input_range=scale,
            preprocessor=preprocessor,
            no_warn=True,
        )
        for _ in range(count)
    ]

    run_multiple_models_test(yolos, NUM_ITERS)

    for yolo in yolos:
        del yolo


def yolo_results(
    version: int, preprocessor: str = "cpu", *, use_dla: bool | None = None
) -> None:
    """
    Check if the results are valid for a YOLO model.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.

    """
    engine_path = build_yolo(version, use_dla=use_dla)

    scale = (0, 1) if version != 0 else (0, 255)
    yolo = YOLO(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        input_range=scale,
        preprocessor=preprocessor,
        no_warn=True,
    )

    validate_model_results(yolo, IMAGE_PATHS, GROUND_TRUTHS)
    del yolo


def yolo_swapping_preproc_results(version: int, *, use_dla: bool | None = None) -> None:
    """
    Check if the results are valid for a YOLO model.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.

    """
    engine_path = build_yolo(version, use_dla=use_dla)

    scale = (0, 1) if version != 0 else (0, 255)
    yolo = YOLO(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        input_range=scale,
        preprocessor="cpu",
        no_warn=True,
    )

    validate_swapping_preproc(yolo, IMAGE_PATHS, GROUND_TRUTHS)
    del yolo


def yolo_pagelocked_perf(version: int, *, use_dla: bool | None = None) -> None:
    """Check if the results are valid for a YOLO model."""
    engine_path = build_yolo(version, use_dla=use_dla)
    scale = (0, 1) if version != 0 else (0, 255)
    model_name = f"YOLOv{version}" if version != 0 else "YOLOX"

    test_pagelocked_performance(YOLO, engine_path, model_name, scale, NUM_ITERS)
