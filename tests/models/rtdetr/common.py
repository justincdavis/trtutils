# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

from trtutils.models import DEIM, DFINE, RFDETR, RTDETRv1, RTDETRv2, RTDETRv3, DEIMv2

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
from ..paths import GROUND_TRUTHS, IMAGE_PATHS, RTDETR_ENGINE_PATHS, RTDETR_ONNX_PATHS

# Re-export for backward compatibility
__all__ = [
    "DLA_ENGINES",
    "GPU_ENGINES",
    "NUM_ITERS",
    "build_rtdetr",
    "rtdetr_run",
    "rtdetr_run_multiple",
    "rtdetr_results",
    "rtdetr_swapping_preproc_results",
    "rtdetr_pagelocked_perf",
]

# Map model names to their classes
MODEL_CLASSES: dict[str, type] = {
    "rtdetrv1": RTDETRv1,
    "rtdetrv2": RTDETRv2,
    "rtdetrv3": RTDETRv3,
    "dfine": DFINE,
    "deim": DEIM,
    "deimv2": DEIMv2,
    "rfdetr": RFDETR,
}


def build_rtdetr(model_name: str, *, use_dla: bool | None = None) -> Path:
    """
    Build an RT-DETR family engine.

    Parameters
    ----------
    model_name : str
        The model name to use (rtdetrv1, rtdetrv2, etc.).
    use_dla : bool, optional
        Whether or not to build the engine using DLA.

    Returns
    -------
    Path
        The location of the compiled engine.

    """
    onnx_path = RTDETR_ONNX_PATHS[model_name]
    engine_path = RTDETR_ENGINE_PATHS[model_name]
    timing_cache_path = Path(__file__).parent / "timing.cache"

    return build_model_engine(
        onnx_path,
        engine_path,
        timing_cache_path,
        use_dla=use_dla,
        requires_static_shapes=False,
    )


def rtdetr_run(
    model_name: str, preprocessor: str = "cpu", *, use_dla: bool | None = None
) -> None:
    """Check if an RT-DETR family engine will run."""
    engine_path = build_rtdetr(model_name, use_dla=use_dla)

    model_class = MODEL_CLASSES[model_name]
    model = model_class(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        input_range=(0, 1),
        preprocessor=preprocessor,
        no_warn=True,
    )

    run_model_test(model, NUM_ITERS)
    del model


def rtdetr_run_multiple(
    model_name: str,
    preprocessor: str = "cpu",
    count: int = 4,
    *,
    use_dla: bool | None = None,
) -> None:
    """Check if multiple RT-DETR family engines can run at once."""
    engine_path = build_rtdetr(model_name, use_dla=use_dla)

    model_class = MODEL_CLASSES[model_name]
    models = [
        model_class(
            engine_path,
            conf_thres=0.25,
            warmup=False,
            input_range=(0, 1),
            preprocessor=preprocessor,
            no_warn=True,
        )
        for _ in range(count)
    ]

    run_multiple_models_test(models, NUM_ITERS)

    for model in models:
        del model


def rtdetr_results(
    model_name: str, preprocessor: str = "cpu", *, use_dla: bool | None = None
) -> None:
    """
    Check if the results are valid for an RT-DETR family model.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.

    """
    engine_path = build_rtdetr(model_name, use_dla=use_dla)

    model_class = MODEL_CLASSES[model_name]
    model = model_class(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        input_range=(0, 1),
        preprocessor=preprocessor,
        no_warn=True,
    )

    validate_model_results(model, IMAGE_PATHS, GROUND_TRUTHS)
    del model


def rtdetr_swapping_preproc_results(model_name: str, *, use_dla: bool | None = None) -> None:
    """
    Check if the results are valid for an RT-DETR family model with swapping preprocessing.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.

    """
    engine_path = build_rtdetr(model_name, use_dla=use_dla)

    model_class = MODEL_CLASSES[model_name]
    model = model_class(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        input_range=(0, 1),
        preprocessor="cpu",
        no_warn=True,
    )

    validate_swapping_preproc(model, IMAGE_PATHS, GROUND_TRUTHS)
    del model


def rtdetr_pagelocked_perf(model_name: str, *, use_dla: bool | None = None) -> None:
    """Check if the results are valid for an RT-DETR family model."""
    engine_path = build_rtdetr(model_name, use_dla=use_dla)
    model_class = MODEL_CLASSES[model_name]

    test_pagelocked_performance(
        model_class, engine_path, model_name.upper(), (0, 1), NUM_ITERS
    )
