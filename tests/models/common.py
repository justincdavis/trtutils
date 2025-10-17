# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Shared test infrastructure for all model tests."""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from trtutils import build_engine

# Shared constants
DLA_ENGINES = 2
GPU_ENGINES = 4
NUM_ITERS = 100


def build_model_engine(
    onnx_path: Path,
    engine_path: Path,
    timing_cache_path: Path,
    *,
    use_dla: bool | None = None,
    requires_static_shapes: bool | None = None,
) -> Path:
    """
    Build a model engine.

    Parameters
    ----------
    onnx_path : Path
        Path to the ONNX model.
    engine_path : Path
        Path where the engine will be saved.
    timing_cache_path : Path
        Path to the timing cache file.
    use_dla : bool, optional
        Whether or not to build the engine using DLA.
    requires_static_shapes : bool, optional
        Whether the model requires static shapes (e.g., YOLOv9).

    Returns
    -------
    Path
        The location of the compiled engine.

    """
    if engine_path.exists():
        return engine_path

    shapes = None
    if requires_static_shapes:
        shapes = [("images", (1, 3, 640, 640))]

    build_engine(
        onnx_path,
        engine_path,
        timing_cache=timing_cache_path,
        dla_core=0 if use_dla else None,
        gpu_fallback=bool(use_dla),
        fp16=True,
        shapes=shapes,
    )

    return engine_path


def run_model_test(
    model_instance,
    num_iters: int = NUM_ITERS,
) -> None:
    """
    Run a basic model test.

    Parameters
    ----------
    model_instance
        The model instance to test.
    num_iters : int
        Number of iterations to run.

    """
    outputs = None
    for _ in range(num_iters):
        outputs = model_instance.mock_run()

    assert outputs is not None


def run_multiple_models_test(
    model_instances: list,
    num_iters: int = NUM_ITERS,
) -> None:
    """
    Run multiple model instances test.

    Parameters
    ----------
    model_instances : list
        List of model instances to test.
    num_iters : int
        Number of iterations to run.

    """
    outputs = None
    for _ in range(num_iters):
        outputs = [model.mock_run() for model in model_instances]

    assert outputs is not None
    for o in outputs:
        assert o is not None


def validate_model_results(
    model_instance,
    image_paths: list[str],
    ground_truths: list[int],
) -> None:
    """
    Validate model detection results.

    Parameters
    ----------
    model_instance
        The model instance to test.
    image_paths : list[str]
        Paths to test images.
    ground_truths : list[int]
        Expected number of detections per image.

    Raises
    ------
    FileNotFoundError
        If an image file does not exist.

    """
    for gt, ipath in zip(ground_truths, image_paths):
        image = cv2.imread(ipath)
        if image is None:
            err_msg = f"Failed to read image: {ipath}"
            raise FileNotFoundError(err_msg)

        outputs = model_instance.run(image)
        bboxes = [bbox for (bbox, _, _) in model_instance.get_detections(outputs)]

        # check within +-2 bounding boxes from ground truth
        assert max(1, gt - 1) <= len(bboxes) <= gt + 1
        # we always have at least one detection per image
        assert len(bboxes) >= 1


def validate_swapping_preproc(
    model_instance,
    image_paths: list[str],
    ground_truths: list[int],
) -> None:
    """
    Validate model results with swapping preprocessing methods.

    Parameters
    ----------
    model_instance
        The model instance to test.
    image_paths : list[str]
        Paths to test images.
    ground_truths : list[int]
        Expected number of detections per image.

    Raises
    ------
    FileNotFoundError
        If an image file does not exist.

    """
    for gt, ipath in zip(ground_truths, image_paths):
        image = cv2.imread(ipath)
        if image is None:
            err_msg = f"Failed to read image: {ipath}"
            raise FileNotFoundError(err_msg)

        for preproc in ["cpu", "cuda", "trt"]:
            tensor, ratios, padding = model_instance.preprocess(
                image, method=preproc, no_copy=True
            )
            outputs = model_instance.run(
                tensor,
                ratios,
                padding,
                preprocessed=True,
                postprocess=True,
                no_copy=True,
            )
            bboxes = [bbox for (bbox, _, _) in model_instance.get_detections(outputs)]

            # check within +-2 bounding boxes from ground truth
            assert max(1, gt - 1) <= len(bboxes) <= gt + 1
            # we always have at least one detection per image
            assert len(bboxes) >= 1


def test_pagelocked_performance(
    model_class,
    engine_path: Path,
    model_name: str,
    input_range: tuple[float, float],
    num_iters: int = NUM_ITERS,
) -> None:
    """
    Test pagelocked memory performance.

    Parameters
    ----------
    model_class
        The model class to instantiate.
    engine_path : Path
        Path to the engine file.
    model_name : str
        Name of the model for logging.
    input_range : tuple[float, float]
        Input range for the model.
    num_iters : int
        Number of iterations to run.

    """
    model = model_class(
        engine_path,
        conf_thres=0.25,
        warmup=True,
        input_range=input_range,
        preprocessor="cuda",
        pagelocked_mem=False,
    )
    model_pagelocked = model_class(
        engine_path,
        conf_thres=0.25,
        warmup=True,
        input_range=input_range,
        preprocessor="cuda",
        pagelocked_mem=True,
    )

    times = []
    times_pagelocked = []

    for _ in range(num_iters):
        t0 = time.time()
        model.mock_run()
        t1 = time.time()
        times.append(t1 - t0)

        t00 = time.time()
        model_pagelocked.mock_run()
        t11 = time.time()
        times_pagelocked.append(t11 - t00)

    model_mean = sum(times) / len(times)
    model_pagelocked_mean = sum(times_pagelocked) / len(times_pagelocked)
    speedup = model_mean / model_pagelocked_mean

    assert speedup > 1.0
    print(f"{model_name} Pagelocked Speedup: {speedup:.2f}x")

    del model
    del model_pagelocked

