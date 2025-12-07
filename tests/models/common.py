# Copyright (c) 2024-2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from pathlib import Path
from threading import Thread

import cv2
from typing_extensions import TypedDict

import trtutils.models as models
from trtutils.image._detector import Detector

from .paths import (
    DETR_ENGINE_PATHS,
    DETR_ONNX_PATHS,
    IMAGE_PATHS,
    YOLO_ENGINE_PATHS,
    YOLO_ONNX_PATHS,
)

DLA_ENGINES = 2
GPU_ENGINES = 4
NUM_ITERS = 100


class DetectorConfig(TypedDict):
    """Configuration for a detector model."""

    model_class: type[Detector]
    onnx_path: Path
    engine_path: Path


# Unified detector configuration - all models use string identifiers
DETECTOR_CONFIG: dict[str, DetectorConfig] = {
    # YOLO models
    "yolox": {
        "model_class": models.YOLOX,
        "onnx_path": YOLO_ONNX_PATHS[0],
        "engine_path": YOLO_ENGINE_PATHS[0],
    },
    "yolov3": {
        "model_class": models.YOLO3,
        "onnx_path": YOLO_ONNX_PATHS[3],
        "engine_path": YOLO_ENGINE_PATHS[3],
    },
    "yolov5": {
        "model_class": models.YOLO5,
        "onnx_path": YOLO_ONNX_PATHS[5],
        "engine_path": YOLO_ENGINE_PATHS[5],
    },
    "yolov7": {
        "model_class": models.YOLO7,
        "onnx_path": YOLO_ONNX_PATHS[7],
        "engine_path": YOLO_ENGINE_PATHS[7],
    },
    "yolov8": {
        "model_class": models.YOLO8,
        "onnx_path": YOLO_ONNX_PATHS[8],
        "engine_path": YOLO_ENGINE_PATHS[8],
    },
    "yolov9": {
        "model_class": models.YOLO9,
        "onnx_path": YOLO_ONNX_PATHS[9],
        "engine_path": YOLO_ENGINE_PATHS[9],
    },
    "yolov10": {
        "model_class": models.YOLO10,
        "onnx_path": YOLO_ONNX_PATHS[10],
        "engine_path": YOLO_ENGINE_PATHS[10],
    },
    "yolov11": {
        "model_class": models.YOLO11,
        "onnx_path": YOLO_ONNX_PATHS[11],
        "engine_path": YOLO_ENGINE_PATHS[11],
    },
    "yolov12": {
        "model_class": models.YOLO12,
        "onnx_path": YOLO_ONNX_PATHS[12],
        "engine_path": YOLO_ENGINE_PATHS[12],
    },
    "yolov13": {
        "model_class": models.YOLO13,
        "onnx_path": YOLO_ONNX_PATHS[13],
        "engine_path": YOLO_ENGINE_PATHS[13],
    },
    # DETR models
    "rtdetrv1": {
        "model_class": models.RTDETRv1,
        "onnx_path": DETR_ONNX_PATHS["rtdetrv1"],
        "engine_path": DETR_ENGINE_PATHS["rtdetrv1"],
    },
    "rtdetrv2": {
        "model_class": models.RTDETRv2,
        "onnx_path": DETR_ONNX_PATHS["rtdetrv2"],
        "engine_path": DETR_ENGINE_PATHS["rtdetrv2"],
    },
    "rtdetrv3": {
        "model_class": models.RTDETRv3,
        "onnx_path": DETR_ONNX_PATHS["rtdetrv3"],
        "engine_path": DETR_ENGINE_PATHS["rtdetrv3"],
    },
    "dfine": {
        "model_class": models.DFINE,
        "onnx_path": DETR_ONNX_PATHS["dfine"],
        "engine_path": DETR_ENGINE_PATHS["dfine"],
    },
    "deim": {
        "model_class": models.DEIM,
        "onnx_path": DETR_ONNX_PATHS["deim"],
        "engine_path": DETR_ENGINE_PATHS["deim"],
    },
    "deimv2": {
        "model_class": models.DEIMv2,
        "onnx_path": DETR_ONNX_PATHS["deimv2"],
        "engine_path": DETR_ENGINE_PATHS["deimv2"],
    },
    "rfdetr": {
        "model_class": models.RFDETR,
        "onnx_path": DETR_ONNX_PATHS["rfdetr"],
        "engine_path": DETR_ENGINE_PATHS["rfdetr"],
    },
}


def _get_model_name_from_path(path: Path) -> str:
    """Extract model name from path filename (without extension)."""
    return path.stem


def build_detector(model_id: str, *, use_dla: bool | None = None) -> Path:
    """
    Build a detector engine (works for both YOLO and DETR models).

    Parameters
    ----------
    model_id : str
        The model identifier string (e.g., "yolov8", "rtdetrv1").
    use_dla : bool, optional
        Whether or not to build the engine using DLA.

    Returns
    -------
    Path
        The location of the compiled engine.

    """
    config = DETECTOR_CONFIG[model_id]
    onnx_path = config["onnx_path"]
    engine_path = config["engine_path"]
    model_class = config["model_class"]

    # Return early if engine already exists
    if engine_path.exists():
        return engine_path

    download_model_name = _get_model_name_from_path(onnx_path)

    # Download ONNX if it doesn't exist - class handles all parameters
    if not onnx_path.exists():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        model_class.download(
            model=download_model_name,
            output=onnx_path,
            accept=True,
            no_warn=True,
        )

    # Build engine - class handles all parameters (imgsz, shapes, etc.)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    # Use model class's build method - it handles all parameters including dla_core
    model_class.build(
        onnx=onnx_path,
        output=engine_path,
        dla_core=0 if use_dla else None,
        verbose=False,
    )

    return engine_path


def detector_run(model_id: str, preprocessor: str = "cpu", *, use_dla: bool | None = None) -> None:
    """Check if a detector engine will run."""
    engine_path = build_detector(model_id, use_dla=use_dla)
    config = DETECTOR_CONFIG[model_id]
    model_class = config["model_class"]

    # Use class defaults for all parameters (input_range, etc.)
    detector = model_class(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        preprocessor=preprocessor,
        no_warn=True,
    )

    outputs = None
    for _ in range(NUM_ITERS):
        outputs = detector.mock_run()

    assert outputs is not None

    del detector


def detector_run_multiple(
    model_id: str,
    preprocessor: str = "cpu",
    count: int = 4,
    *,
    use_dla: bool | None = None,
) -> None:
    """Check if multiple detector engines can run at once."""
    engine_path = build_detector(model_id, use_dla=use_dla)
    config = DETECTOR_CONFIG[model_id]
    model_class = config["model_class"]

    detectors = [
        model_class(
            engine_path,
            conf_thres=0.25,
            warmup=False,
            preprocessor=preprocessor,
            no_warn=True,
        )
        for _ in range(count)
    ]

    outputs = None
    for _ in range(NUM_ITERS):
        outputs = [detector.mock_run() for detector in detectors]

    assert outputs is not None
    for o in outputs:
        assert o is not None

    for detector in detectors:
        del detector


def detector_run_in_thread(
    model_id: str, preprocessor: str = "cpu", *, use_dla: bool | None = None
) -> None:
    """Check if a detector engine can be run in another thread."""
    result = [False]

    def run(result: list[bool]) -> None:
        engine_path = build_detector(model_id, use_dla=use_dla)
        config = DETECTOR_CONFIG[model_id]
        model_class = config["model_class"]

        detector = model_class(
            engine_path,
            conf_thres=0.25,
            warmup=False,
            preprocessor=preprocessor,
            no_warn=True,
        )

        outputs = None
        for _ in range(NUM_ITERS):
            outputs = detector.mock_run()

        assert outputs is not None

        result[0] = True

        del detector

    thread = Thread(target=run, args=(result,), daemon=True)
    thread.start()

    thread.join()

    assert result[0]


def detector_run_multiple_threads(
    model_id: str,
    preprocessor: str = "cpu",
    count: int = 4,
    *,
    use_dla: bool | None = None,
) -> None:
    """Check if multiple detector engines can run across multiple threads."""
    results = [False] * count
    engine_path = build_detector(model_id, use_dla=use_dla)
    config = DETECTOR_CONFIG[model_id]
    model_class = config["model_class"]

    def run(tid: int, results: list[bool], engine_path: Path) -> None:
        detector = model_class(
            engine_path,
            conf_thres=0.25,
            warmup=False,
            preprocessor=preprocessor,
            no_warn=True,
        )

        outputs = None
        for _ in range(NUM_ITERS):
            outputs = detector.mock_run()

        assert outputs is not None

        results[tid] = True

        del detector

    threads = [
        Thread(
            target=run,
            args=(
                i,
                results,
                engine_path,
            ),
            daemon=True,
        )
        for i in range(count)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    for result in results:
        assert result


def detector_results(model_id: str, preprocessor: str = "cpu", *, use_dla: bool | None = None) -> None:
    """
    Check if the results are valid for a detector model.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.

    """
    engine_path = build_detector(model_id, use_dla=use_dla)
    config = DETECTOR_CONFIG[model_id]
    model_class = config["model_class"]

    detector = model_class(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        preprocessor=preprocessor,
        no_warn=True,
    )

    for gt, ipath in zip(
        [1, 4],
        IMAGE_PATHS,
    ):
        image = cv2.imread(ipath)
        if image is None:
            err_msg = f"Failed to read image: {ipath}"
            raise FileNotFoundError(err_msg)

        outputs = detector.run(image)
        bboxes = [bbox for (bbox, _, _) in detector.get_detections(outputs)]

        # check within +-2 bounding boxes from ground truth
        assert max(1, gt - 1) <= len(bboxes) <= gt + 1
        # we always have at least one detection per image
        assert len(bboxes) >= 1

    del detector


def detector_swapping_preproc_results(model_id: str, *, use_dla: bool | None = None) -> None:
    """
    Check if the results are valid when swapping preprocessing methods.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.

    """
    engine_path = build_detector(model_id, use_dla=use_dla)
    config = DETECTOR_CONFIG[model_id]
    model_class = config["model_class"]

    detector = model_class(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        preprocessor="cpu",
        no_warn=True,
    )

    for gt, ipath in zip(
        [1, 4],
        IMAGE_PATHS,
    ):
        image = cv2.imread(ipath)
        if image is None:
            err_msg = f"Failed to read image: {ipath}"
            raise FileNotFoundError(err_msg)

        for preproc in ["cpu", "cuda", "trt"]:
            tensor, ratios, padding = detector.preprocess(image, method=preproc, no_copy=True)
            outputs = detector.run(
                tensor,
                ratios,
                padding,
                preprocessed=True,
                postprocess=True,
                no_copy=True,
            )
            bboxes = [bbox for (bbox, _, _) in detector.get_detections(outputs)]

            # check within +-2 bounding boxes from ground truth
            assert max(1, gt - 1) <= len(bboxes) <= gt + 1
            # we always have at least one detection per image
            assert len(bboxes) >= 1

    del detector


def detector_pagelocked_perf(model_id: str, *, use_dla: bool | None = None) -> None:
    """Check pagelocked memory performance for a detector model."""
    engine_path = build_detector(model_id, use_dla=use_dla)
    config = DETECTOR_CONFIG[model_id]
    model_class = config["model_class"]

    detector = model_class(
        engine_path,
        conf_thres=0.25,
        warmup=True,
        preprocessor="cuda",
        pagelocked_mem=False,
    )
    detector_pagelocked = model_class(
        engine_path,
        conf_thres=0.25,
        warmup=True,
        preprocessor="cuda",
        pagelocked_mem=True,
    )

    times = []
    times_pagelocked = []

    for _ in range(NUM_ITERS):
        t0 = time.time()
        detector.mock_run()
        t1 = time.time()
        times.append(t1 - t0)

        t00 = time.time()
        detector_pagelocked.mock_run()
        t11 = time.time()
        times_pagelocked.append(t11 - t00)

    detector_mean = sum(times) / len(times)
    detector_pagelocked_mean = sum(times_pagelocked) / len(times_pagelocked)
    speedup = detector_mean / detector_pagelocked_mean

    assert speedup > 1.0
    print(f"Detector Pagelocked Speedup: {speedup:.2f}x")


# Backward compatibility aliases for YOLO (mapping int versions to string IDs)
_VERSION_TO_ID: dict[int, str] = {
    0: "yolox",
    3: "yolov3",
    5: "yolov5",
    7: "yolov7",
    8: "yolov8",
    9: "yolov9",
    10: "yolov10",
    11: "yolov11",
    12: "yolov12",
    13: "yolov13",
}


def build_yolo(version: int, *, use_dla: bool | None = None) -> Path:
    """Build a YOLO engine (backward compatibility)."""
    model_id = _VERSION_TO_ID[version]
    return build_detector(model_id, use_dla=use_dla)


def yolo_run(version: int, preprocessor: str = "cpu", *, use_dla: bool | None = None) -> None:
    """Check if a YOLO engine will run (backward compatibility)."""
    model_id = _VERSION_TO_ID[version]
    detector_run(model_id, preprocessor=preprocessor, use_dla=use_dla)


def yolo_run_multiple(
    version: int,
    preprocessor: str = "cpu",
    count: int = 4,
    *,
    use_dla: bool | None = None,
) -> None:
    """Check if multiple YOLO engines can run at once (backward compatibility)."""
    model_id = _VERSION_TO_ID[version]
    detector_run_multiple(model_id, preprocessor=preprocessor, count=count, use_dla=use_dla)


def yolo_run_in_thread(
    version: int, preprocessor: str = "cpu", *, use_dla: bool | None = None
) -> None:
    """Check if a YOLO engine can be run in another thread (backward compatibility)."""
    model_id = _VERSION_TO_ID[version]
    detector_run_in_thread(model_id, preprocessor=preprocessor, use_dla=use_dla)


def yolo_run_multiple_threads(
    version: int,
    preprocessor: str = "cpu",
    count: int = 4,
    *,
    use_dla: bool | None = None,
) -> None:
    """Check if multiple YOLO engines can run across multiple threads (backward compatibility)."""
    model_id = _VERSION_TO_ID[version]
    detector_run_multiple_threads(model_id, preprocessor=preprocessor, count=count, use_dla=use_dla)


def yolo_results(version: int, preprocessor: str = "cpu", *, use_dla: bool | None = None) -> None:
    """Check if the results are valid for a YOLO model (backward compatibility)."""
    model_id = _VERSION_TO_ID[version]
    detector_results(model_id, preprocessor=preprocessor, use_dla=use_dla)


def yolo_swapping_preproc_results(version: int, *, use_dla: bool | None = None) -> None:
    """Check swapping preprocessing results for YOLO (backward compatibility)."""
    model_id = _VERSION_TO_ID[version]
    detector_swapping_preproc_results(model_id, use_dla=use_dla)


def yolo_pagelocked_perf(version: int, *, use_dla: bool | None = None) -> None:
    """Check pagelocked memory performance for YOLO (backward compatibility)."""
    model_id = _VERSION_TO_ID[version]
    detector_pagelocked_perf(model_id, use_dla=use_dla)


# Backward compatibility aliases for DETR
def build_detr(model_name: str, *, use_dla: bool | None = None) -> Path:
    """Build a DETR engine (backward compatibility)."""
    return build_detector(model_name, use_dla=use_dla)


def detr_run(model_name: str, preprocessor: str = "cpu", *, use_dla: bool | None = None) -> None:
    """Check if a DETR engine will run (backward compatibility)."""
    detector_run(model_name, preprocessor=preprocessor, use_dla=use_dla)


def detr_results(model_name: str, preprocessor: str = "cpu", *, use_dla: bool | None = None) -> None:
    """Check if the results are valid for a DETR model (backward compatibility)."""
    detector_results(model_name, preprocessor=preprocessor, use_dla=use_dla)
