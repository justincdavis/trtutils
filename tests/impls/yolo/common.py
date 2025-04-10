# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from threading import Thread
from typing import TYPE_CHECKING

import cv2

import trtutils

if TYPE_CHECKING:
    from pathlib import Path

try:
    from paths import ENGINE_PATHS, IMAGE_PATHS, ONNX_PATHS
except ModuleNotFoundError:
    from .paths import ENGINE_PATHS, IMAGE_PATHS, ONNX_PATHS

DLA_ENGINES = 2
GPU_ENGINES = 4


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
    onnx_path = ONNX_PATHS[version]
    engine_path = ENGINE_PATHS[version]
    if engine_path.exists():
        return engine_path

    if version != 9:
        trtutils.builder.build_engine(
            onnx_path,
            engine_path,
            use_dla_core=0 if use_dla else None,
            allow_gpu_fallback=True if use_dla else None,
            fp16=True,
        )
    else:
        trtutils.trtexec.build_engine(
            onnx_path,
            engine_path,
            use_dla_core=0 if use_dla else None,
            allow_gpu_fallback=True if use_dla else None,
            shapes=[("images", (1, 3, 640, 640))],
            fp16=True,
        )

    return engine_path


def yolo_run(
    version: int, preprocessor: str = "cpu", *, use_dla: bool | None = None
) -> None:
    """Check if a YOLO engine will run."""
    engine_path = build_yolo(version, use_dla=use_dla)

    scale = (0, 1) if version != 0 else (0, 255)
    yolo = trtutils.impls.yolo.YOLO(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        input_range=scale,
        preprocessor=preprocessor,
    )

    outputs = yolo.mock_run()

    assert outputs is not None

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
        trtutils.impls.yolo.YOLO(
            engine_path,
            conf_thres=0.25,
            warmup=False,
            input_range=scale,
            preprocessor=preprocessor,
        )
        for _ in range(count)
    ]

    outputs = [yolo.mock_run() for yolo in yolos]

    for o in outputs:
        assert o is not None

    for yolo in yolos:
        del yolo


def yolo_run_in_thread(
    version: int, preprocessor: str = "cpu", *, use_dla: bool | None = None
) -> None:
    """Check if a YOLO engine can be run in another thread."""
    result = [False]

    def run(result: list[bool]) -> None:
        engine_path = build_yolo(version, use_dla=use_dla)

        scale = (0, 1) if version != 0 else (0, 255)
        yolo = trtutils.impls.yolo.YOLO(
            engine_path,
            conf_thres=0.25,
            warmup=False,
            input_range=scale,
            preprocessor=preprocessor,
        )

        outputs = yolo.mock_run()

        assert outputs is not None

        result[0] = True

        del yolo

    thread = Thread(target=run, args=(result,), daemon=True)
    thread.start()

    thread.join()

    assert result[0]


def yolo_run_multiple_threads(
    version: int,
    preprocessor: str = "cpu",
    count: int = 4,
    *,
    use_dla: bool | None = None,
) -> None:
    """Check if multiple YOLO engines can run across multiple threads."""
    results = [False] * count

    def run(tid: int, results: list[bool]) -> None:
        engine_path = build_yolo(version, use_dla=use_dla)

        scale = (0, 1) if version != 0 else (0, 255)
        yolo = trtutils.impls.yolo.YOLO(
            engine_path,
            conf_thres=0.25,
            warmup=False,
            input_range=scale,
            preprocessor=preprocessor,
        )

        outputs = yolo.mock_run()

        assert outputs is not None

        results[tid] = True

        del yolo

    threads = [
        Thread(
            target=run,
            args=(
                i,
                results,
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


def yolo_results(
    version: int, preprocessor: str = "cpu", *, use_dla: bool | None = None
) -> None:
    """Check if the results are valid for a YOLO model."""
    engine_path = build_yolo(version, use_dla=use_dla)

    scale = (0, 1) if version != 0 else (0, 255)
    yolo = trtutils.impls.yolo.YOLO(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        input_range=scale,
        preprocessor=preprocessor,
    )

    for gt, ipath in zip(
        [1, 4],
        IMAGE_PATHS,
    ):
        image = cv2.imread(ipath)

        outputs = yolo.run(image)
        bboxes = [bbox for (bbox, _, _) in yolo.get_detections(outputs)]

        # check within +-2 bounding boxes from ground truth
        assert max(0, gt - 2) <= len(bboxes) <= gt + 2
        # we always have at least one detection per image
        assert len(bboxes) >= 1

    del yolo


def bboxes_close(
    bbox1: tuple[int, int, int, int],
    bbox2: tuple[int, int, int, int],
    tolerance: int = 2,
) -> bool:
    """
    Check if two bboxes are close to each other.

    Parameters
    ----------
    bbox1 : tuple[int, int, int, int]
        Bbox1
    bbox2 : tuple[int, int, int, int]
        Bbox2
    tolerance : int, optional
        The pixel value tolerance

    Returns
    -------
    bool
        Whether or not they are close.

    """
    return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(bbox1, bbox2))
