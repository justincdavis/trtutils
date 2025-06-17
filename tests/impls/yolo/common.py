# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from pathlib import Path
from threading import Thread

import cv2

import trtutils

from .paths import ENGINE_PATHS, IMAGE_PATHS, ONNX_PATHS

DLA_ENGINES = 2
GPU_ENGINES = 4
NUM_ITERS = 100
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
    onnx_path = ONNX_PATHS[version]
    engine_path = ENGINE_PATHS[version]
    if engine_path.exists():
        return engine_path

    trtutils.builder.build_engine(
        onnx_path,
        engine_path,
        timing_cache=Path(__file__).parent / "timing.cache",
        dla_core=0 if use_dla else None,
        gpu_fallback=bool(use_dla),
        fp16=True,
        shapes=[("images", (1, 3, 640, 640))] if version == YOLOV9_VERSION else None,
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
        no_warn=True,
    )

    outputs = None
    for _ in range(NUM_ITERS):
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
            no_warn=True,
        )
        for _ in range(count)
    ]

    outputs = None
    for _ in range(NUM_ITERS):
        outputs = [yolo.mock_run() for yolo in yolos]

    assert outputs is not None
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
            no_warn=True,
        )

        outputs = None
        for _ in range(NUM_ITERS):
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
    engine_path = build_yolo(version, use_dla=use_dla)

    def run(tid: int, results: list[bool], engine_path: Path) -> None:
        scale = (0, 1) if version != 0 else (0, 255)
        yolo = trtutils.impls.yolo.YOLO(
            engine_path,
            conf_thres=0.25,
            warmup=False,
            input_range=scale,
            preprocessor=preprocessor,
            no_warn=True,
        )

        outputs = None
        for _ in range(NUM_ITERS):
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
        no_warn=True,
    )

    for gt, ipath in zip(
        [1, 4],
        IMAGE_PATHS,
    ):
        image = cv2.imread(ipath)

        outputs = yolo.run(image)
        bboxes = [bbox for (bbox, _, _) in yolo.get_detections(outputs)]

        # check within +-2 bounding boxes from ground truth
        assert max(1, gt - 1) <= len(bboxes) <= gt + 1
        # we always have at least one detection per image
        assert len(bboxes) >= 1

    del yolo


def yolo_swapping_preproc_results(version: int, *, use_dla: bool | None = None) -> None:
    """Check if the results are valid for a YOLO model."""
    engine_path = build_yolo(version, use_dla=use_dla)

    scale = (0, 1) if version != 0 else (0, 255)
    yolo = trtutils.impls.yolo.YOLO(
        engine_path,
        conf_thres=0.25,
        warmup=False,
        input_range=scale,
        preprocessor="cpu",
        no_warn=True,
    )

    for gt, ipath in zip(
        [1, 4],
        IMAGE_PATHS,
    ):
        image = cv2.imread(ipath)
        for preproc in ["cpu", "cuda", "trt"]:
            tensor, ratios, padding = yolo.preprocess(
                image, method=preproc, no_copy=True
            )
            outputs = yolo.run(
                tensor,
                ratios,
                padding,
                preprocessed=True,
                postprocess=True,
                no_copy=True,
            )
            bboxes = [bbox for (bbox, _, _) in yolo.get_detections(outputs)]

            # check within +-2 bounding boxes from ground truth
            assert max(1, gt - 1) <= len(bboxes) <= gt + 1
            # we always have at least one detection per image
            assert len(bboxes) >= 1

    del yolo


def yolo_pagelocked_perf(version: int, *, use_dla: bool | None = None) -> None:
    """Check if the results are valid for a YOLO model."""
    engine_path = build_yolo(version, use_dla=use_dla)

    scale = (0, 1) if version != 0 else (0, 255)
    yolo = trtutils.impls.yolo.YOLO(
        engine_path,
        conf_thres=0.25,
        warmup=True,
        input_range=scale,
        preprocessor="cuda",
        pagelocked_mem=False,
    )
    yolo_pagelocked = trtutils.impls.yolo.YOLO(
        engine_path,
        conf_thres=0.25,
        warmup=True,
        input_range=scale,
        preprocessor="cuda",
        pagelocked_mem=True,
    )

    times = []
    times_pagelocked = []

    for _ in range(NUM_ITERS):
        t0 = time.time()
        yolo.mock_run()
        t1 = time.time()
        times.append(t1 - t0)

        t00 = time.time()
        yolo_pagelocked.mock_run()
        t11 = time.time()
        times_pagelocked.append(t11 - t00)

    yolo_mean = sum(times) / len(times)
    yolo_pagelocked_mean = sum(times_pagelocked) / len(times_pagelocked)
    speedup = yolo_mean / yolo_pagelocked_mean

    assert speedup > 1.0
    print(f"YOLO Pagelocked Speedup: {speedup:.2f}x")
