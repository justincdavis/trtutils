# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import cv2

import trtutils

try:
    from paths import ENGINE_PATHS, IMAGE_PATHS, ONNX_PATHS
except ModuleNotFoundError:
    from .paths import ENGINE_PATHS, IMAGE_PATHS, ONNX_PATHS


def build_yolo(version: int, *, use_dla: bool | None = None) -> Path:
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
        )
    else:
        trtutils.trtexec.build_engine(
            onnx_path,
            engine_path,
            use_dla_core=0 if use_dla else None,
            allow_gpu_fallback=True if use_dla else None,
            shapes=[("images", (1, 3, 640, 640))],
        )

    return engine_path


def yolo_results(
    version: int, preprocessor: str = "cpu", *, use_dla: bool | None = None
) -> None:
    engine_path = build_yolo(version, use_dla=use_dla)

    scale = (0, 1) if version != 0 else (0, 255)
    trt_model = trtutils.impls.yolo.YOLO(
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

        outputs = trt_model.run(image)
        trt_bboxes = [bbox for (bbox, _, _) in trt_model.get_detections(outputs)]

        # check within +-2 bounding boxes from ground truth
        assert max(0, gt - 2) <= len(trt_bboxes) <= gt + 2
        # we always have at least one detection per image
        assert len(trt_bboxes) >= 1

    del trt_model


def bboxes_close(
    bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int], tolerance=2
) -> bool:
    for c1, c2 in zip(bbox1, bbox2):
        if abs(c1 - c2) > tolerance:
            return False
    return True
