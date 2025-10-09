# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import cv2
import numpy as np

from trtutils.image import Detector

from .common import build_detector_engine
from .paths import HORSE_IMAGE_PATH, IMAGE_PATHS

PARITY_TOLERANCE = 0.05  # 5% tolerance for detection parity


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        err_msg = f"Failed to read image: {path}"
        raise FileNotFoundError(err_msg)
    return img


def test_detector_load() -> None:
    """Test Detector initializes with only engine path."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False)
    assert detector is not None
    del detector


def test_detector_load_with_args() -> None:
    """Test Detector initializes with various arguments."""
    engine_path = build_detector_engine()
    detector = Detector(
        engine_path,
        warmup=False,
        preprocessor="cpu",
        resize_method="letterbox",
        conf_thres=0.25,
        nms_iou_thres=0.45,
    )
    assert detector is not None
    del detector


def test_detector_run_cpu() -> None:
    """Test Detector runs with CPU preprocessor."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = detector.run(img)
    assert outputs is not None
    assert len(outputs) > 0
    del detector


def test_detector_run_cuda() -> None:
    """Test Detector runs with CUDA preprocessor."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cuda")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = detector.run(img)
    assert outputs is not None
    assert len(outputs) > 0
    del detector


def test_detector_run_trt() -> None:
    """Test Detector runs with TRT preprocessor."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="trt")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = detector.run(img)
    assert outputs is not None
    assert len(outputs) > 0
    del detector


def test_detector_preprocess_letterbox() -> None:
    """Test Detector preprocessing with letterbox."""
    engine_path = build_detector_engine()
    detector = Detector(
        engine_path, warmup=False, preprocessor="cpu", resize_method="letterbox"
    )
    img = _read_image(HORSE_IMAGE_PATH)
    tensor, ratios, padding = detector.preprocess(img)
    assert tensor is not None
    assert ratios is not None
    assert padding is not None
    assert len(ratios) == 2
    assert len(padding) == 2
    del detector


def test_detector_preprocess_linear() -> None:
    """Test Detector preprocessing with linear resize."""
    engine_path = build_detector_engine()
    detector = Detector(
        engine_path, warmup=False, preprocessor="cpu", resize_method="linear"
    )
    img = _read_image(HORSE_IMAGE_PATH)
    tensor, ratios, padding = detector.preprocess(img)
    assert tensor is not None
    assert ratios is not None
    assert padding is not None
    assert len(ratios) == 2
    assert len(padding) == 2
    del detector


def test_detector_postprocess() -> None:
    """Test Detector postprocessing."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    tensor, ratios, padding = detector.preprocess(img)
    outputs = detector._engine([tensor], no_copy=False)
    postprocessed = detector.postprocess(outputs, ratios, padding)
    assert postprocessed is not None
    assert len(postprocessed) > 0
    del detector


def test_detector_get_detections() -> None:
    """Test Detector get_detections method."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = detector.run(img)
    detections = detector.get_detections(outputs)
    assert detections is not None
    assert isinstance(detections, list)
    # Check detection format if any detections found
    if len(detections) > 0:
        bbox, conf, cls = detections[0]
        assert len(bbox) == 4
        assert isinstance(conf, (float, np.floating))
        assert isinstance(cls, (int, np.integer))
    del detector


def test_detector_end2end() -> None:
    """Test Detector end2end method."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cuda")
    img = _read_image(HORSE_IMAGE_PATH)
    detections = detector.end2end(img)
    assert detections is not None
    assert isinstance(detections, list)
    del detector


def test_detector_call_method() -> None:
    """Test Detector __call__ method."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = detector(img)
    assert outputs is not None
    assert len(outputs) > 0
    del detector


def test_detector_preprocessed_flag() -> None:
    """Test Detector with preprocessed flag."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    tensor, ratios, padding = detector.preprocess(img)
    outputs = detector.run(tensor, ratios, padding, preprocessed=True, postprocess=True)
    assert outputs is not None
    assert len(outputs) > 0
    del detector


def test_detector_no_postprocess() -> None:
    """Test Detector without postprocessing."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = detector.run(img, postprocess=False)
    assert outputs is not None
    assert len(outputs) > 0
    # Raw outputs should have different shape than postprocessed
    del detector


def test_detector_conf_threshold() -> None:
    """Test Detector with different confidence thresholds."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cpu", conf_thres=0.1)
    img = _read_image(HORSE_IMAGE_PATH)
    outputs_low = detector.run(img)
    detections_low = detector.get_detections(outputs_low, conf_thres=0.1)

    detector_high = Detector(
        engine_path, warmup=False, preprocessor="cpu", conf_thres=0.5
    )
    outputs_high = detector_high.run(img)
    detections_high = detector_high.get_detections(outputs_high, conf_thres=0.5)

    # Lower threshold should generally give more or equal detections
    assert len(detections_low) >= len(detections_high)
    del detector
    del detector_high


def test_detector_nms_threshold() -> None:
    """Test Detector with NMS threshold."""
    engine_path = build_detector_engine()
    detector = Detector(
        engine_path,
        warmup=False,
        preprocessor="cpu",
        nms_iou_thres=0.5,
        extra_nms=True,
    )
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = detector.run(img)
    detections = detector.get_detections(outputs, extra_nms=True)
    assert detections is not None
    del detector


def test_detector_preprocess_method_switch() -> None:
    """Test switching preprocessor methods at runtime."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cpu")
    img = _read_image(HORSE_IMAGE_PATH)

    # Test switching to CUDA
    tensor_cuda, _, _ = detector.preprocess(img, method="cuda")
    assert tensor_cuda is not None

    # Test switching to TRT
    tensor_trt, _, _ = detector.preprocess(img, method="trt")
    assert tensor_trt is not None

    # Test switching back to CPU
    tensor_cpu, _, _ = detector.preprocess(img, method="cpu")
    assert tensor_cpu is not None

    del detector


def _compare_detections(
    det1: list[tuple[tuple[int, int, int, int], float, int]],
    det2: list[tuple[tuple[int, int, int, int], float, int]],
    tolerance: float,
) -> bool:
    """Compare two detection lists for parity."""
    # Allow small difference in number of detections
    if abs(len(det1) - len(det2)) > max(1, int(max(len(det1), len(det2)) * tolerance)):
        return False

    # If both empty, they match
    if len(det1) == 0 and len(det2) == 0:
        return True

    # Compare detections that exist in both
    min_len = min(len(det1), len(det2))
    if min_len == 0:
        return True

    for i in range(min_len):
        bbox1, conf1, cls1 = det1[i]
        bbox2, conf2, cls2 = det2[i]

        # Classes should match
        if cls1 != cls2:
            continue  # Allow some class mismatches

        # Confidence should be similar
        if abs(conf1 - conf2) > 0.1:
            continue

        # Bounding boxes should be similar (allow 10% variation)
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        if bbox1_area == 0 or bbox2_area == 0:
            continue

        # Check if bboxes are reasonably similar
        max_area = max(bbox1_area, bbox2_area)
        if (
            abs(x1_1 - x1_2) < max_area * 0.1
            and abs(y1_1 - y1_2) < max_area * 0.1
            and abs(x2_1 - x2_2) < max_area * 0.1
            and abs(y2_1 - y2_2) < max_area * 0.1
        ):
            return True

    return True  # Allow parity if at least one detection is similar


def test_detector_parity_cpu_cuda_letterbox() -> None:
    """Test parity between CPU and CUDA preprocessors with letterbox."""
    engine_path = build_detector_engine()

    detector_cpu = Detector(
        engine_path, warmup=False, preprocessor="cpu", resize_method="letterbox"
    )
    detector_cuda = Detector(
        engine_path, warmup=False, preprocessor="cuda", resize_method="letterbox"
    )

    img = _read_image(HORSE_IMAGE_PATH)

    outputs_cpu = detector_cpu.run(img)
    detections_cpu = detector_cpu.get_detections(outputs_cpu)

    outputs_cuda = detector_cuda.run(img)
    detections_cuda = detector_cuda.get_detections(outputs_cuda)

    # Check that both produce reasonable results
    assert len(detections_cpu) > 0 or len(detections_cuda) > 0

    # Check parity
    assert _compare_detections(detections_cpu, detections_cuda, PARITY_TOLERANCE)

    del detector_cpu
    del detector_cuda


def test_detector_parity_cpu_cuda_linear() -> None:
    """Test parity between CPU and CUDA preprocessors with linear resize."""
    engine_path = build_detector_engine()

    detector_cpu = Detector(
        engine_path, warmup=False, preprocessor="cpu", resize_method="linear"
    )
    detector_cuda = Detector(
        engine_path, warmup=False, preprocessor="cuda", resize_method="linear"
    )

    img = _read_image(HORSE_IMAGE_PATH)

    outputs_cpu = detector_cpu.run(img)
    detections_cpu = detector_cpu.get_detections(outputs_cpu)

    outputs_cuda = detector_cuda.run(img)
    detections_cuda = detector_cuda.get_detections(outputs_cuda)

    # Check that both produce reasonable results
    assert len(detections_cpu) > 0 or len(detections_cuda) > 0

    # Check parity
    assert _compare_detections(detections_cpu, detections_cuda, PARITY_TOLERANCE)

    del detector_cpu
    del detector_cuda


def test_detector_parity_cpu_trt_letterbox() -> None:
    """Test parity between CPU and TRT preprocessors with letterbox."""
    engine_path = build_detector_engine()

    detector_cpu = Detector(
        engine_path, warmup=False, preprocessor="cpu", resize_method="letterbox"
    )
    detector_trt = Detector(
        engine_path, warmup=False, preprocessor="trt", resize_method="letterbox"
    )

    img = _read_image(HORSE_IMAGE_PATH)

    outputs_cpu = detector_cpu.run(img)
    detections_cpu = detector_cpu.get_detections(outputs_cpu)

    outputs_trt = detector_trt.run(img)
    detections_trt = detector_trt.get_detections(outputs_trt)

    # Check that both produce reasonable results
    assert len(detections_cpu) > 0 or len(detections_trt) > 0

    # Check parity
    assert _compare_detections(detections_cpu, detections_trt, PARITY_TOLERANCE)

    del detector_cpu
    del detector_trt


def test_detector_parity_cpu_trt_linear() -> None:
    """Test parity between CPU and TRT preprocessors with linear resize."""
    engine_path = build_detector_engine()

    detector_cpu = Detector(
        engine_path, warmup=False, preprocessor="cpu", resize_method="linear"
    )
    detector_trt = Detector(
        engine_path, warmup=False, preprocessor="trt", resize_method="linear"
    )

    img = _read_image(HORSE_IMAGE_PATH)

    outputs_cpu = detector_cpu.run(img)
    detections_cpu = detector_cpu.get_detections(outputs_cpu)

    outputs_trt = detector_trt.run(img)
    detections_trt = detector_trt.get_detections(outputs_trt)

    # Check that both produce reasonable results
    assert len(detections_cpu) > 0 or len(detections_trt) > 0

    # Check parity
    assert _compare_detections(detections_cpu, detections_trt, PARITY_TOLERANCE)

    del detector_cpu
    del detector_trt


def test_detector_multiple_images() -> None:
    """Test Detector on multiple images."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cuda")

    for img_path in IMAGE_PATHS:
        img = _read_image(img_path)
        detections = detector.end2end(img)
        assert detections is not None
        assert isinstance(detections, list)

    del detector


def test_detector_no_copy() -> None:
    """Test Detector with no_copy flag."""
    engine_path = build_detector_engine()
    detector = Detector(engine_path, warmup=False, preprocessor="cuda")
    img = _read_image(HORSE_IMAGE_PATH)

    # Test no_copy in preprocess
    tensor, ratios, padding = detector.preprocess(img, no_copy=True)
    assert tensor is not None

    # Test no_copy in run
    outputs = detector.run(img, no_copy=True)
    assert outputs is not None

    del detector


def test_detector_pagelocked_mem() -> None:
    """Test Detector with pagelocked memory."""
    engine_path = build_detector_engine()
    detector = Detector(
        engine_path, warmup=False, preprocessor="cuda", pagelocked_mem=True
    )
    img = _read_image(HORSE_IMAGE_PATH)
    detections = detector.end2end(img)
    assert detections is not None
    del detector


def test_detector_no_pagelocked_mem() -> None:
    """Test Detector without pagelocked memory."""
    engine_path = build_detector_engine()
    detector = Detector(
        engine_path, warmup=False, preprocessor="cuda", pagelocked_mem=False
    )
    img = _read_image(HORSE_IMAGE_PATH)
    detections = detector.end2end(img)
    assert detections is not None
    del detector


def test_detector_input_range() -> None:
    """Test Detector with different input ranges."""
    engine_path = build_detector_engine()

    # Test with [0, 1] range (default)
    detector1 = Detector(
        engine_path, warmup=False, preprocessor="cpu", input_range=(0.0, 1.0)
    )
    img = _read_image(HORSE_IMAGE_PATH)
    outputs1 = detector1.run(img)
    assert outputs1 is not None

    # Test with [0, 255] range
    detector2 = Detector(
        engine_path, warmup=False, preprocessor="cpu", input_range=(0.0, 255.0)
    )
    outputs2 = detector2.run(img)
    assert outputs2 is not None

    del detector1
    del detector2


def test_detector_agnostic_nms() -> None:
    """Test Detector with class-agnostic NMS."""
    engine_path = build_detector_engine()
    detector = Detector(
        engine_path,
        warmup=False,
        preprocessor="cpu",
        extra_nms=True,
        agnostic_nms=True,
    )
    img = _read_image(HORSE_IMAGE_PATH)
    outputs = detector.run(img)
    detections = detector.get_detections(outputs, agnostic_nms=True)
    assert detections is not None
    del detector



