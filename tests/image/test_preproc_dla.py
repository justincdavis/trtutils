# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/_dla.py and _dla_network.py -- the DLA preprocessor."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from trtutils import Buffer, MemoryLocation
from trtutils._flags import FLAGS
from trtutils.compat._libs import trt
from trtutils.core import create_engine
from trtutils.image import Detector
from trtutils.image.preprocessors import CUDAPreprocessor, DLAPreprocessor
from trtutils.image.preprocessors._cpu import letterbox
from trtutils.image.preprocessors._dla_network import (
    build_dla_preproc,
    normalization,
    plan_resize,
    resize_stages,
)

RANGE = (0.0, 1.0)
IMAGENET = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
DLA_SYSFS = Path("/sys/devices/platform/bus@0/13e00000.host1x")
DLA_DEVICES = ("15880000.nvdla0", "158c0000.nvdla1")
CONFIGS = [
    pytest.param((1080, 1920), (640, 640), "letterbox", id="1080p-640-letterbox-k3"),
    pytest.param((720, 1280), (320, 320), "letterbox", id="720p-320-letterbox-k4"),
    pytest.param((1080, 1920), (320, 320), "letterbox", id="1080p-320-letterbox-k6"),
    pytest.param((720, 1280), (640, 360), "linear", id="720p-640x360-linear-k2"),
    pytest.param((1080, 1920), (640, 540), "linear", id="1080p-640x540-linear-k3x2"),
]


def _frame(images, height: int, width: int) -> np.ndarray:
    return cv2.resize(images["people"].array, (width, height), interpolation=cv2.INTER_AREA)


def _stage_matrix(size: int, stride: int, taps: np.ndarray) -> np.ndarray:
    matrix = np.zeros((size // stride, size))
    for out in range(size // stride):
        span = taps[: size - out * stride]
        matrix[out, out * stride : out * stride + len(span)] = span
    return matrix


def _bilinear(size: int, factor: int) -> np.ndarray:
    matrix = np.zeros((size // factor, size))
    for out in range(size // factor):
        x = (out + 0.5) * factor - 0.5
        left = math.floor(x)
        matrix[out, left] += 1.0 - (x - left)
        matrix[out, min(left + 1, size - 1)] += x - left
    return matrix


def _active_ms() -> list[int]:
    return [
        int((DLA_SYSFS / device / "power" / "runtime_active_time").read_text())
        for device in DLA_DEVICES
    ]


def _wait_dla_idle(timeout: float = 10.0) -> None:
    status = [DLA_SYSFS / device / "power" / "runtime_status" for device in DLA_DEVICES]
    deadline = time.monotonic() + timeout
    while any(path.read_text().strip() != "suspended" for path in status):
        if time.monotonic() > deadline:
            pytest.skip("a DLA is busy")
        time.sleep(0.1)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("shape", "size", "method", "factors", "top", "rows"),
    [
        pytest.param((1080, 1920), (640, 640), "letterbox", (3, 3), 140, 360, id="1080p-640"),
        pytest.param((720, 1280), (320, 320), "letterbox", (4, 4), 70, 180, id="720p-320"),
        pytest.param((2160, 3840), (640, 640), "letterbox", (6, 6), 140, 360, id="4k-640"),
        pytest.param((480, 640), (640, 640), "letterbox", (1, 1), 80, 480, id="480p-640"),
        pytest.param((1089, 1920), (640, 640), "letterbox", (3, 3), 138, 363, id="odd-padding"),
        pytest.param((1080, 1920), (640, 540), "linear", (2, 3), 0, 540, id="linear-k3x2"),
    ],
)
def test_plan_resize_matches_gpu_geometry(shape, size, method, factors, top, rows) -> None:
    """Plans give the GPU preprocessors' ratios and padding, and the rows the engine writes."""
    plan = plan_resize(*shape, size, method)
    assert plan.factors == factors
    assert plan.top == top
    assert plan.rows == rows
    if method == "letterbox":
        _, ratios, padding = letterbox(np.zeros((*shape, 3), dtype=np.uint8), size)
        assert plan.ratios == ratios
        assert plan.padding == padding
    else:
        assert plan.ratios == (size[0] / shape[1], size[1] / shape[0])
        assert plan.padding == (0.0, 0.0)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("shape", "size", "method", "match"),
    [
        pytest.param((1920, 1080), (640, 640), "letterbox", "whole factor", id="portrait"),
        pytest.param((700, 1000), (640, 640), "letterbox", "whole factor", id="fractional"),
        pytest.param((1000, 1920), (640, 640), "linear", "whole factors", id="linear-rows"),
        pytest.param((1080, 1920), (224, 224), "center_crop", "letterbox and linear", id="crop"),
    ],
)
def test_plan_resize_rejects(shape, size, method, match) -> None:
    """Resizes the DLA cannot do raise ValueError naming the reason."""
    with pytest.raises(ValueError, match=match):
        plan_resize(*shape, size, method)


@pytest.mark.cpu
@pytest.mark.parametrize("factor", [1, 2, 3, 4, 6, 8, 9, 10, 12, 16, 24, 32])
def test_resize_stages_reproduce_bilinear(factor) -> None:
    """The stride/tap stages compose to the GPU kernels' bilinear sampling at 1 / factor."""
    size = factor * 20
    stages = resize_stages(factor)
    assert stages[0][0] <= 4
    assert all(stride <= 8 for stride, _ in stages)
    assert math.prod(stride for stride, _ in stages) == factor
    matrix = np.eye(size)
    for stride, taps in stages:
        matrix = _stage_matrix(matrix.shape[0], stride, taps) @ matrix
    np.testing.assert_allclose(matrix, _bilinear(size, factor))


@pytest.mark.cpu
@pytest.mark.parametrize("factor", [5, 7, 11, 13, 40])
def test_resize_stages_reject_factor(factor) -> None:
    """Factors without a DLA stride split raise ValueError."""
    with pytest.raises(ValueError, match=f"factor of {factor}"):
        resize_stages(factor)


@pytest.mark.cpu
def test_dla_preprocessor_without_dla_raises() -> None:
    """Creating a DLAPreprocessor on a system without a DLA raises RuntimeError."""
    if FLAGS.HAS_DLA:
        pytest.skip("this system has a DLA")
    with pytest.raises(RuntimeError, match="no DLA"):
        DLAPreprocessor((640, 640), RANGE, np.float16)


@pytest.mark.dla
@pytest.mark.parametrize(("shape", "size", "method"), CONFIGS)
@pytest.mark.parametrize("norm", [{}, IMAGENET], ids=["raw", "imagenet"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32], ids=["fp16", "fp32"])
def test_dla_matches_cuda(images, shape, size, method, norm, dtype) -> None:
    """DLA output is within about a level of the CUDA preprocessor, with the same ratios and padding."""
    frame = _frame(images, *shape)
    dla = DLAPreprocessor(size, RANGE, dtype, resize=method, **norm)
    cuda = CUDAPreprocessor(size, RANGE, np.float32, resize=method, **norm)
    result, ratios, padding = dla.preprocess(frame)
    expected, cuda_ratios, cuda_padding = cuda.preprocess(frame)
    assert result.dtype == dtype
    assert result.shape == expected.shape
    assert ratios == cuda_ratios
    assert padding == cuda_padding
    level = 1.0 / (255.0 * min(norm.get("std", (1.0,))))
    diff = np.abs(result.astype(np.float32) - expected)
    assert diff.max() <= 1.5 * level + 0.004
    assert diff.mean() <= 0.25 * level


@pytest.mark.dla
def test_dla_inputs_and_batches_match(images) -> None:
    """Host, mapped and device images give identical results, alone or in mixed-size batches."""
    frames = [_frame(images, 1080, 1920), _frame(images, 720, 1280)]
    dla = DLAPreprocessor((640, 640), RANGE, np.float16, batch_size=2)
    singles = [dla.preprocess(frame)[0][0] for frame in frames]
    for placement in ({"location": MemoryLocation.DEVICE}, {"pinned": True, "mapped": True}):
        for frame, single in zip(frames, singles):
            result, _, _ = dla.preprocess(Buffer.from_array(frame, **placement))
            np.testing.assert_array_equal(result[0], single)
    for order in ((0, 1), (1, 0), (0, 1)):
        batch, ratios, padding = dla.preprocess([frames[i] for i in order])
        for slot, i in enumerate(order):
            np.testing.assert_array_equal(batch[slot], singles[i])
        assert ratios == [
            plan_resize(*frames[i].shape[:2], (640, 640), "letterbox").ratios for i in order
        ]
        assert len(padding) == len(order)


@pytest.mark.dla
@pytest.mark.parametrize(("shape", "size", "method"), CONFIGS)
def test_dla_engine_runs_only_on_dla(cuda_stream, shape, size, method) -> None:
    """The built engine is one DLA subgraph reading DLA_HWC4 and writing CHW16, with no GPU layers."""
    plan = plan_resize(*shape, size, method)
    path = build_dla_preproc(plan, *normalization(RANGE, None, None), 0)
    engine, _, _, _ = create_engine(path, stream=cuda_stream, dla_core=0)
    inspector = engine.create_engine_inspector()
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    assert [layer["LayerType"] for layer in layers["Layers"]] == ["DLA"]
    # tensorrt 8.6 reports the DLA_HWC4 input of a 3-channel image as CHW4, the same bytes
    assert engine.get_tensor_format("input") in (trt.TensorFormat.CHW4, trt.TensorFormat.DLA_HWC4)
    for name in ("out_r", "out_g", "out_b"):
        assert engine.get_tensor_format(name) == trt.TensorFormat.CHW16
        assert tuple(engine.get_tensor_shape(name)) == (1, 16, plan.rows, size[0] // 16)


@pytest.mark.dla
@pytest.mark.parametrize("dla_core", [0, 1], ids=["dla0", "dla1"])
def test_dla_core_runs_only_that_core(images, dla_core) -> None:
    """A preprocessor on one DLA core keeps that core busy and leaves the other idle."""
    if dla_core >= FLAGS.NUM_DLA_CORES:
        pytest.skip(f"no DLA core {dla_core}")
    frame = Buffer.from_array(_frame(images, 1080, 1920), MemoryLocation.DEVICE)
    dla = DLAPreprocessor((640, 640), RANGE, np.float16, dla_core=dla_core)
    dla.preprocess(frame)
    _wait_dla_idle()
    before = _active_ms()
    for _ in range(50):
        dla.preprocess(frame)
    after = _active_ms()
    active = [end - start for start, end in zip(before, after)]
    assert active[dla_core] > 0
    assert active[1 - dla_core] == 0


@pytest.mark.dla
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"dla_core": 2}, "DLA core 2", id="core"),
        pytest.param({"output_shape": (650, 650)}, "multiple of 16", id="width"),
        pytest.param({"resize": "center_crop"}, "letterbox", id="crop"),
        pytest.param({"dtype": np.uint8}, "float16 or float32", id="dtype"),
    ],
)
def test_dla_preprocessor_rejects_config(kwargs, match) -> None:
    """Configurations the DLA path cannot run raise ValueError at construction."""
    args = {"output_shape": (640, 640), "output_range": RANGE, "dtype": np.float16, **kwargs}
    with pytest.raises(ValueError, match=match):
        DLAPreprocessor(**args)


@pytest.mark.dla
def test_dla_preprocessor_rejects_inputs(images, random_images) -> None:
    """Unsupported input sizes and oversized batches raise ValueError at the call."""
    dla = DLAPreprocessor((640, 640), RANGE, np.float16)
    with pytest.raises(ValueError, match="whole factor"):
        dla.preprocess(random_images(1, 700, 1000))
    with pytest.raises(ValueError, match="exceeds configured batch size"):
        dla.preprocess([_frame(images, 1080, 1920)] * 2)


@pytest.mark.dla
def test_detector_dla_matches_cuda(yolov10_engine, images) -> None:
    """A Detector with preprocessor='dla' finds the same objects as with 'cuda'."""
    frame = _frame(images, 1080, 1920)
    results = []
    for preprocessor in ("cuda", "dla"):
        detector = Detector(
            yolov10_engine,
            preprocessor=preprocessor,
            resize_method="letterbox",
            conf_thres=0.5,
            warmup=False,
        )
        results.append(sorted(detector.end2end(frame), key=lambda det: (det[2], det[0])))
    cuda, dla = results
    assert len(dla) == len(cuda) > 0
    for (box, score, cls), (cuda_box, cuda_score, cuda_cls) in zip(dla, cuda):
        assert cls == cuda_cls
        assert abs(score - cuda_score) < 0.05
        np.testing.assert_allclose(box, cuda_box, atol=3)
    assert isinstance(detector.get_preprocessor(), DLAPreprocessor)
