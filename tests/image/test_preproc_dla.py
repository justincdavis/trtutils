# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/preprocessors/_dla.py and _dla_network.py -- DLA planning and engines."""

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
from trtutils.image.preprocessors import CUDAPreprocessor, DLAPreprocessor, _dla
from trtutils.image.preprocessors._base import crop_geometry
from trtutils.image.preprocessors._cpu import resize_image
from trtutils.image.preprocessors._dla_network import (
    build_dla_preproc,
    normalization,
    plan_resize,
    resize_stages,
)
from trtutils.models import ResNet

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
CROPS = [
    pytest.param((1080, 1920), (2, 2), id="1080p-224-crop-k2"),
    pytest.param((720, 1280), (1, 1), id="720p-224-crop-k1"),
    pytest.param((2160, 3840), (4, 4), id="4k-224-crop-k4"),
    pytest.param((480, 640), (1, 1), id="480p-224-crop-k1"),
    pytest.param((480, 720), (1, 1), id="720x480-224-crop-k1"),
]
CROP = (224, 224)


def _frame(images, height: int, width: int, name: str = "people") -> np.ndarray:
    return cv2.resize(images[name].array, (width, height), interpolation=cv2.INTER_AREA)


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


def _crop_triangles(shape: tuple[int, int], factors: tuple[int, int]) -> list[np.ndarray]:
    """Row and column filters of torchvision's antialiased crop, over box samples of ``factors``."""
    (resized_width, resized_height, left, top), _, _ = crop_geometry(shape[1], shape[0], CROP)
    triangles = []
    for size, resized, start, count, factor in zip(
        shape, (resized_height, resized_width), (top, left), CROP[::-1], factors
    ):
        scale = size / resized
        support = max(scale / factor, 1.0)
        centres = (np.arange(start, start + count) + 0.5) * scale / factor - 0.5
        weights = np.maximum(1.0 - np.abs(np.arange(size // factor) - centres[:, None]) / support, 0)
        triangles.append(weights / weights.sum(axis=1, keepdims=True))
    return triangles


def _crop_maps(shape: tuple[int, int], factors: tuple[int, int]) -> list[np.ndarray]:
    """Row and column maps of the center crop: box filters by ``factors``, then triangle filters."""
    maps = []
    for triangle, size, factor in zip(_crop_triangles(shape, factors), shape, factors):
        matrix = np.zeros((len(triangle), size))
        matrix[:, : triangle.shape[1] * factor] = np.repeat(triangle, factor, axis=1) / factor
        maps.append(matrix)
    return maps


def _box_filter(image: np.ndarray, factors: tuple[int, int]) -> np.ndarray:
    rows, cols = (size // factor for size, factor in zip(image.shape, factors))
    blocks = image[: rows * factors[0], : cols * factors[1]].astype(np.float64)
    return blocks.reshape(rows, factors[0], cols, factors[1], -1).mean(axis=(1, 3))


def _apply(image: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return np.einsum("oh,hwc,pw->opc", rows, image.astype(np.float64), columns, optimize=True)


def _active_ms() -> list[int]:
    return [
        int((DLA_SYSFS / device / "power" / "runtime_active_time").read_text())
        for device in DLA_DEVICES
    ]


def _wait_dla_idle(timeout: float = 10.0) -> None:
    status = [DLA_SYSFS / device / "power" / "runtime_status" for device in DLA_DEVICES]
    if not all(path.exists() for path in status):
        pytest.skip("no Orin DLA power counters in sysfs")
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
    """Plans give the CPU resize's ratios and padding, as the GPU preprocessors do, and the rows."""
    plan = plan_resize(*shape, size, method)
    assert plan.factors == factors
    assert plan.top == top
    assert plan.rows == rows
    _, ratios, padding = resize_image(np.zeros((*shape, 3), dtype=np.uint8), size, method)
    assert plan.ratios == ratios
    assert plan.padding == padding


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("shape", "size", "method", "match"),
    [
        pytest.param((1920, 1080), (640, 640), "letterbox", "whole factor", id="portrait"),
        pytest.param((700, 1000), (640, 640), "letterbox", "whole factor", id="fractional"),
        pytest.param((1000, 1920), (640, 640), "linear", "whole factors", id="linear-rows"),
        pytest.param((1080, 1080), CROP, "center_crop", "multiple of 16", id="crop-width"),
        pytest.param((2160, 4096), CROP, "center_crop", "up to 3840", id="crop-wide"),
        pytest.param((1080, 1920), (640, 640), "nearest", "and center_crop", id="method"),
        pytest.param((1080, 1920), (640, 640), "pad", r"not pad\. Resize it on the CPU", id="pad"),
    ],
)
def test_plan_resize_rejects(shape, size, method, match) -> None:
    """Resizes the DLA cannot do raise ValueError naming the reason."""
    with pytest.raises(ValueError, match=match):
        plan_resize(*shape, size, method)


@pytest.mark.cpu
@pytest.mark.parametrize(("shape", "factors"), CROPS)
def test_plan_center_crop(images, shape, factors) -> None:
    """Center crop plans match crop_geometry, with filters close to the antialiased resize."""
    plan = plan_resize(*shape, CROP, "center_crop")
    _, ratios, padding = crop_geometry(shape[1], shape[0], CROP)
    assert (plan.factors, plan.top, plan.rows) == (factors, 0, CROP[1])
    assert plan.ratios == ratios
    assert plan.padding == padding
    crop = plan.crop
    for first, weights, expected in zip(
        (crop.first_row, crop.first_column),
        (crop.rows, crop.columns),
        _crop_triangles(shape, factors),
    ):
        padded = np.zeros_like(expected)
        padded[:, first : first + weights.shape[1]] = weights
        np.testing.assert_allclose(padded, expected, atol=1e-12)
    frame = _frame(images, *shape)
    antialiased = _apply(frame, *_crop_maps(shape, (1, 1)))
    assert np.abs(_apply(frame, *_crop_maps(shape, factors)) - antialiased).mean() < 0.5


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
@pytest.mark.parametrize(
    ("shape", "size", "method", "build_error", "on_cpu", "builds"),
    [
        pytest.param((1080, 1920), (640, 640), "letterbox", None, False, 1, id="dla"),
        pytest.param((750, 1000), (640, 640), "letterbox", None, True, 0, id="plan-fails"),
        pytest.param((1080, 1920), CROP, "center_crop", None, False, 1, id="dla-crop"),
        pytest.param((750, 1000), CROP, "center_crop", None, True, 0, id="crop-plan-fails"),
        pytest.param((1080, 1920), (640, 640), "pad", None, True, 0, id="pad-plan-fails"),
        pytest.param(
            (1080, 1920), (640, 640), "letterbox", RuntimeError("build"), True, 1, id="build-fails"
        ),
    ],
)
@pytest.mark.parametrize("cpu_resize", [None, True, False], ids=["auto", "cpu_resize", "no"])
def test_resizes_on_cpu_decides_once_from_plan_and_build(
    monkeypatch, shape, size, method, build_error, on_cpu, builds, cpu_resize
) -> None:
    """None resizes on the CPU only when the DLA plan or build fails; True and False never build."""
    built = []

    def build(plan, *_args: object):
        built.append(plan)
        if build_error is not None:
            raise build_error
        return Path("engine")

    monkeypatch.setattr(_dla, "build_dla_preproc", build)
    dla = DLAPreprocessor.__new__(DLAPreprocessor)
    dla._tag = "test"
    dla._o_shape = size
    dla._resize = method
    dla._crop_pct = 0.875
    dla._norm = normalization(RANGE, None, None)
    dla._dla_core = 0
    dla._cpu_resize = cpu_resize
    dla._on_cpu = {}
    expected = on_cpu if cpu_resize is None else cpu_resize
    assert dla.resizes_on_cpu(*shape) is expected
    assert dla.resizes_on_cpu(*shape, method) is expected
    assert len(built) == (builds if cpu_resize is None else 0)


@pytest.mark.cpu
def test_dla_preprocessor_without_dla_raises() -> None:
    """Creating a DLAPreprocessor on a system without a DLA raises RuntimeError."""
    if FLAGS.HAS_DLA:
        pytest.skip("this system has a DLA")
    with pytest.raises(RuntimeError, match="no DLA"):
        DLAPreprocessor((640, 640), RANGE, np.float16)


@pytest.mark.dla
@pytest.mark.parametrize(("shape", "factors"), CROPS)
@pytest.mark.parametrize("dtype", [np.float16, np.float32], ids=["fp16", "fp32"])
def test_dla_center_crop_matches_reference(images, shape, factors, dtype) -> None:
    """Center crops are within a level of the filters, a few levels from CUDA's bilinear crop."""
    frame = _frame(images, *shape)
    dla = DLAPreprocessor(CROP, RANGE, dtype, resize="center_crop", **IMAGENET)
    cuda = CUDAPreprocessor(CROP, RANGE, np.float32, resize="center_crop", **IMAGENET)
    result, ratios, padding = dla.preprocess(frame)
    expected, cuda_ratios, cuda_padding = cuda.preprocess(frame)
    assert result.dtype == dtype
    assert result.shape == expected.shape
    assert ratios == cuda_ratios
    assert padding == cuda_padding
    # the DLA rounds the box filter to whole levels
    boxed = np.round(_box_filter(frame, factors))
    scale, offset = normalization(RANGE, IMAGENET["mean"], IMAGENET["std"])
    reference = _apply(boxed, *_crop_triangles(shape, factors))[..., ::-1].transpose(2, 0, 1)
    reference = reference * scale[:, None, None] + offset[:, None, None]
    level = 1.0 / (255.0 * min(IMAGENET["std"]))
    result = result[0].astype(np.float64)
    diff = np.abs(result - reference)
    assert diff.max() <= level
    assert diff.mean() <= 0.1 * level
    assert np.abs(result - expected[0]).mean() <= 8 * level


@pytest.mark.dla
@pytest.mark.parametrize(
    ("shape", "size", "method"),
    [
        *CONFIGS,
        *(pytest.param(crop.values[0], CROP, "center_crop", id=crop.id) for crop in CROPS),
        pytest.param((1536, 2048), CROP, "center_crop", id="1536p-224-crop-k3"),
        pytest.param((640, 640), (640, 640), "linear", id="cpu_resize-640"),
        pytest.param((384, 640), (640, 384), "linear", id="cpu_resize-640x384"),
        pytest.param((224, 224), (224, 224), "linear", id="cpu_resize-224"),
    ],
)
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
        pytest.param({"resize": "center_crop", "crop_pct": 1.5}, "crop_pct", id="crop-pct"),
        pytest.param({"dtype": np.uint8}, "float16 or float32", id="dtype"),
    ],
)
def test_dla_preprocessor_rejects_config(kwargs, match) -> None:
    """Configurations the DLA path cannot run raise ValueError at construction."""
    args = {"output_shape": (640, 640), "output_range": RANGE, "dtype": np.float16, **kwargs}
    with pytest.raises(ValueError, match=match):
        DLAPreprocessor(**args)


@pytest.mark.dla
def test_detector_dla_matches_cuda(yolov10_engine, images) -> None:
    """A Detector with preprocessor='dla0' finds the same objects as with 'cuda'."""
    frame = _frame(images, 1080, 1920)
    results = []
    for preprocessor in ("cuda", "dla0"):
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


@pytest.mark.dla
def test_classifier_dla_matches_cuda(resnet18_engine, images) -> None:
    """A ResNet-18 with preprocessor='dla0' ranks the ground-truth class first, as with 'cuda'."""
    models = [ResNet(resnet18_engine, preprocessor=name, warmup=False) for name in ("cuda", "dla0")]
    assert isinstance(models[1].get_preprocessor(), DLAPreprocessor)
    for shape in ((1080, 1920), (288, 448)):
        frame = _frame(images, *shape, name="horse")
        (cuda_cls, cuda_score), (dla_cls, dla_score) = (
            model.end2end(frame, top_k=1)[0] for model in models
        )
        assert dla_cls == cuda_cls == images["horse"].gt_cls_id
        assert abs(dla_score - cuda_score) < 0.1


@pytest.mark.dla
def test_detector_dla_cpu_resize_matches_cpu(yolov10_engine, images) -> None:
    """A Detector on dla0 takes a size the DLA cannot resize and finds what the CPU path finds."""
    frame = _frame(images, 750, 1000)
    results = []
    for preprocessor in ("cpu", "dla0"):
        detector = Detector(
            yolov10_engine,
            preprocessor=preprocessor,
            resize_method="letterbox",
            conf_thres=0.5,
            warmup=False,
        )
        results.append(sorted(detector.end2end(frame), key=lambda det: (det[2], det[0])))
    cpu, dla = results
    assert len(dla) == len(cpu) > 0
    for (box, score, cls), (cpu_box, cpu_score, cpu_cls) in zip(dla, cpu):
        assert cls == cpu_cls
        assert abs(score - cpu_score) < 0.02
        np.testing.assert_allclose(box, cpu_box, atol=1)
    assert detector.get_preprocessor().resizes_on_cpu(750, 1000)


def test_dla_preprocessor_needs_core(yolov10_engine) -> None:
    """preprocessor='dla' without a core is an invalid method listing the DLA cores."""
    with pytest.raises(ValueError, match=r"Invalid preprocessing method 'dla'.*'dla0', 'dla1'"):
        Detector(yolov10_engine, preprocessor="dla", warmup=False)
