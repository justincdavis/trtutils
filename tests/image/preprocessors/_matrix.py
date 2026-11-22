# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Axes, images and ids of the preprocessor test matrix, with the marks and skips of each case."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from trtutils.image.preprocessors._dla_network import ResizePlan, plan_resize

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from _pytest.mark.structures import ParameterSet

BACKENDS = ("cpu", "cuda", "trt", "dla0", "dla1")
RESIZES = ("letterbox", "linear", "center_crop", "pad")
FLOATS = ("fp32", "fp16")
DTYPES = {
    "fp32": np.dtype(np.float32),
    "fp16": np.dtype(np.float16),
    "uint8": np.dtype(np.uint8),
    "int8": np.dtype(np.int8),
    "float64": np.dtype(np.float64),
}
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
NORMS = {
    "range": ((0.0, 1.0), None, None),
    "imagenet": ((0.0, 1.0), *IMAGENET),
}
KINDS = ("numpy", "host", "mapped", "device")
BUFFER_KINDS = ("host", "pinned", "mapped", "device")
CPU_KINDS = ("numpy", "host")
CPU_RESIZE = (None, True, False)
SIZES = {
    "letterbox": (640, 640),
    "linear": (640, 360),
    "center_crop": (224, 224),
    "pad": (640, 640),
}
BATCH_SIZE = 3

FRAMES = {
    "people-1920x1080": ("people.jpeg", 1080, 1920),
    "horse-1920x1080": ("horse.jpg", 1080, 1920),
    "people-1280x720": ("people.jpeg", 720, 1280),
    "people-1000x750": ("people.jpeg", 750, 1000),
    "horse-447x288": ("horse.jpg", 288, 447),
}
BATCHES = {
    "single": ("people-1920x1080",),
    "batch2": ("people-1920x1080", "horse-1920x1080"),
    "mixed": ("people-1280x720", "people-1920x1080"),
    "unaligned": ("people-1000x750", "horse-447x288"),
}
BAD_IMAGES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "float32": lambda image: image.astype(np.float32),
    "gray": lambda image: np.ascontiguousarray(image[..., 0]),
    "bgra": lambda image: np.concatenate([image, image[..., :1]], axis=2),
}
ID_ORDER = (
    "backend",
    "geometry",
    "resize",
    "size",
    "dtype",
    "norm",
    "kind",
    "batch",
    "bad",
    "position",
    "cpu_resize",
)


@dataclass(frozen=True)
class Geometry:
    """One photo at one input size, resized to one output size."""

    name: str
    photo: str
    height: int
    width: int
    size: tuple[int, int]
    resize: str


GEOMETRIES = (
    Geometry("1280x720-320-letterbox-k4", "people.jpeg", 720, 1280, (320, 320), "letterbox"),
    Geometry("1920x1080-320-letterbox-k6", "people.jpeg", 1080, 1920, (320, 320), "letterbox"),
    Geometry("1920x1080-640x540-linear-k3x2", "people.jpeg", 1080, 1920, (640, 540), "linear"),
    Geometry("3840x2160-224-center_crop-k4", "people.jpeg", 2160, 3840, (224, 224), "center_crop"),
    Geometry("2048x1536-224-center_crop-k3", "people.jpeg", 1536, 2048, (224, 224), "center_crop"),
    Geometry("640x480-224-center_crop-k1", "people.jpeg", 480, 640, (224, 224), "center_crop"),
    Geometry("720x480-224-center_crop-k1", "people.jpeg", 480, 720, (224, 224), "center_crop"),
    Geometry("1000x750-640x384-letterbox", "people.jpeg", 750, 1000, (640, 384), "letterbox"),
    Geometry("1920x1080-640-linear", "people.jpeg", 1080, 1920, (640, 640), "linear"),
    Geometry("1920x1280-1024-letterbox", "horse.jpg", 1280, 1920, (1024, 1024), "letterbox"),
    Geometry("4000x3000-640-center_crop", "horse.jpg", 3000, 4000, (640, 640), "center_crop"),
    Geometry("4000x3000-384x640-letterbox", "horse.jpg", 3000, 4000, (384, 640), "letterbox"),
    Geometry("3001x1999-640x384-linear", "horse.jpg", 1999, 3001, (640, 384), "linear"),
    Geometry("3001x1999-1024-center_crop", "horse.jpg", 1999, 3001, (1024, 1024), "center_crop"),
    Geometry("1080x1920-640x384-center_crop", "horse.jpg", 1920, 1080, (640, 384), "center_crop"),
    Geometry("1080x1920-384x640-linear", "horse.jpg", 1920, 1080, (384, 640), "linear"),
    Geometry("1000x750-640x384-pad", "people.jpeg", 750, 1000, (640, 384), "pad"),
    Geometry("1920x1080-1024-pad", "horse.jpg", 1080, 1920, (1024, 1024), "pad"),
    Geometry("1080x1920-1024-pad", "horse.jpg", 1920, 1080, (1024, 1024), "pad"),
    Geometry("447x288-1024-pad", "horse.jpg", 288, 447, (1024, 1024), "pad"),
    Geometry("640x360-640-pad", "people.jpeg", 360, 640, (640, 640), "pad"),
)


@dataclass(frozen=True)
class Case:
    """One combination of the matrix axes; fields a test does not parametrize keep their default."""

    backend: str
    resize: str = "letterbox"
    size: tuple[int, int] | None = None
    dtype: str = "fp32"
    norm: str = "range"
    kind: str = "numpy"
    batch: str = "single"
    cpu_resize: bool | None = None
    geometry: Geometry | None = None
    bad: str | None = None
    position: str | None = None

    @property
    def output_shape(self) -> tuple[int, int]:
        """The (width, height) the preprocessor writes."""
        return self.size if self.size is not None else SIZES[self.resize]

    @property
    def dla(self) -> bool:
        """Whether the backend is a DLA core."""
        return self.backend.startswith("dla")


def dla_plan(case: Case, height: int, width: int) -> ResizePlan | None:
    """The DLA resize of an input size, or None when the DLA cannot do it."""
    try:
        return plan_resize(height, width, case.output_shape, case.resize)
    except ValueError:
        return None


def _label(name: str, value: object) -> str:
    if name == "cpu_resize":
        return f"cpu_resize={value}"
    if isinstance(value, tuple):
        return "x".join(str(side) for side in value)
    if isinstance(value, Geometry):
        return value.name
    return str(value)


def _hardware(case: Case) -> list[pytest.MarkDecorator]:
    if case.dla:
        return [pytest.mark.dla]
    if case.backend == "cpu" and case.kind in CPU_KINDS:
        return [pytest.mark.cpu]
    return []


def impossible(case: Case) -> str | None:
    """Why a combination cannot run, or None."""
    if case.backend == "cpu" and case.cpu_resize is not None:
        return "CPUPreprocessor has no cpu_resize: it always resizes with cv2"
    if case.backend in ("cuda", "trt") and case.cpu_resize is False:
        return "cuda and trt store bool(cpu_resize), so cpu_resize=False is cpu_resize=None"
    if case.dla and case.batch == "unaligned" and case.cpu_resize is False:
        return "the DLA cannot resize 1000x750 or 447x288 and cpu_resize=False forbids the CPU"
    if case.dla and case.batch == "unaligned" and case.kind == "device":
        return "device Buffers are always resized on the DLA, which cannot resize 1000x750"
    if case.dla and case.resize == "pad" and (case.cpu_resize is False or case.kind == "device"):
        return "the DLA cannot pad resize, only the CPU does it for host images"
    return None


def matrix(
    *,
    skip: Callable[[Case], str | None] | None = None,
    xfail: Callable[[Case], str | None] | None = None,
    **axes: Sequence[Any],
) -> list[ParameterSet]:
    """Every backend crossed with the given axes, with readable ids, hardware marks and skips."""
    names = ["backend", *axes]
    params = []
    for values in itertools.product(BACKENDS, *axes.values()):
        fields: dict[str, Any] = dict(zip(names, values))
        case = Case(**fields)
        if case.geometry is not None:
            case = replace(case, resize=case.geometry.resize, size=case.geometry.size)
        marks = _hardware(case)
        reason = impossible(case) or (skip(case) if skip is not None else None)
        expected_failure = xfail(case) if xfail is not None else None
        if reason is not None:
            marks.append(pytest.mark.skip(reason=reason))
        elif expected_failure is not None:
            marks.append(pytest.mark.xfail(reason=expected_failure, strict=True))
        case_id = "-".join(_label(name, fields[name]) for name in ID_ORDER if name in fields)
        params.append(pytest.param(case, id=case_id, marks=marks))
    return params
