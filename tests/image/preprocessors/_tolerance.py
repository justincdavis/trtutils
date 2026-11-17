# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""How closely each backend matches CPUPreprocessor, in levels: one uint8 step after normalizing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tests.image.preprocessors._matrix import NORMS, Case, dla_plan
from trtutils.image.preprocessors._cpu import resize_image

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class Tolerance:
    """The largest and the mean difference allowed, in levels."""

    max: float
    mean: float = math.inf


TOLERANCES = {
    "float64": Tolerance(0.001),
    "cpu-resize": Tolerance(0.001),
    "gpu-resize": Tolerance(1.0, 0.25),
    "dla-normalize": Tolerance(0.3),
    "dla-even": Tolerance(2.0, 0.35),
    "dla-crop": Tolerance(math.inf, 8.0),
}
TRT_FP16 = Tolerance(0.5, 0.25)


def path(case: Case, height: int, width: int) -> str:
    """Which resize and normalization produce an image's output, as a key of TOLERANCES."""
    on_device = case.kind == "device"
    if case.backend == "cpu":
        return "float64"
    if case.backend in ("cuda", "trt"):
        return "cpu-resize" if case.cpu_resize and not on_device else "gpu-resize"
    plan = dla_plan(case, height, width)
    if not on_device and (case.cpu_resize or plan is None):
        return "dla-normalize"
    if case.resize == "center_crop":
        return "dla-crop"
    if plan is not None and any(factor % 2 == 0 for factor in plan.factors):
        return "dla-even"
    return "dla-normalize"


def levels(norm: str) -> np.ndarray:
    """The size of one level in each output channel."""
    (low, high), _, std = NORMS[norm]
    if std is None:
        return np.full((3, 1, 1), (high - low) / 255.0)
    return 1.0 / (255.0 * np.array(std).reshape(3, 1, 1))


def reference(
    images: Sequence[np.ndarray], case: Case
) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    """The float64 pipeline CPUPreprocessor computes in a single float32 pass."""
    (low, high), mean, std = NORMS[case.norm]
    tensors, ratios, padding = [], [], []
    for image in images:
        resized, ratio, pad = resize_image(image, case.output_shape, case.resize)
        rgb = resized[..., ::-1].transpose(2, 0, 1).astype(np.float64)
        if mean is None or std is None:
            tensors.append(rgb / 255.0 * (high - low) + low)
        else:
            shift = np.array(mean).reshape(3, 1, 1)
            tensors.append((rgb / 255.0 - shift) / np.array(std).reshape(3, 1, 1))
        ratios.append(ratio)
        padding.append(pad)
    return np.stack(tensors), ratios, padding


def assert_close(
    result: np.ndarray,
    expected: np.ndarray,
    case: Case,
    images: Sequence[np.ndarray],
) -> None:
    """Each image of a batch is within the tolerance of the path that produced it."""
    level = levels(case.norm)
    rounded = case.dtype == "fp16" or case.dla
    for i, image in enumerate(images):
        key = path(case, *image.shape[:2])
        tolerance = TOLERANCES[key]
        extra = TRT_FP16 if case.backend == "trt" and case.dtype == "fp16" else Tolerance(0.0, 0.0)
        reference_values = expected[i].astype(np.float64)
        diff = np.abs(result[i].astype(np.float64) - reference_values)
        allowed = (tolerance.max + extra.max) * level
        if rounded:
            allowed = allowed + np.spacing(np.abs(reference_values).astype(np.float16))
        worst = float((diff / level).max())
        assert (diff <= allowed).all(), f"image {i} ({key}): {worst:.4f} levels off"
        mean = float((diff / level).mean())
        assert mean <= tolerance.mean + extra.mean, f"image {i} ({key}): mean {mean:.4f} levels"
