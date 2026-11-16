# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
"""Resize, channel swap and normalization networks that run entirely on a DLA."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.compat._libs import trt
from trtutils.core import cache as caching_tools

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# strides a DLA convolution takes: on the image (DLA_HWC4) input, and anywhere else
_INPUT_STRIDE = 4
_MAX_STRIDE = 8
# pixels per row in one CHW16 vector, so the per-plane outputs are planar rows
PHASES = 16
_COLOR_CHANNELS = 3
_INT8_RANGE = 127.0
INPUT_NAME = "input"
OUTPUT_NAMES = ("out_r", "out_g", "out_b")


@dataclass(frozen=True)
class ResizePlan:
    """
    How one input size maps onto the output.

    The engine writes ``rows`` rows starting at row ``top``; the rows above and
    below are the letterbox border.
    """

    height: int
    width: int
    factors: tuple[int, int]
    top: int
    rows: int
    ratios: tuple[float, float]
    padding: tuple[float, float]


def plan_resize(
    height: int,
    width: int,
    output_shape: tuple[int, int],
    method: str,
) -> ResizePlan:
    """
    Get the DLA resize of an input size, with the ratios and padding of the GPU preprocessors.

    Parameters
    ----------
    height : int
        The input height.
    width : int
        The input width.
    output_shape : tuple[int, int]
        The output (width, height).
    method : str
        The resize method, 'letterbox' or 'linear'.

    Returns
    -------
    ResizePlan
        The resize factors and the rows the engine writes.

    Raises
    ------
    ValueError
        If the DLA cannot do the resize: a method other than letterbox or linear,
        a factor that is not a whole number, letterbox padding on the left and
        right, or a factor with no DLA stride split (a prime above 4, or above 32).

    """
    out_width, out_height = output_shape
    if method == "letterbox":
        scale = min(out_width / width, out_height / height)
        new_width, new_height = round(width * scale), round(height * scale)
        pad_width, pad_height = (out_width - new_width) / 2, (out_height - new_height) / 2
        factor = width // out_width
        if (
            new_width != out_width
            or width % out_width
            or height % factor
            or height // factor != new_height
        ):
            err_msg = (
                f"The DLA preprocessor letterboxes by a whole factor across the full width: "
                f"{width}x{height} -> {out_width}x{out_height} is not one. "
                "Use the 'cuda' or 'trt' preprocessor for this input size."
            )
            raise ValueError(err_msg)
        plan = ResizePlan(
            height,
            width,
            (factor, factor),
            round(pad_height - 0.1),
            height // factor,
            (scale, scale),
            (pad_width, pad_height),
        )
    elif method == "linear":
        if width % out_width or height % out_height:
            err_msg = (
                f"The DLA preprocessor resizes by whole factors: {width}x{height} -> "
                f"{out_width}x{out_height} is not. Use the 'cuda' or 'trt' preprocessor "
                "for this input size."
            )
            raise ValueError(err_msg)
        plan = ResizePlan(
            height,
            width,
            (height // out_height, width // out_width),
            0,
            out_height,
            (out_width / width, out_height / height),
            (0.0, 0.0),
        )
    else:
        err_msg = f"The DLA preprocessor supports letterbox and linear resizing, not {method}"
        raise ValueError(err_msg)
    for factor in plan.factors:
        resize_stages(factor)
    return plan


def _point_taps(factor: int) -> np.ndarray:
    """Bilinear sampling at exactly 1 / factor: the centre pixel, or the mean of the two."""
    taps = np.zeros(factor, dtype=np.float32)
    taps[(factor - 1) // 2] = 1.0
    if factor % 2 == 0:
        taps[factor // 2 - 1 : factor // 2 + 1] = 0.5
    return taps


def resize_stages(factor: int) -> list[tuple[int, np.ndarray]]:
    """
    Split a 1-D resize by ``factor`` into (stride, taps) stages the DLA can run.

    Parameters
    ----------
    factor : int
        The resize factor.

    Returns
    -------
    list[tuple[int, np.ndarray]]
        One stage, or two whose strides multiply to ``factor``.

    Raises
    ------
    ValueError
        If no split has a first stride of at most 4 and a second of at most 8.

    """
    if factor <= _INPUT_STRIDE:
        return [(factor, _point_taps(factor))]
    centre = (factor - 1) // 2
    for first in range(_INPUT_STRIDE, 1, -1):
        second, rest = divmod(factor, first)
        if rest or second > _MAX_STRIDE:
            continue
        block, offset = divmod(centre, first)
        pair = factor % 2 == 0
        taps = np.zeros(first + int(pair and offset == first - 1), dtype=np.float32)
        taps[offset : offset + 1 + int(pair)] = 0.5 if pair else 1.0
        pick = np.zeros(second, dtype=np.float32)
        pick[block] = 1.0
        return [(first, taps), (second, pick)]
    err_msg = (
        f"The DLA preprocessor cannot resize by a factor of {factor}: "
        "it needs a factor of at most 4, or a product of one up to 4 and one up to 8."
    )
    raise ValueError(err_msg)


def normalization(
    output_range: tuple[float, float],
    mean: Sequence[float] | np.ndarray | None,
    std: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the per-channel scale and offset that map a 0-255 pixel to the model's input.

    Parameters
    ----------
    output_range : tuple[float, float]
        The output range, used without mean and std.
    mean : Sequence[float] | np.ndarray, optional
        The ImageNet mean of each RGB channel.
    std : Sequence[float] | np.ndarray, optional
        The ImageNet standard deviation of each RGB channel.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The float64 scale and offset of each RGB channel.

    """
    if mean is not None and std is not None:
        std_arr = np.asarray(std, dtype=np.float64).reshape(-1)
        return 1.0 / (255.0 * std_arr), -np.asarray(mean, dtype=np.float64).reshape(-1) / std_arr
    low, high = output_range
    return np.full(_COLOR_CHANNELS, (high - low) / 255.0), np.full(_COLOR_CHANNELS, low)


class _NetworkWriter:
    """Adds convolutions to a network, keeping their weights alive until the build."""

    def __init__(self, network: trt.INetworkDefinition) -> None:
        self.network = network
        self.weights: list[np.ndarray] = []

    def conv(
        self,
        tensor: trt.ITensor,
        weights: np.ndarray,
        stride: tuple[int, int],
        post_padding: tuple[int, int] = (0, 0),
        bias: np.ndarray | None = None,
    ) -> trt.ITensor:
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        bias = np.zeros(len(weights), dtype=np.float32) if bias is None else bias.astype(np.float32)
        self.weights.extend((weights, bias))
        layer = self.network.add_convolution_nd(
            tensor, weights.shape[0], weights.shape[2:], trt.Weights(weights), trt.Weights(bias)
        )
        layer.stride_nd = stride
        layer.post_padding = post_padding
        return layer.get_output(0)


def _int8(tensor: trt.ITensor) -> trt.ITensor:
    tensor.set_dynamic_range(-_INT8_RANGE, _INT8_RANGE)
    return tensor


def _phase_count(stages: int, last_stride: int) -> int:
    """Output columns the last resize convolution moves onto channels."""
    limit = _INPUT_STRIDE if stages == 1 else _MAX_STRIDE
    phases = 1
    while last_stride * phases * 2 <= limit:
        phases *= 2
    return phases


def _phase_weights(phases: int, step: int) -> np.ndarray:
    """One-hot weights taking ``step`` columns of ``phases`` phases each onto channels."""
    out = np.arange(_COLOR_CHANNELS * phases * step)
    color, rest = np.divmod(out, phases * step)
    column, phase = np.divmod(rest, phases)
    weights = np.zeros((len(out), _COLOR_CHANNELS * phases, 1, step))
    weights[out, color * phases + phase, 0, column] = 1.0
    return weights


def define_network(
    network: trt.INetworkDefinition,
    plan: ResizePlan,
    scale: np.ndarray,
    offset: np.ndarray,
) -> list[np.ndarray]:
    """
    Define the DLA network of a plan.

    The input is the int8 BGRx image (``image ^ 0x80``) in DLA_HWC4. Strided
    convolutions at int8 scale 1 resize it and swap BGR to RGB, one-hot
    convolutions move each run of 16 output columns onto channels, and the last
    one normalizes into one float16 CHW16 output per RGB plane, of shape
    (1, 16, rows, width / 16), whose bytes are the plane's rows.

    Parameters
    ----------
    network : trt.INetworkDefinition
        The empty network to define.
    plan : ResizePlan
        The resize.
    scale : np.ndarray
        The per-channel scale from :func:`normalization`.
    offset : np.ndarray
        The per-channel offset from :func:`normalization`.

    Returns
    -------
    list[np.ndarray]
        The weights the network references, to keep alive until it is built.

    """
    writer = _NetworkWriter(network)
    image = network.add_input(INPUT_NAME, trt.int8, (1, _COLOR_CHANNELS, plan.height, plan.width))
    image.allowed_formats = 1 << int(trt.TensorFormat.DLA_HWC4)
    tensor = _int8(image)

    rows, cols = (resize_stages(factor) for factor in plan.factors)
    count = max(len(rows), len(cols))
    identity = (1, np.ones(1, dtype=np.float32))
    rows += [identity] * (count - len(rows))
    cols += [identity] * (count - len(cols))
    phases = _phase_count(count, cols[-1][0])
    channels = _COLOR_CHANNELS
    for index, ((row_stride, row_taps), (col_stride, col_taps)) in enumerate(zip(rows, cols)):
        group = phases if index == count - 1 else 1
        width = col_stride * (group - 1) + len(col_taps)
        weights = np.zeros((_COLOR_CHANNELS * group, channels, len(row_taps), width))
        for color in range(_COLOR_CHANNELS):
            source = _COLOR_CHANNELS - 1 - color if index == 0 else color
            for phase in range(group):
                start = phase * col_stride
                weights[color * group + phase, source, :, start : start + len(col_taps)] = np.outer(
                    row_taps, col_taps
                )
        tensor = _int8(
            writer.conv(
                tensor,
                weights,
                (row_stride, col_stride * group),
                (len(row_taps) - row_stride, len(col_taps) - col_stride),
            )
        )
        channels = len(weights)

    while phases * _MAX_STRIDE < PHASES:
        tensor = _int8(writer.conv(tensor, _phase_weights(phases, _MAX_STRIDE), (1, _MAX_STRIDE)))
        phases *= _MAX_STRIDE
    step = PHASES // phases
    weights = _phase_weights(phases, step)
    for color, name in enumerate(OUTPUT_NAMES):
        output = writer.conv(
            tensor,
            weights[color * PHASES : (color + 1) * PHASES] * scale[color],
            (1, step),
            bias=np.full(PHASES, 128.0 * scale[color] + offset[color]),
        )
        output.name = name
        network.mark_output(output)
        output.dtype = trt.float16
        output.allowed_formats = 1 << int(trt.TensorFormat.CHW16)
    return writer.weights


def build_dla_preproc(
    plan: ResizePlan,
    scale: np.ndarray,
    offset: np.ndarray,
    dla_core: int,
) -> Path:
    """
    Build the DLA engine of a plan, or get it from the engine cache.

    The build has no GPU fallback, so TensorRT fails rather than put a layer on the
    GPU. Engines are cached by input size, output size, normalization and TensorRT
    version. An engine built for one DLA core runs on any.

    Parameters
    ----------
    plan : ResizePlan
        The resize.
    scale : np.ndarray
        The per-channel scale from :func:`normalization`.
    offset : np.ndarray
        The per-channel offset from :func:`normalization`.
    dla_core : int
        The DLA core to build for.

    Returns
    -------
    Path
        The path of the cached engine.

    Raises
    ------
    RuntimeError
        If TensorRT cannot build the network for the DLA, e.g. for some inputs wider
        than 4K UHD (3840 pixels).

    """
    out_width = plan.width // plan.factors[1]
    digest = hashlib.sha256(np.concatenate([scale, offset]).tobytes())
    name = (
        f"dla_preproc_{plan.width}x{plan.height}_{out_width}x{plan.rows}"
        f"_{digest.hexdigest()[:12]}_{trt.__version__}"
    )
    exists, path = caching_tools.query(name)
    if exists:
        return path

    builder = trt.Builder(LOG)
    flags = 0
    if FLAGS.EXPLICIT_BATCH_FLAG:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    weights = define_network(network, plan, scale, offset)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = dla_core
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # tensorrt warns about every float16 tensor of an int8 build
    with LOG.suppress(), LOG.capture_errors() as errors:
        serialized = builder.build_serialized_network(network, config)
    del weights
    if serialized is None:
        err_msg = (
            f"TensorRT could not build the DLA preprocessing engine {name}: {errors}. "
            "Use the 'cuda' or 'trt' preprocessor for this input size."
        )
        raise RuntimeError(err_msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    caching_tools.write_atomic(path, memoryview(serialized))
    return path
