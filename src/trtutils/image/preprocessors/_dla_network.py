# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
"""Resize, channel swap and normalization networks that run entirely on a DLA."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG
from trtutils.builder._build import build_engine
from trtutils.builder.hooks import int8_input_hook
from trtutils.compat._libs import trt
from trtutils.core import cache as caching_tools

from ._base import DEFAULT_CROP_PCT, crop_geometry

if TYPE_CHECKING:
    from collections.abc import Sequence

# strides a DLA convolution takes: on the image (DLA_HWC4) input, and anywhere else
_INPUT_STRIDE = 4
_MAX_STRIDE = 8
# pixels per row in one CHW16 vector, so the per-plane outputs are planar rows
PHASES = 16
_COLOR_CHANNELS = 3
_INT8_RANGE = 127.0
# outputs per banded 1x1 convolution, and inputs per concatenation, of a center crop
_BAND = 32
_FAN_IN = 8
# the DLA reads image rows in 64-byte steps, and rejects a 4x4 box filter on wider rows
_ROW_PIXELS = 16
_MAX_CROP_WIDTH = 3840
INPUT_NAME = "input"
OUTPUT_NAMES = ("out_r", "out_g", "out_b")
_ADVICE = "Resize it on the CPU (cpu_resize), or use the 'cuda' or 'trt' preprocessor."


@dataclass(frozen=True, eq=False)
class CenterCrop:
    """
    The second stage of a center crop, over the box-filtered image.

    ``rows`` and ``columns`` weigh the box samples from ``first_row`` and
    ``first_column`` on, one row per output row or column.
    """

    first_row: int
    first_column: int
    rows: np.ndarray
    columns: np.ndarray


@dataclass(frozen=True)
class ResizePlan:
    """
    How one input size maps onto the output.

    The engine writes ``rows`` rows starting at row ``top``; the rows above and
    below are the letterbox border. A center crop box filters by ``factors``
    and then applies ``crop``.
    """

    height: int
    width: int
    factors: tuple[int, int]
    top: int
    rows: int
    ratios: tuple[float, float]
    padding: tuple[float, float]
    crop: CenterCrop | None = None

    @property
    def input_rows(self) -> range:
        """The image rows the engine reads."""
        if self.crop is None:
            return range(self.height)
        factor = self.factors[0]
        first = self.crop.first_row * factor
        return range(first, first + self.crop.rows.shape[1] * factor)


def _crop_axis(size: int, resized: int, start: int, count: int) -> tuple[int, int, np.ndarray]:
    """
    Split a center crop axis into a box filter and a triangle filter over its samples.

    The box factor is half the downscale (at most 4, the DLA_HWC4 input stride);
    the triangle, as wide as the rest of the downscale and centred on the
    antialiased resize's sample positions, covers the rest.

    Returns the box factor, the first box sample read, and the triangle weights.
    """
    scale = size / resized
    factor = min(max(int(scale / 2), 1), _INPUT_STRIDE)
    support = max(scale / factor, 1.0)
    centres = (np.arange(start, start + count) + 0.5) * scale / factor - 0.5
    distance = np.abs(np.arange(size // factor) - centres[:, None]) / support
    weights = np.maximum(1.0 - distance, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    used = np.flatnonzero(weights.any(axis=0))
    return factor, int(used[0]), weights[:, used[0] : used[-1] + 1]


def plan_resize(
    height: int,
    width: int,
    output_shape: tuple[int, int],
    method: str,
    crop_pct: float = DEFAULT_CROP_PCT,
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
        The resize method, 'letterbox', 'linear' or 'center_crop'.
    crop_pct : float
        The fraction of the resized image a center crop keeps. Default is 0.875.

    Returns
    -------
    ResizePlan
        The resize factors and the rows the engine writes.

    Raises
    ------
    ValueError
        If the DLA cannot do the resize: an unknown method, a letterbox or linear
        factor that is not a whole number, letterbox padding on the left and
        right, a factor with no DLA stride split (a prime above 4, or above 32),
        or a center crop of an image wider than 3840 or whose width is not a
        multiple of 16.

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
                f"{width}x{height} -> {out_width}x{out_height} is not one. {_ADVICE}"
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
                f"{out_width}x{out_height} is not. {_ADVICE}"
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
    elif method == "center_crop":
        if width % _ROW_PIXELS or width > _MAX_CROP_WIDTH:
            err_msg = (
                f"The DLA preprocessor center crops images up to {_MAX_CROP_WIDTH} pixels wide "
                f"whose width is a multiple of {_ROW_PIXELS}: {width}x{height} is not one. {_ADVICE}"
            )
            raise ValueError(err_msg)
        (resized_width, resized_height, left, top), ratios, padding = crop_geometry(
            width, height, output_shape, crop_pct
        )
        row_factor, first_row, rows = _crop_axis(height, resized_height, top, out_height)
        col_factor, first_column, columns = _crop_axis(width, resized_width, left, out_width)
        plan = ResizePlan(
            height,
            width,
            (row_factor, col_factor),
            0,
            out_height,
            ratios,
            padding,
            CenterCrop(first_row, first_column, rows, columns),
        )
    else:
        err_msg = (
            f"The DLA preprocessor supports letterbox, linear and center_crop resizing, not {method}"
        )
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

    def concat(self, tensors: list[trt.ITensor]) -> trt.ITensor:
        """Concatenate on channels as a tree, since TensorRT runs wide concatenations on the GPU."""
        while len(tensors) > 1:
            groups = [tensors[start : start + _FAN_IN] for start in range(0, len(tensors), _FAN_IN)]
            tensors = [
                group[0] if len(group) == 1 else self.network.add_concatenation(group).get_output(0)
                for group in groups
            ]
        return tensors[0]

    def transpose(self, tensor: trt.ITensor, order: tuple[int, ...]) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.first_transpose = order
        return layer.get_output(0)

    def banded(self, tensor: trt.ITensor, matrix: np.ndarray) -> trt.ITensor:
        """Weigh the channels of ``tensor`` by ``matrix``, as 1x1 convolutions over output bands."""
        shape = tuple(tensor.shape)
        groups: list[trt.ITensor] = []
        bands: list[trt.ITensor] = []
        for first in range(0, len(matrix), _BAND):
            band = matrix[first : first + _BAND]
            used = np.flatnonzero(band.any(axis=0))
            start, stop = int(used[0]), int(used[-1]) + 1
            inputs = self.network.add_slice(
                tensor, (0, start, 0, 0), (1, stop - start, *shape[2:]), (1, 1, 1, 1)
            ).get_output(0)
            if tensor.dynamic_range is not None:
                _int8(inputs)
            bands.append(self.conv(inputs, band[:, start:stop, None, None], (1, 1)))
            # tensorrt 8.6 splits the DLA graph when many tensors wait for a concatenation
            if len(bands) == _FAN_IN or first + _BAND >= len(matrix):
                groups.append(self.concat(bands))
                bands = []
        return self.concat(groups)


def _int8(tensor: trt.ITensor) -> trt.ITensor:
    tensor.set_dynamic_range(-_INT8_RANGE, _INT8_RANGE)
    return tensor


def _center_crop(writer: _NetworkWriter, tensor: trt.ITensor, crop: CenterCrop) -> trt.ITensor:
    """Apply the triangle filters to the box-filtered int8 image, giving float16 RGB planes."""
    _, channels, rows, _ = tuple(tensor.shape)
    window = (1, channels, rows, crop.columns.shape[1])
    start = (0, 0, 0, crop.first_column)
    tensor = _int8(writer.network.add_slice(tensor, start, window, (1, 1, 1, 1)).get_output(0))
    # columns on channels, then rows on channels
    tensor = writer.banded(_int8(writer.transpose(tensor, (0, 3, 1, 2))), crop.columns)
    tensor = writer.banded(writer.transpose(tensor, (0, 3, 2, 1)), crop.rows)
    return writer.transpose(tensor, (0, 2, 1, 3))


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
    Define the network of a plan, with float32 I/O that the build retypes.

    The input is the BGR image less 128, which :func:`build_dla_preproc` makes the
    int8 BGRx image (``image ^ 0x80``) in DLA_HWC4. Strided convolutions at int8
    scale 1 resize it and swap BGR to RGB, one-hot convolutions move each run of
    16 output columns onto channels, and the last one normalizes into one output
    per RGB plane, of shape (1, 16, rows, width / 16), whose float16 CHW16 bytes
    are the plane's rows. A center crop box filters with the strided convolutions,
    then applies its triangle filters as 1x1 convolutions over the transposed
    image, columns then rows, in float16.

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
    crop = plan.crop
    shape = (1, _COLOR_CHANNELS, len(plan.input_rows), plan.width)
    tensor = network.add_input(INPUT_NAME, trt.float32, shape)

    if crop is None:
        rows, cols = (resize_stages(factor) for factor in plan.factors)
    else:
        rows, cols = ([(factor, np.full(factor, 1.0 / factor))] for factor in plan.factors)
    count = max(len(rows), len(cols))
    identity = (1, np.ones(1, dtype=np.float32))
    rows += [identity] * (count - len(rows))
    cols += [identity] * (count - len(cols))
    phases = 1 if crop is not None else _phase_count(count, cols[-1][0])
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

    if crop is not None:
        tensor = _center_crop(writer, tensor, crop)
    while phases * _MAX_STRIDE < PHASES:
        tensor = writer.conv(tensor, _phase_weights(phases, _MAX_STRIDE), (1, _MAX_STRIDE))
        if crop is None:
            tensor = _int8(tensor)
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
    return writer.weights


def build_dla_preproc(
    plan: ResizePlan,
    scale: np.ndarray,
    offset: np.ndarray,
    dla_core: int,
) -> Path:
    """
    Build the DLA engine of a plan, or get it from the engine cache.

    ``builder.hooks.int8_input_hook`` makes the input int8 at scale 1, and the
    outputs are float16. The build has no GPU fallback and no I/O reformats, so
    TensorRT fails rather than put a layer on the GPU. Engines are cached by input
    size, output size, normalization, center crop filters and TensorRT version. An
    engine built for one DLA core runs on any.

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
    values = [scale, offset]
    kind = ""
    if plan.crop is not None:
        crop = plan.crop
        out_width = len(crop.columns)
        values += [crop.rows.ravel(), crop.columns.ravel()]
        kind = f"_crop{crop.first_column}x{crop.first_row}_k{plan.factors[1]}x{plan.factors[0]}"
    digest = hashlib.sha256(np.concatenate(values).tobytes())
    name = (
        f"dla_preproc_{plan.width}x{plan.height}_{out_width}x{plan.rows}{kind}"
        f"_{digest.hexdigest()[:12]}_{trt.__version__}"
    )
    exists, path = caching_tools.query(name)
    if exists:
        return path

    def define(network: trt.INetworkDefinition, config: trt.IBuilderConfig) -> list[np.ndarray]:
        # else tensorrt 8.6 sometimes picks a GPU reformat of a float16 output and fails
        config.set_flag(trt.BuilderFlag.DIRECT_IO)
        return define_network(network, plan, scale, offset)

    # tensorrt warns about every float16 tensor of an int8 build
    with tempfile.TemporaryDirectory() as tmpdir, LOG.suppress(), LOG.capture_errors() as errors:
        try:
            build_engine(
                define,
                Path(tmpdir) / f"{name}.engine",
                default_device=trt.DeviceType.DLA,
                workspace=1.0,
                dla_core=dla_core,
                input_tensor_formats=[(INPUT_NAME, trt.DataType.INT8, trt.TensorFormat.DLA_HWC4)],
                output_tensor_formats=[
                    (output, trt.DataType.HALF, trt.TensorFormat.CHW16) for output in OUTPUT_NAMES
                ],
                hooks=[int8_input_hook],
                profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
                fp16=True,
                int8=True,
                cache=True,
            )
        except RuntimeError:
            err_msg = (
                f"TensorRT could not build the DLA preprocessing engine {name}: {errors}. {_ADVICE}"
            )
            raise RuntimeError(err_msg) from None
    return path
