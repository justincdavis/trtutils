# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils.compat._libs import trt

from ._common import check_dynamic_range

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from typing_extensions import TypeAlias

    _Interval: TypeAlias = "tuple[float, float]"

_INT8_MAX = 127.0
_INT8_MIN = -128.0
_SLICE_FILL_INPUT = 4


def int8_ranges_hook(
    input_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> Callable[[trt.INetworkDefinition], trt.INetworkDefinition]:
    """
    Create a hook that sets INT8 dynamic ranges without calibration data.

    Interval arithmetic carries the (min, max) of each network input through the
    network, and every tensor whose interval is bounded gets the tightest dynamic range
    that covers it. The bounds are exact for convex combinations (resize, average
    pooling, blur kernels) and affine maps, so deterministic pipelines such as image
    preprocessing build in INT8 with no calibration data, on the GPU and on the DLA.
    Through learned layers the bounds grow with depth (each layer can scale them by the
    sum of its absolute weights), so calibrate those networks with a ``data_batcher``.

    Handled layers: convolution, deconvolution, scale, activation (ReLU, clip, sigmoid,
    tanh, leaky ReLU), unary abs and neg, pooling, resize (not cubic), elementwise
    (sum, sub, prod, max, min, div), concatenation, shuffle, slice, padding, cast,
    identity, softmax and constants. Tensors after any other layer get no range and run
    in higher precision; ``build_engine`` warns with their count. Tensors that already
    have a dynamic range keep it.

    Parameters
    ----------
    input_ranges : Mapping[str, tuple[float, float]], optional
        The (min, max) of each float network input, by name. An int8 input with a
        dynamic range, such as one from ``int8_input_hook``, needs no entry: its
        interval is the int8 values at its scale. By default None.

    Returns
    -------
    Callable[[trt.INetworkDefinition], trt.INetworkDefinition]
        The hook, for ``build_engine(..., int8=True, hooks=[...])`` without a
        ``data_batcher`` or ``calibration_cache``. It raises ``ValueError`` if a float
        or int8 network input has no range.

    Raises
    ------
    RuntimeError
        If the installed TensorRT has no per-tensor dynamic ranges (TensorRT 11).

    Examples
    --------
    A uint8 image fed as int8 at scale 1, with a float16 output, in INT8 on the DLA:

    >>> build_engine(
    ...     "preprocess.onnx",
    ...     "preprocess.engine",
    ...     default_device="dla",
    ...     gpu_fallback=True,
    ...     int8=True,
    ...     hooks=[int8_input_hook, fp16_io_hook, int8_ranges_hook()],
    ... )

    """
    check_dynamic_range()
    ranges = dict(input_ranges) if input_ranges is not None else {}

    def hook(network: trt.INetworkDefinition) -> trt.INetworkDefinition:
        inputs = [network.get_input(idx) for idx in range(network.num_inputs)]
        intervals = _propagate(network, _input_intervals(inputs, ranges))
        tensors = list(inputs)
        for idx in range(network.num_layers):
            layer = network.get_layer(idx)
            tensors.extend(layer.get_output(out) for out in range(layer.num_outputs))
        for tensor in tensors:
            interval = intervals.get(tensor.name)
            if interval is None or tensor.dynamic_range is not None:
                continue
            bound = _symmetric_bound(interval)
            if bound > 0:
                tensor.set_dynamic_range(-bound, bound)
        return network

    return hook


def _input_intervals(
    inputs: list[trt.ITensor],
    ranges: Mapping[str, tuple[float, float]],
) -> dict[str, _Interval]:
    known: dict[str, _Interval] = {}
    missing = []
    for tensor in inputs:
        if tensor.name in ranges:
            lo, hi = ranges[tensor.name]
            known[tensor.name] = (float(lo), float(hi))
        elif tensor.dtype == trt.int8 and tensor.dynamic_range is not None:
            scale = max(abs(v) for v in tensor.dynamic_range) / _INT8_MAX
            known[tensor.name] = (_INT8_MIN * scale, _INT8_MAX * scale)
        elif tensor.dtype in (trt.float32, trt.float16, trt.int8):
            missing.append(tensor.name)
    if missing:
        err_msg = f"int8_ranges_hook has no input_ranges entry for the network inputs {missing}"
        raise ValueError(err_msg)
    return known


def _symmetric_bound(interval: _Interval) -> float:
    # int8 reaches -128 steps but only +127, so the scale is max(hi / 127, -lo / 128)
    lo, hi = interval
    return max(hi, lo * _INT8_MAX / _INT8_MIN, 0.0)


def _propagate(
    network: trt.INetworkDefinition,
    known: dict[str, _Interval],
) -> dict[str, _Interval]:
    lt = trt.LayerType
    rules: dict[object, Callable[..., _Interval | None]] = {
        lt.ACTIVATION: _activation,
        lt.CAST: _same,
        lt.CONCATENATION: _concatenation,
        lt.CONVOLUTION: _convolution,
        lt.DECONVOLUTION: _deconvolution,
        lt.ELEMENTWISE: _elementwise,
        lt.IDENTITY: _same,
        lt.PADDING: _with_zero,
        lt.POOLING: _pooling,
        lt.RESIZE: _resize,
        lt.SCALE: _scale,
        lt.SHUFFLE: _same,
        lt.SLICE: _slice,
        lt.SOFTMAX: _softmax,
        lt.UNARY: _unary,
    }
    intervals = dict(known)
    # parsed networks list their layers in topological order
    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        out = layer.get_output(0)
        if out is None or out.dtype not in (trt.float32, trt.float16, trt.int8):
            continue
        tensors = [layer.get_input(i) for i in range(layer.num_inputs)]
        ivs = [intervals.get(t.name) if t is not None else None for t in tensors]
        rule = rules.get(layer.type)
        result = None
        if layer.type == lt.CONSTANT:
            result = _constant(layer)
        elif rule is not None and ivs and ivs[0] is not None:
            result = rule(layer, ivs[0], ivs)
        if result is not None and all(math.isfinite(v) for v in result):
            intervals[out.name] = result
    return intervals


def _cast(layer: trt.ILayer, cls: type) -> trt.ILayer:
    layer.__class__ = cls
    return layer


def _values(weights: object) -> np.ndarray:
    return np.asarray(weights, dtype=np.float64).reshape(-1)


def _hull(*ivs: _Interval) -> _Interval:
    return min(iv[0] for iv in ivs), max(iv[1] for iv in ivs)


def _mul(a: _Interval, b: _Interval) -> _Interval:
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(products), max(products)


def _constant(layer: trt.ILayer) -> _Interval | None:
    weights = _values(_cast(layer, trt.IConstantLayer).weights)
    return (float(weights.min()), float(weights.max())) if weights.size else None


def _same(_layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    return x


def _with_zero(_layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    return _hull(x, (0.0, 0.0))


def _softmax(_layer: trt.ILayer, _x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    return 0.0, 1.0


def _concatenation(
    _layer: trt.ILayer, _x: _Interval, ivs: list[_Interval | None]
) -> _Interval | None:
    bounded = [iv for iv in ivs if iv is not None]
    return _hull(*bounded) if len(bounded) == len(ivs) else None


def _weighted_sum(weights: np.ndarray, bias: np.ndarray, x: _Interval) -> _Interval:
    # zero padding adds zero terms
    lo_x, hi_x = min(x[0], 0.0), max(x[1], 0.0)
    if bias.size != weights.shape[0]:
        bias = np.zeros(weights.shape[0])
    lo = bias + np.minimum(weights * lo_x, weights * hi_x).sum(axis=1)
    hi = bias + np.maximum(weights * lo_x, weights * hi_x).sum(axis=1)
    return float(lo.min()), float(hi.max())


def _convolution(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    conv = _cast(layer, trt.IConvolutionLayer)
    kernel = _values(conv.kernel)
    # a kernel given as a tensor input (e.g. Q/DQ weights) is not known here
    if conv.num_inputs > 1 or kernel.size == 0:
        return None
    return _weighted_sum(kernel.reshape(conv.num_output_maps, -1), _values(conv.bias), x)


def _deconvolution(
    layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]
) -> _Interval | None:
    deconv = _cast(layer, trt.IDeconvolutionLayer)
    kernel = _values(deconv.kernel)
    if deconv.num_inputs > 1 or kernel.size == 0:
        return None
    # the kernel is (in, out / groups, ...): bound each output by every input channel
    groups = deconv.num_groups
    per_group = deconv.num_output_maps // groups
    taps = int(np.prod(tuple(deconv.kernel_size_nd)))
    kernel = kernel.reshape(-1, per_group, taps).transpose(1, 0, 2)
    per_output = np.tile(kernel.reshape(per_group, -1), (groups, 1))
    return _weighted_sum(per_output, _values(deconv.bias), x)


def _scale(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    scale_layer = _cast(layer, trt.IScaleLayer)
    power = _values(scale_layer.power)
    if power.size and not np.all(power == 1.0):
        return None
    scale = _values(scale_layer.scale)
    shift = _values(scale_layer.shift)
    scale = scale if scale.size else np.ones(1)
    shift = shift if shift.size else np.zeros(1)
    lo = np.minimum(scale * x[0], scale * x[1]) + shift
    hi = np.maximum(scale * x[0], scale * x[1]) + shift
    return float(lo.min()), float(hi.max())


def _activation(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    act = _cast(layer, trt.IActivationLayer)
    kind = trt.ActivationType
    lo, hi = x
    if act.type == kind.RELU:
        return max(lo, 0.0), max(hi, 0.0)
    if act.type == kind.CLIP:
        return min(max(lo, act.alpha), act.beta), min(max(hi, act.alpha), act.beta)
    if act.type == kind.SIGMOID:
        return 0.5 * (1.0 + math.tanh(0.5 * lo)), 0.5 * (1.0 + math.tanh(0.5 * hi))
    if act.type == kind.TANH:
        return math.tanh(lo), math.tanh(hi)
    if act.type == kind.LEAKY_RELU and act.alpha >= 0:
        return _leaky_relu(lo, act.alpha), _leaky_relu(hi, act.alpha)
    return None


def _leaky_relu(value: float, alpha: float) -> float:
    return value if value >= 0 else alpha * value


def _unary(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    op = _cast(layer, trt.IUnaryLayer).op
    lo, hi = x
    if op == trt.UnaryOperation.NEG:
        return -hi, -lo
    if op != trt.UnaryOperation.ABS:
        return None
    if lo >= 0:
        return x
    if hi <= 0:
        return -hi, -lo
    return 0.0, max(-lo, hi)


def _pooling(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    # max pooling skips padding, average pooling may count it as zeros
    if _cast(layer, trt.IPoolingLayer).type == trt.PoolingType.MAX:
        return x
    return _hull(x, (0.0, 0.0))


def _resize(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    # cubic interpolation overshoots the input range
    if _cast(layer, trt.IResizeLayer).resize_mode == trt.InterpolationMode.CUBIC:
        return None
    return x


def _slice(layer: trt.ILayer, x: _Interval, ivs: list[_Interval | None]) -> _Interval | None:
    if _cast(layer, trt.ISliceLayer).mode != trt.SampleMode.FILL:
        return x
    fill = ivs[_SLICE_FILL_INPUT] if len(ivs) > _SLICE_FILL_INPUT else (0.0, 0.0)
    return _hull(x, fill) if fill is not None else None


def _elementwise(layer: trt.ILayer, a: _Interval, ivs: list[_Interval | None]) -> _Interval | None:
    b = ivs[1] if len(ivs) > 1 else None
    if b is None:
        return None
    op = _cast(layer, trt.IElementWiseLayer).op
    kind = trt.ElementWiseOperation
    if op == kind.SUM:
        return a[0] + b[0], a[1] + b[1]
    if op == kind.SUB:
        return a[0] - b[1], a[1] - b[0]
    if op == kind.PROD:
        return _mul(a, b)
    if op == kind.MAX:
        return max(a[0], b[0]), max(a[1], b[1])
    if op == kind.MIN:
        return min(a[0], b[0]), min(a[1], b[1])
    if op == kind.DIV and (b[0] > 0 or b[1] < 0):
        return _mul(a, (1.0 / b[1], 1.0 / b[0]))
    return None
