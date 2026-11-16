# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trtutils.builder._utils import check_dynamic_range
from trtutils.compat._libs import trt

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
        or int8 network input has no range, or an entry has min > max.

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
        for tensor, (lo, hi) in _propagate(network, ranges):
            bound = max(hi, lo * _INT8_MAX / _INT8_MIN)
            if bound > 0 and tensor.dynamic_range is None:
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
            lo, hi = map(float, ranges[tensor.name])
            if not lo <= hi:
                err_msg = f"input_ranges[{tensor.name!r}] must be (min, max), got ({lo}, {hi})"
                raise ValueError(err_msg)
            known[tensor.name] = (lo, hi)
        elif tensor.dtype == trt.int8 and tensor.dynamic_range is not None:
            scale = max(abs(v) for v in tensor.dynamic_range) / _INT8_MAX
            known[tensor.name] = (_INT8_MIN * scale, _INT8_MAX * scale)
        elif tensor.dtype in (trt.float32, trt.float16, trt.int8):
            missing.append(tensor.name)
    if missing:
        err_msg = f"int8_ranges_hook has no input_ranges entry for the network inputs {missing}"
        raise ValueError(err_msg)
    return known


def _propagate(
    network: trt.INetworkDefinition,
    ranges: Mapping[str, tuple[float, float]],
) -> list[tuple[trt.ITensor, _Interval]]:
    rules: dict[str, Callable[..., _Interval | None]] = {
        "ACTIVATION": _activation,
        "CAST": _same,
        "CONCATENATION": _concatenation,
        "CONVOLUTION": _convolution,
        "DECONVOLUTION": _convolution,
        "ELEMENTWISE": _elementwise,
        "IDENTITY": _same,
        "PADDING": lambda _layer, x, _ivs: _with_zero(x),
        "POOLING": _pooling,
        "RESIZE": _resize,
        "SCALE": _scale,
        "SHUFFLE": _same,
        "SLICE": _slice,
        "UNARY": _unary,
    }
    inputs = [network.get_input(idx) for idx in range(network.num_inputs)]
    intervals = _input_intervals(inputs, ranges)
    bounded = [(tensor, intervals[tensor.name]) for tensor in inputs if tensor.name in intervals]
    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        out = layer.get_output(0)
        if out is None or out.dtype not in (trt.float32, trt.float16, trt.int8):
            continue
        tensors = [layer.get_input(i) for i in range(layer.num_inputs)]
        ivs = [intervals.get(t.name) if t is not None else None for t in tensors]
        kind = layer.type.name
        rule = rules.get(kind)
        interval = None
        if kind == "CONSTANT":
            interval = _constant(layer)
        elif kind == "SOFTMAX":
            interval = (0.0, 1.0)
        elif rule is not None and ivs[0] is not None:
            interval = rule(layer, ivs[0], ivs)
        if interval is not None and all(math.isfinite(v) for v in interval):
            intervals[out.name] = interval
            bounded.append((out, interval))
    return bounded


def _cast(layer: trt.ILayer, cls: type) -> trt.ILayer:
    layer.__class__ = cls
    return layer


def _values(weights: object, default: float) -> np.ndarray:
    values = np.asarray(weights).reshape(-1)
    return values if values.size else np.full(1, default)


def _hull(*ivs: _Interval) -> _Interval:
    return min(iv[0] for iv in ivs), max(iv[1] for iv in ivs)


def _with_zero(x: _Interval) -> _Interval:
    return min(x[0], 0.0), max(x[1], 0.0)


def _mul(a: _Interval, b: _Interval) -> _Interval:
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(products), max(products)


def _affine(rows: np.ndarray, bias: np.ndarray, x: _Interval) -> _Interval:
    pos = np.maximum(rows, 0.0).sum(axis=1, dtype=np.float64)
    neg = np.minimum(rows, 0.0).sum(axis=1, dtype=np.float64)
    lo = bias + pos * x[0] + neg * x[1]
    hi = bias + pos * x[1] + neg * x[0]
    return float(lo.min()), float(hi.max())


def _constant(layer: trt.ILayer) -> _Interval | None:
    weights = np.asarray(_cast(layer, trt.IConstantLayer).weights)
    return (float(weights.min()), float(weights.max())) if weights.size else None


def _same(_layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    return x


def _concatenation(
    _layer: trt.ILayer, _x: _Interval, ivs: list[_Interval | None]
) -> _Interval | None:
    bounded = [iv for iv in ivs if iv is not None]
    return _hull(*bounded) if len(bounded) == len(ivs) else None


def _convolution(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    deconv = layer.type == trt.LayerType.DECONVOLUTION
    conv = _cast(layer, trt.IDeconvolutionLayer if deconv else trt.IConvolutionLayer)
    kernel = np.asarray(conv.kernel)
    if conv.num_inputs > 1 or kernel.size == 0:
        return None
    if deconv:
        per_group = conv.num_output_maps // conv.num_groups
        taps = int(np.prod(tuple(conv.kernel_size_nd)))
        kernel = kernel.reshape(conv.num_groups, -1, per_group, taps).swapaxes(1, 2)
    rows = kernel.reshape(conv.num_output_maps, -1)
    return _affine(rows, _values(conv.bias, 0.0), _with_zero(x))


def _scale(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    scale_layer = _cast(layer, trt.IScaleLayer)
    if not np.all(_values(scale_layer.power, 1.0) == 1.0):
        return None
    scale = _values(scale_layer.scale, 1.0)
    return _affine(scale.reshape(-1, 1), _values(scale_layer.shift, 0.0), x)


def _activation(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    act = _cast(layer, trt.IActivationLayer)
    kind, alpha, beta = trt.ActivationType, act.alpha, act.beta
    monotone: dict[object, Callable[[float], float]] = {
        kind.RELU: lambda v: max(v, 0.0),
        kind.CLIP: lambda v: min(max(v, alpha), beta),
        kind.SIGMOID: lambda v: 0.5 * (1.0 + math.tanh(0.5 * v)),
        kind.TANH: math.tanh,
    }
    if alpha >= 0:
        monotone[kind.LEAKY_RELU] = lambda v: v if v >= 0 else alpha * v
    func = monotone.get(act.type)
    return (func(x[0]), func(x[1])) if func is not None else None


def _unary(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    op = _cast(layer, trt.IUnaryLayer).op
    lo, hi = x
    if op == trt.UnaryOperation.NEG:
        return -hi, -lo
    if op == trt.UnaryOperation.ABS:
        return max(lo, -hi, 0.0), max(hi, -lo)
    return None


def _pooling(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    if _cast(layer, trt.IPoolingLayer).type == trt.PoolingType.MAX:
        return x
    return _with_zero(x)


def _resize(layer: trt.ILayer, x: _Interval, _ivs: list[_Interval | None]) -> _Interval | None:
    if _cast(layer, trt.IResizeLayer).resize_mode == trt.InterpolationMode.CUBIC:
        return None
    return x


def _slice(layer: trt.ILayer, x: _Interval, ivs: list[_Interval | None]) -> _Interval | None:
    if _cast(layer, trt.ISliceLayer).mode != trt.SampleMode.FILL:
        return x
    fill = ivs[_SLICE_FILL_INPUT] if len(ivs) > _SLICE_FILL_INPUT else (0.0, 0.0)
    return _hull(x, fill) if fill is not None else None


def _elementwise(layer: trt.ILayer, a: _Interval, ivs: list[_Interval | None]) -> _Interval | None:
    b = ivs[1]
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
