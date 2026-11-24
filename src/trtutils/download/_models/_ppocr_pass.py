# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""ONNX graph pass that keeps the PP-OCRv5 server detector's neck inside the FP16 range."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import onnx
    from onnx import numpy_helper
else:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        onnx = None
        numpy_helper = None

# the four 9x9 lateral convs that start the neck, and the batch norm after its fusion conv
_NECK_START = ("Conv.91", "Conv.92", "Conv.93", "Conv.94")
_NECK_END = "BatchNormalization.84"
# ops that commute with a positive scale
_HOMOGENEOUS = ("Identity", "Relu", "Resize", "MaxPool", "Concat")


def _neck_factors(
    node: onnx.NodeProto,
    scaled: set[str],
    constants: set[str],
    scale: float,
) -> dict[int, float]:
    # input index -> factor for the constants of a node fed by the scaled neck
    op = node.op_type
    if op in _HOMOGENEOUS:
        data = node.input[:1] if op in ("Resize", "MaxPool") else node.input
        if not all(i in scaled for i in data):
            err_msg = f"{node.name} mixes scaled and unscaled inputs"
            raise RuntimeError(err_msg)
        return {}
    if op == "Add":
        factors = {k: 1 / scale for k, name in enumerate(node.input) if name not in scaled}
        if any(node.input[k] not in constants for k in factors):
            err_msg = f"{node.name} adds an unscaled activation"
            raise RuntimeError(err_msg)
        return factors
    if op == "Conv" and node.input[0] in scaled:
        return {2: 1 / scale} if len(node.input) > 2 else {}  # noqa: PLR2004
    if op == "BatchNormalization":
        # y = gamma * (x - mean) / sqrt(var + eps) + beta, with x divided by scale:
        # the mean follows x, and the shift follows y unless this norm restores the scale
        if node.name == _NECK_END:
            return {1: scale, 3: 1 / scale}
        return {2: 1 / scale, 3: 1 / scale}
    err_msg = f"unexpected {op} {node.name} in the neck"
    raise RuntimeError(err_msg)


def scale_ppocrv5_server_det_neck(onnx_path: Path, scale: float = 256.0) -> None:
    """
    Divide the neck activations of the PP-OCRv5 server detector by ``scale``, in place.

    The neck's activations reach about 1.5e6 on ordinary pages, far past the FP16 maximum
    (65504), so an FP16 engine overflows there and detects nothing. The neck is linear up to
    its batch norms (convolutions, adds, nearest resizes, concats, ReLUs), so dividing the
    weights and biases of its first convolutions, every bias and batch norm shift inside it,
    and multiplying the closing batch norm's scale back gives the same output in exact
    arithmetic, with the neck about 6e3 at most.

    Parameters
    ----------
    onnx_path : Path
        The ``ch_PP-OCRv5_det_server.onnx`` model. Overwritten.
    scale : float
        The factor the neck is divided by. Default is 256.

    Raises
    ------
    RuntimeError
        If the graph does not have the expected neck.
    ImportError
        If onnx is not installed.

    """
    if onnx is None or numpy_helper is None:
        err_msg = "onnx is required for the PP-OCRv5 server detector export"
        raise ImportError(err_msg)
    model = onnx.load(str(onnx_path))
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}

    def _scaled(name: str, factor: float) -> str:
        # a new initializer, the original may be shared with nodes outside the neck
        arr = numpy_helper.to_array(inits[name]).astype(np.float32) * np.float32(factor)
        new = numpy_helper.from_array(arr, f"{name}_neck_x{factor:g}")
        graph.initializer.append(new)
        inits[new.name] = new
        return new.name

    scaled: set[str] = set()
    closed = False
    for node in graph.node:
        if node.name in _NECK_START:
            factors = dict.fromkeys(range(1, len(node.input)), 1 / scale)
        elif any(i in scaled for i in node.input):
            factors = _neck_factors(node, scaled, set(inits), scale)
        else:
            continue
        for k, factor in factors.items():
            node.input[k] = _scaled(node.input[k], factor)
        if node.name == _NECK_END:
            closed = True
        else:
            scaled.add(node.output[0])
    if not closed or any(o.name in scaled for o in graph.output):
        err_msg = f"{onnx_path} is not the PP-OCRv5 server detector"
        raise RuntimeError(err_msg)
    used = {i for node in graph.node for i in node.input}
    kept = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.save(model, str(onnx_path))
