# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

from trtutils._log import LOG

if TYPE_CHECKING:
    from collections.abc import Sequence


@lru_cache(maxsize=1)
def _myl_suffix_pattern() -> re.Pattern[str]:
    return re.compile(r"_myl\d+_\d+$")


@lru_cache(maxsize=200)
def strip_myelin_suffix(layer_name: str) -> str:
    """
    Strip TRT 10+ Myelin ``_myl<N>_<N>`` suffix from a profiled layer name.

    Parameters
    ----------
    layer_name : str
        A TRT profiled layer name, e.g. ``"/model.0/conv/Conv_myl0_4"``.

    Returns
    -------
    str
        The base name without suffix, or the original name if no suffix found.

    """
    return _myl_suffix_pattern().sub("", layer_name)


def build_fused_layer_map(
    profiled_layer_names: Sequence[str],
    onnx_layer_names: list[str] | None = None,
) -> dict[str, tuple[str, int]]:
    """
    Map individual ONNX layer names to their fused TRT layer.

    TensorRT fuses layers in two ways:

    - **GPU engines**: ``"LayerA + LayerB + LayerC"``
    - **DLA engines**: ``"{ForeignNode[FirstLayer...LastLayer]}"``

    This function parses both formats and maps each constituent ONNX layer
    name to the fused name and total constituent count.

    Parameters
    ----------
    profiled_layer_names : Sequence[str]
        Layer names from a profiled TensorRT engine (possibly fused).
    onnx_layer_names : list[str] | None, optional
        Ordered list of all ONNX layer names. Required to resolve DLA
        ``ForeignNode`` ranges which only name the first and last layer.

    Returns
    -------
    dict[str, tuple[str, int]]
        Maps individual layer name -> ``(fused_name, num_constituents)``.

    """
    fused_map: dict[str, tuple[str, int]] = {}
    for fused_name in profiled_layer_names:
        # case 1: GPU fusion: "LayerA + LayerB + LayerC"
        parts = [p.strip() for p in fused_name.split(" + ")]
        if len(parts) > 1:
            for part in parts:
                fused_map[part] = (fused_name, len(parts))
            continue

        # case 2: DLA ForeignNode: "{ForeignNode[/first/Layer.../last/Layer]}"
        if fused_name.startswith("{ForeignNode[") and fused_name.endswith("]}"):
            inner = fused_name[len("{ForeignNode[") : -len("]}") :]
            range_parts = inner.split("...")
            name_length = 2
            # case 2.a: range of layers (ForeignNode[first...last])
            if len(range_parts) == name_length and onnx_layer_names is not None:
                first_name, last_name = range_parts[0].strip(), range_parts[1].strip()
                try:
                    first_idx = onnx_layer_names.index(first_name)
                    last_idx = onnx_layer_names.index(last_name)
                    covered = onnx_layer_names[first_idx : last_idx + 1]
                    for name in covered:
                        fused_map[name] = (fused_name, len(covered))
                except ValueError:
                    pass
            # case 2.b: single layer (ForeignNode[/layer/Name])
            elif len(range_parts) == 1 and onnx_layer_names is not None:
                single_name = range_parts[0].strip()
                if single_name in onnx_layer_names:
                    fused_map[single_name] = (fused_name, 1)
            continue

        # case 3: TRT 10+ Myelin suffix: "/model.0/conv/Conv_myl0_4" -> "/model.0/conv/Conv"
        if onnx_layer_names is not None:
            base_name = strip_myelin_suffix(fused_name)
            if base_name != fused_name and base_name in onnx_layer_names:
                fused_map[base_name] = (fused_name, 1)

    return fused_map


def resolve_fused_layer_value(
    layer_name: str,
    layer_values: dict[str, float],
    fused_map: dict[str, tuple[str, int]],
    *,
    verbose: bool | None = None,
    label: str = "",
) -> float | None:
    """
    Look up a per-layer metric value, handling TensorRT layer fusion.

    Resolution order:

    1. Exact match in ``layer_values``
    2. Constituent of a fused layer (value split evenly among constituents)
    3. ``None`` — no data available

    Parameters
    ----------
    layer_name : str
        The ONNX layer name to look up.
    layer_values : dict[str, float]
        Metric values keyed by TRT layer name (e.g., execution times).
    fused_map : dict[str, tuple[str, int]]
        Mapping from constituent names to ``(fused_name, num_parts)``,
        as returned by :func:`build_fused_layer_map`.
    verbose : bool | None, optional
        Log warnings for unresolved layers.
    label : str, optional
        Label for log messages (e.g., ``"FP16"`` or ``"INT8"``).

    Returns
    -------
    float | None
        The resolved value, or ``None`` if no data found.

    """
    # case 1: exact match
    if layer_name in layer_values:
        return layer_values[layer_name]

    # case 2: fused layer match, split value evenly
    if layer_name in fused_map:
        fused_name, num_parts = fused_map[layer_name]
        if fused_name in layer_values:
            return layer_values[fused_name] / num_parts

    # case 3: substring match, TRT 10+ appends suffixes like _myl0_N
    for profiled_name, value in layer_values.items():
        if layer_name in profiled_name:
            return value

    # case 4: no data
    if verbose:
        LOG.warning(f"Layer {layer_name} not found in {label} profile")
    return None
