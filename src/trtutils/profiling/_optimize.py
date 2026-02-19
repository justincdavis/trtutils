# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

from trtutils._log import LOG
from trtutils.builder._build import build_engine
from trtutils.compat._libs import trt
from trtutils.inspect._onnx import inspect_onnx_layers

from ._fusion import build_fused_layer_map, resolve_fused_layer_value
from ._profiler import ProfilerResult, profile_engine


class BuildFunc(Protocol):
    def __call__(
        self,
        onnx_path: Path | str,
        output_path: Path | str,
        *,
        fp16: bool | None = None,
        int8: bool | None = None,
    ) -> None: ...


def identify_quantize_speedups_by_layer(
    onnx: Path | str,
    *,
    build_func: BuildFunc | None = None,
    iterations: int = 100,
    warmup_iterations: int = 10,
    workspace: float = 4.0,
    ignore_mismatch_layers: bool = False,
    verbose: bool | None = None,
) -> tuple[ProfilerResult, ProfilerResult, list[tuple[str, float]]]:
    """
    Identify speedup by layer from INT8 quantization.

    This function builds both FP16 and INT8 engines from an ONNX model (using either a
    user-provided build function or :func:`trtutils.builder.build_engine`), profiles
    them layer-by-layer, and computes the speedup percentage for each layer.

    When TensorRT fuses layers differently at FP16 vs INT8, ONNX layer names are used
    as a common reference and fused layer costs are split evenly among their
    constituents.

    Parameters
    ----------
    onnx : Path | str
        The path to the ONNX model.
    build_func : BuildFunc | None, optional
        A callable responsible for building engines. It must accept
        ``(onnx, output, *, fp16=None, int8=None)`` (or **kwargs including these)
        and build an engine at ``output`` with the requested precision flags.

        - When ``build_func`` is ``None`` (default), a thin wrapper over
          :func:`trtutils.builder.build_engine` is used with:

          - ``workspace`` forwarded from this function
          - ``profiling_verbosity=trt.ProfilingVerbosity.DETAILED``
          - ``verbose`` forwarded from this function

        - To customize calibration, shapes, timing cache, hooks, etc., pass a
          ``functools.partial`` or custom function that captures those arguments.
    iterations : int, optional
        The number of profiling iterations to run, by default 100.
    warmup_iterations : int, optional
        The number of warmup iterations to run before profiling, by default 10.
    workspace : float, optional
        The size of the workspace in gigabytes. Default is 4.0 GiB.
    ignore_mismatch_layers : bool, optional
        Whether to ignore layers that cannot be resolved in both profiles.
        Default is False.
    verbose : bool, optional
        Whether to output additional information to stdout.
        Default None/False.

    Returns
    -------
    tuple[ProfilerResult, ProfilerResult, list[tuple[str, float]]]
        A tuple containing:
        - FP16 profiling results
        - INT8 profiling results
        - List of (layer_name, speedup_percent) tuples in ONNX layer order.
          Positive values indicate INT8 is faster, negative values indicate INT8 is slower.

    Raises
    ------
    ValueError
        If a layer cannot be resolved in both profiles and mismatches are not ignored.

    """
    if build_func is None:

        def build_func(
            onnx_path: Path | str,
            output_path: Path | str,
            *,
            fp16: bool | None = None,
            int8: bool | None = None,
        ) -> None:
            build_engine(
                onnx=onnx_path,
                output=output_path,
                workspace=workspace,
                fp16=fp16,
                int8=int8,
                profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
                verbose=verbose,
            )

    onnx_path = Path(onnx)

    # Get ONNX layer names as common reference
    onnx_layers = inspect_onnx_layers(onnx_path, verbose=False)
    onnx_layer_names = [layer.name for layer in onnx_layers]

    # Create temporary files for engines
    with tempfile.NamedTemporaryFile(
        suffix=".engine", delete=False
    ) as fp16_file, tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as int8_file:
        fp16_path = Path(fp16_file.name)
        int8_path = Path(int8_file.name)

        if verbose:
            LOG.info(f"Building FP16 engine for ONNX model {onnx_path.stem}")
        build_func(
            onnx_path=onnx_path,
            output_path=fp16_path,
            fp16=True,
            int8=False,
        )

        if verbose:
            LOG.info(f"Building INT8 engine for ONNX model {onnx_path.stem}")
        build_func(
            onnx_path=onnx_path,
            output_path=int8_path,
            fp16=False,
            int8=True,
        )

        if verbose:
            LOG.info(f"Profiling FP16 engine for ONNX model {onnx_path.stem}")
        fp16_results = profile_engine(
            fp16_path,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            verbose=verbose,
        )

        if verbose:
            LOG.info(f"Profiling INT8 engine for ONNX model {onnx_path.stem}")
        int8_results = profile_engine(
            int8_path,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            verbose=verbose,
        )

        # Build fusion maps from profiled layer names
        fp16_layer_names = [layer.name for layer in fp16_results.layers]
        int8_layer_names = [layer.name for layer in int8_results.layers]

        fp16_fused_map = build_fused_layer_map(fp16_layer_names, onnx_layer_names)
        int8_fused_map = build_fused_layer_map(int8_layer_names, onnx_layer_names)

        # Build value dicts keyed by profiled layer name
        fp16_values: dict[str, float] = {layer.name: layer.mean for layer in fp16_results.layers}
        int8_values: dict[str, float] = {layer.name: layer.mean for layer in int8_results.layers}

        # Resolve per-ONNX-layer timing through fusion maps
        layer_deltas: list[tuple[str, float]] = []

        for layer_name in onnx_layer_names:
            fp16_time = resolve_fused_layer_value(
                layer_name,
                fp16_values,
                fp16_fused_map,
            )
            int8_time = resolve_fused_layer_value(
                layer_name,
                int8_values,
                int8_fused_map,
            )

            if fp16_time is None or int8_time is None:
                if ignore_mismatch_layers:
                    continue
                err_msg = f"Layer {layer_name} could not be resolved in both FP16 and INT8 profiles"
                raise ValueError(err_msg)

            # Positive = INT8 is faster, negative = INT8 is slower
            speedup_percent = ((fp16_time - int8_time) / fp16_time) * 100.0 if fp16_time > 0 else 0.0

            layer_deltas.append((layer_name, speedup_percent))

        return fp16_results, int8_results, layer_deltas
