# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Callable
from pathlib import Path

from typing_extensions import Concatenate, ParamSpec

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils._log import LOG
from trtutils.builder._build import build_engine

from ._profiler import ProfilerResult, profile_engine


P = ParamSpec("P")

BuildFunc = Callable[Concatenate[Path | str, Path | str, P], None]


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
    them layer-by-layer, and computes the speedup percentage for each layer. Layers
    are returned in execution order.

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
        Whether to ignore layers that are only present in one datatype.
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
        - List of (layer_name, speedup_percent) tuples in execution order.
          Positive values indicate INT8 is faster, negative values indicate INT8 is slower.

    Raises
    ------
    RuntimeError
        If the ONNX model cannot be parsed or engines fail to build
    ValueError
        If engine building or profiling logic encounters invalid configuration

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

    # Create temporary files for engines
    with tempfile.NamedTemporaryFile(
        suffix=".engine", delete=False
    ) as fp16_file, tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as int8_file:
        fp16_path = Path(fp16_file.name)
        int8_path = Path(int8_file.name)

        if verbose:
            LOG.info(f"Building FP16 engine for ONNX model {onnx.stem}")
        build_func(
            onnx_path=onnx,
            output_path=fp16_path,
            fp16=True,
            int8=False,
        )

        if verbose:
            LOG.info(f"Building INT8 engine for ONNX model {onnx.stem}")
        build_func(
            onnx_path=onnx,
            output_path=int8_path,
            fp16=False,
            int8=True,
        )

        if verbose:
            LOG.info(f"Profiling FP16 engine for ONNX model {onnx.stem}")
        fp16_results = profile_engine(
            fp16_path,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            verbose=verbose,
        )

        if verbose:
            LOG.info(f"Profiling INT8 engine for ONNX model {onnx.stem}")
        int8_results = profile_engine(
            int8_path,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            verbose=verbose,
        )

        fp16_layer_map = {layer.name: layer for layer in fp16_results.layers}
        int8_layer_map = {layer.name: layer for layer in int8_results.layers}

        # validate the layer names are the same
        if set(fp16_layer_map.keys()) != set(int8_layer_map.keys()) and not ignore_mismatch_layers:
            err_msg = "Layer names do not match between FP16 and INT8 results"
            raise ValueError(err_msg)

        layer_deltas: list[tuple[str, float]] = []

        for fp16_layer_name, fp16_layer in fp16_layer_map.items():
            if fp16_layer_name not in int8_layer_map:
                if ignore_mismatch_layers:
                    continue
                else:
                    err_msg = f"Layer {fp16_layer_name} found in FP16 but not in INT8 results"
                    raise ValueError(err_msg)

            int8_layer = int8_layer_map[fp16_layer_name]
            fp16_time = fp16_layer.mean
            int8_time = int8_layer.mean

            # Compute speedup percentage: (fp16_time - int8_time) / fp16_time * 100
            # Positive = INT8 is faster, negative = INT8 is slower
            if fp16_time > 0:
                speedup_percent = ((fp16_time - int8_time) / fp16_time) * 100.0
            else:
                speedup_percent = 0.0

            layer_deltas.append((fp16_layer_name, speedup_percent))

        return fp16_results, int8_results, layer_deltas
