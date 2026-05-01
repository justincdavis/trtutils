# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
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

from ._fusion import strip_myelin_suffix
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


def _build_chunk_map(
    layers: list[tuple[str, float]],
) -> tuple[list[str], dict[str, float]]:
    """
    Split profiled layers into chunks keyed by anchor names.

    An "anchor" is a profiled layer whose name (after stripping the TRT suffix)
    starts with ``/`` — i.e. it originated from an ONNX node. All layers between
    consecutive anchors are grouped into the preceding anchor's chunk.

    Layers before the first anchor are grouped into a ``"__prologue__"`` chunk.

    Parameters
    ----------
    layers : list[tuple[str, float]]
        Ordered list of ``(profiled_name, mean_time)`` pairs.

    Returns
    -------
    tuple[list[str], dict[str, float]]
        ``(ordered_keys, chunk_times)`` where ``ordered_keys`` preserves
        insertion order and ``chunk_times`` maps each key to its total time.

    """
    ordered_keys: list[str] = []
    chunk_times: dict[str, float] = {}
    current_key = "__prologue__"

    for name, time_ms in layers:
        base = strip_myelin_suffix(name)
        if base.startswith("/"):
            # anchor layer, start new chunk
            current_key = base
            if current_key not in chunk_times:
                ordered_keys.append(current_key)
                chunk_times[current_key] = 0.0
        elif current_key not in chunk_times:
            ordered_keys.append(current_key)
            chunk_times[current_key] = 0.0

        chunk_times[current_key] += time_ms

    return ordered_keys, chunk_times


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

    Profiled layers are grouped into chunks anchored by layers whose names
    correspond to ONNX nodes. Fused TRT-internal layers (e.g. ``__myl_Silu``)
    between anchors are included in the preceding anchor's chunk. When FP16
    and INT8 fuse differently, chunks that appear in both profiles are compared
    directly.

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
        - List of (layer_name, speedup_percent) tuples in execution order.
          Positive values indicate INT8 is faster, negative values indicate INT8 is slower.

    Raises
    ------
    ValueError
        If a layer cannot be resolved in both profiles and mismatches are not ignored.

    """
    if build_func is None:
        with tempfile.NamedTemporaryFile(
            suffix=".timingcache",
            delete=False,
        ) as timing_cache_file:
            timing_cache_path = Path(timing_cache_file.name)

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
                timing_cache=timing_cache_path,
                profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
                verbose=verbose,
            )

    onnx_path = Path(onnx)
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
            fp16=True,
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

        fp16_layers = [(layer.name, layer.mean) for layer in fp16_results.layers]
        fp16_keys, fp16_chunks = _build_chunk_map(fp16_layers)
        int8_layers = [(layer.name, layer.mean) for layer in int8_results.layers]
        int8_keys, int8_chunks = _build_chunk_map(int8_layers)

        layer_deltas: list[tuple[str, float]] = []
        for key in fp16_keys:
            if key not in int8_chunks:
                if ignore_mismatch_layers:
                    continue
                err_msg = f"Layer {key} found in FP16 but not INT8 profile"
                raise ValueError(err_msg)

            fp16_time = fp16_chunks[key]
            int8_time = int8_chunks[key]

            speedup_percent = ((fp16_time - int8_time) / fp16_time) * 100.0 if fp16_time > 0 else 0.0
            layer_deltas.append((key, speedup_percent))

        # check missing int8 chunks
        if not ignore_mismatch_layers:
            for key in int8_keys:
                if key not in fp16_chunks:
                    err_msg = f"Layer {key} found in INT8 but not FP16 profile"
                    raise ValueError(err_msg)

        return fp16_results, int8_results, layer_deltas
