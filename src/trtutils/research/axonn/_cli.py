# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CLI subcommands for AxoNN research module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trtutils.__main__ import _parse_shapes_arg
from trtutils._log import LOG
from trtutils.builder import ImageBatcher, SyntheticBatcher
from trtutils.builder.onnx import get_onnx_input, make_onnx_static
from trtutils.research.axonn import build_engine as axonn_build_engine

if TYPE_CHECKING:
    import argparse
    from types import SimpleNamespace


def _axonn_build(args: SimpleNamespace) -> None:
    """
    Build an AxoNN-optimized engine from an ONNX file.

    Fixes dynamic dimensions, profiles per-layer latency and energy on GPU and DLA,
    then solves for an energy-optimal GPU/DLA schedule using the AxoNN ILP.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:

        General (inherited):
        - log_level : str
            Logging level. Default is "INFO".
        - verbose : bool
            Enable verbose output.
        - nvtx : bool
            Enable NVTX markers for Nsight Systems profiling.

        Build common (inherited):
        - onnx : str
            Path to the ONNX model file.
        - output : str
            Path to save the TensorRT engine file.
        - timing_cache : str | None
            Path for timing cache data, or 'global'. Default is None.
        - calibration_cache : str | None
            Path for calibration cache data. Default is None.
        - workspace : float
            Workspace size in GB. Default is 4.0.
        - optimization_level : int
            TensorRT builder optimization level (0-5). Default is 3.
        - shape : list[str] | None
            Input binding shapes as NAME:dim1,dim2,... Default is None.
        - direct_io : bool
            Use direct IO for the engine.
        - prefer_precision_constraints : bool
            Prefer precision constraints.
        - reject_empty_algorithms : bool
            Reject empty algorithms.
        - ignore_timing_mismatch : bool
            Allow different CUDA device timing caches.
        - cache : bool
            Cache the engine in the trtutils engine cache.

        DLA (inherited):
        - dla_core : int | None
            DLA core to use. Defaults to 0 for AxoNN.

        Calibration (inherited):
        - calibration_dir : str | None
            Directory containing images for INT8 calibration.
        - input_shape : list[int] | None
            Input shape in HWC format.
        - input_dtype : str | None
            Input data type (float32, float16, int8).
        - batch_size : int
            Batch size for calibration. Default is 8.
        - data_order : str
            Data ordering (NCHW or NHWC). Default is "NCHW".
        - max_images : int | None
            Maximum calibration images.
        - resize_method : str
            Image resize method (letterbox or linear). Default is "letterbox".
        - input_scale : list[float]
            Input value range. Default is [0.0, 1.0].

        AxoNN-specific:
        - energy_target : float | None
            Explicit ECT in millijoules per inference. Overrides energy_ratio.
        - energy_ratio : float
            Fraction of GPU-only energy to use as ECT (0.0-1.0). Default is 0.8.
        - max_transitions : int
            Maximum GPU<->DLA transitions. Default is 1.
        - profile_iterations : int
            Number of profiling iterations. Default is 1000.
        - warmup_iterations : int
            Number of warmup iterations. Default is 50.
        - num_batches : int
            Number of synthetic calibration batches. Default is 10.

    """
    onnx_path = Path(args.onnx)
    output_path = Path(args.output)

    # make dynamic dims static for DLA compatibility
    make_onnx_static(onnx_path)

    # use --shape if provided, else derive from ONNX
    shapes = _parse_shapes_arg(getattr(args, "shape", None))
    if shapes is None:
        input_name, input_dims = get_onnx_input(onnx_path)
        shapes = [(input_name, input_dims)]
        imgsz = input_dims[-1]
    else:
        imgsz = shapes[0][1][-1]
    LOG.info(f"Input shapes: {shapes}")

    # create batcher
    if args.calibration_dir is not None:
        batcher: ImageBatcher | SyntheticBatcher = ImageBatcher(
            image_dir=args.calibration_dir,
            shape=args.input_shape,
            dtype=np.dtype(args.input_dtype).type,
            batch_size=args.batch_size,
            order=args.data_order,
            max_images=args.max_images,
            resize_method=args.resize_method,
            input_scale=tuple(args.input_scale),
            verbose=args.verbose,
        )
    else:
        batcher = SyntheticBatcher(
            shape=(imgsz, imgsz, 3),
            dtype=np.dtype(np.float32),
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            data_range=(0.0, 1.0),
            order="NCHW",
            verbose=args.verbose,
        )

    # DLA core defaults to 0 for AxoNN
    dla_core = args.dla_core if args.dla_core is not None else 0

    time_ms, energy_mj, transitions, gpu_layers, dla_layers = axonn_build_engine(
        onnx=onnx_path,
        output=output_path,
        calibration_batcher=batcher,
        energy_target=args.energy_target,
        energy_ratio=args.energy_ratio,
        max_transitions=args.max_transitions,
        dla_core=dla_core,
        profile_iterations=args.profile_iterations,
        warmup_iterations=args.warmup_iterations,
        workspace=args.workspace,
        timing_cache=args.timing_cache,
        calibration_cache=args.calibration_cache,
        shapes=shapes,
        optimization_level=args.optimization_level,
        direct_io=args.direct_io,
        prefer_precision_constraints=args.prefer_precision_constraints,
        reject_empty_algorithms=args.reject_empty_algorithms,
        ignore_timing_mismatch=args.ignore_timing_mismatch,
        cache=args.cache,
        verbose=args.verbose,
    )

    LOG.info("AxoNN Build Complete")
    LOG.info("=" * 40)
    LOG.info(f"Engine saved to: {output_path}")
    LOG.info(f"Expected latency : {time_ms:.2f} ms")
    LOG.info(f"Expected energy  : {energy_mj:.2f} mJ")
    LOG.info(f"GPU layers       : {gpu_layers}")
    LOG.info(f"DLA layers       : {dla_layers}")
    LOG.info(f"Transitions      : {transitions}")
    LOG.info("=" * 40)


def register_cli(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
    """
    Register AxoNN CLI subcommands under the research parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers action from the research CLI parser.
    parents : list[argparse.ArgumentParser]
        Parent parsers providing general, DLA, build common, and calibration args.

    """
    axonn_parser = subparsers.add_parser(
        "axonn",
        help="AxoNN: Energy-aware multi-accelerator neural network inference.",
    )
    axonn_subparsers = axonn_parser.add_subparsers(
        title="axonn commands",
        dest="axonn_command",
        required=False,
    )
    axonn_parser.set_defaults(func=lambda _args: axonn_parser.print_help())

    # build command
    build_parser = axonn_subparsers.add_parser(
        "build",
        help="Build an AxoNN-optimized engine from an ONNX file.",
        parents=parents,
    )
    # Only axonn-specific args below; general, build_common, calibration,
    # and dla args are inherited from parents.
    build_parser.add_argument(
        "--energy_target",
        type=float,
        default=None,
        help=(
            "Explicit Energy Consumption Target (ECT) in millijoules per inference. "
            "Overrides --energy_ratio when set."
        ),
    )
    build_parser.add_argument(
        "--energy_ratio",
        type=float,
        default=0.8,
        help=(
            "Fraction of GPU-only baseline energy to use as ECT (0.0-1.0). "
            "E.g. 0.8 = schedule must use <=80%% of GPU-only energy. "
            "Lower values push more layers to DLA. Default is 0.8."
        ),
    )
    build_parser.add_argument(
        "--max_transitions",
        type=int,
        default=1,
        help="Maximum GPU<->DLA transitions. Default is 1.",
    )
    build_parser.add_argument(
        "--profile_iterations",
        type=int,
        default=1000,
        help="Number of profiling iterations. Default is 1000.",
    )
    build_parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=50,
        help="Number of warmup iterations. Default is 50.",
    )
    build_parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of synthetic batches (when no calibration_dir). Default is 10.",
    )
    build_parser.set_defaults(func=_axonn_build)
