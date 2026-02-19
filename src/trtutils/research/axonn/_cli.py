# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CLI subcommands for AxoNN research module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx

from trtutils.__main__ import _parse_shapes_arg
from trtutils._log import LOG
from trtutils.builder import ImageBatcher, SyntheticBatcher
from trtutils.research.axonn import build_engine as axonn_build_engine

if TYPE_CHECKING:
    import argparse
    from types import SimpleNamespace


def _get_onnx_input(onnx_path: Path) -> tuple[str, tuple[int, ...]]:
    """Read the first input tensor name and shape from an ONNX model."""
    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    name = inp.name
    dims = tuple(d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim)
    return name, dims


def _make_onnx_static(onnx_path: Path) -> None:
    """Set any dynamic dimensions in the ONNX model to 1 (batch size)."""
    model = onnx.load(str(onnx_path))
    changed = False
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value <= 0:
                dim.ClearField("dim_param")
                dim.dim_value = 1
                changed = True
    if changed:
        onnx.save(model, str(onnx_path))
        LOG.info(f"Fixed dynamic dimensions in {onnx_path}")


def _make_batcher(
    imgsz: int,
    batch_size: int = 8,
    num_batches: int = 10,
) -> SyntheticBatcher:
    """Create a SyntheticBatcher for calibration."""
    return SyntheticBatcher(
        shape=(imgsz, imgsz, 3),
        dtype=np.dtype(np.float32),
        batch_size=batch_size,
        num_batches=num_batches,
        data_range=(0.0, 1.0),
        order="NCHW",
    )


def _axonn_build(args: SimpleNamespace) -> None:
    """Build an AxoNN-optimized engine from an ONNX file."""
    onnx_path = Path(args.onnx)
    output_path = Path(args.output)

    # Fix dynamic dims for DLA compatibility
    _make_onnx_static(onnx_path)

    # Shapes: use --shape if provided, else derive from ONNX
    shapes = _parse_shapes_arg(getattr(args, "shape", None))
    if shapes is None:
        input_name, input_dims = _get_onnx_input(onnx_path)
        shapes = [(input_name, input_dims)]
        imgsz = input_dims[-1]
    else:
        imgsz = shapes[0][1][-1]
    LOG.info(f"Input shapes: {shapes}")

    # Create batcher: use real images if calibration_dir provided, else synthetic
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
        batcher = _make_batcher(imgsz, args.batch_size, args.num_batches)

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
    """Register AxoNN CLI subcommands under the research parser."""
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

    # -- build subcommand --
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
