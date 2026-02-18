# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""CLI subcommands for AxoNN research module."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx
import tensorrt as trt

from trtutils import build_engine as trt_build_engine
from trtutils._log import LOG
from trtutils.builder import ImageBatcher, SyntheticBatcher
from trtutils.download import download, get_supported_models
from trtutils.jetson import benchmark_engine
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

    # Read ONNX input shape to derive imgsz
    input_name, input_dims = _get_onnx_input(onnx_path)
    imgsz = input_dims[-1]
    shapes = [(input_name, input_dims)]
    LOG.info(f"ONNX input: name={input_name!r}, shape={input_dims}")

    # Create batcher: use real images if calibration_dir provided, else synthetic
    if args.calibration_dir is not None:
        batcher = ImageBatcher(
            image_dir=args.calibration_dir,
            shape=(imgsz, imgsz, 3),
            dtype=np.dtype(np.float32),
            batch_size=args.batch_size,
            order="NCHW",
            verbose=args.verbose,
        )
    else:
        batcher = _make_batcher(imgsz, args.batch_size, args.num_batches)

    time_ms, energy_mj, transitions, gpu_layers, dla_layers = axonn_build_engine(
        onnx=onnx_path,
        output=output_path,
        calibration_batcher=batcher,
        energy_target=args.energy_target,
        energy_ratio=args.energy_ratio,
        max_transitions=args.max_transitions,
        dla_core=args.dla_core,
        profile_iterations=args.profile_iterations,
        warmup_iterations=args.warmup_iterations,
        workspace=args.workspace,
        timing_cache=args.timing_cache,
        calibration_cache=args.calibration_cache,
        shapes=shapes,
        optimization_level=args.optimization_level,
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


def _axonn_benchmark(args: SimpleNamespace) -> None:
    """Full GPU vs DLA vs AxoNN comparison benchmark."""
    model_name: str = args.model
    supported = get_supported_models()
    if model_name not in supported:
        LOG.error(f"Unknown model '{model_name}'. Supported: {', '.join(sorted(supported))}")
        return

    data_dir = Path(args.data_dir)
    model_dir = data_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / f"{model_name}.onnx"
    engine_gpu = model_dir / f"{model_name}_gpu.engine"
    engine_dla = model_dir / f"{model_name}_dla.engine"
    engine_axonn = model_dir / f"{model_name}_axonn.engine"

    # Step 1: Download ONNX
    if onnx_path.exists() and not args.rebuild:
        LOG.info(f"ONNX model already exists: {onnx_path}")
    else:
        LOG.info(f"Downloading {model_name} ONNX model...")
        download(
            model=model_name,
            output=onnx_path,
            imgsz=args.imgsz,
            accept=True,
            verbose=args.verbose,
        )
        LOG.info(f"Saved ONNX model to: {onnx_path}")

    # Fix dynamic dims
    _make_onnx_static(onnx_path)

    # Read input tensor info
    input_name, input_dims = _get_onnx_input(onnx_path)
    imgsz = input_dims[-1]
    shapes = [(input_name, input_dims)]
    LOG.info(f"ONNX input: name={input_name!r}, shape={input_dims}")

    # Step 2: Build GPU engine
    if engine_gpu.exists() and not args.rebuild:
        LOG.info(f"GPU engine already exists: {engine_gpu}")
    else:
        LOG.info("Building GPU-only engine (FP16)...")
        trt_build_engine(
            onnx=onnx_path,
            output=engine_gpu,
            fp16=True,
            shapes=shapes,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            verbose=args.verbose,
        )
        LOG.info(f"Saved GPU engine to: {engine_gpu}")

    # Step 3: Build DLA engine
    if engine_dla.exists() and not args.rebuild:
        LOG.info(f"DLA engine already exists: {engine_dla}")
    else:
        LOG.info(f"Building DLA engine (INT8+FP16, DLA core {args.dla_core}, GPU fallback)...")
        trt_build_engine(
            onnx=onnx_path,
            output=engine_dla,
            default_device=trt.DeviceType.DLA,
            gpu_fallback=True,
            fp16=True,
            int8=True,
            data_batcher=_make_batcher(imgsz),
            dla_core=args.dla_core,
            shapes=shapes,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            verbose=args.verbose,
        )
        LOG.info(f"Saved DLA engine to: {engine_dla}")

    # Step 4: Build AxoNN engine
    if engine_axonn.exists() and not args.rebuild:
        LOG.info(f"AxoNN engine already exists: {engine_axonn}")
        axonn_build = (0.0, 0.0, 0, 0, 0)
    else:
        LOG.info(
            f"Building AxoNN engine "
            f"(energy_ratio={args.energy_ratio}, max_transitions={args.max_transitions})..."
        )
        axonn_build = axonn_build_engine(
            onnx=onnx_path,
            output=engine_axonn,
            calibration_batcher=_make_batcher(imgsz),
            energy_ratio=args.energy_ratio,
            max_transitions=args.max_transitions,
            profile_iterations=args.profile_iterations,
            warmup_iterations=args.warmup_iterations,
            dla_core=args.dla_core,
            shapes=shapes,
            verbose=args.verbose,
        )
        LOG.info(f"Saved AxoNN engine to: {engine_axonn}")

    # Step 5: Benchmark each engine
    LOG.info("Benchmarking GPU-only engine...")
    gpu_result = benchmark_engine(
        engine_gpu,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        cuda_graph=args.cuda_graph or None,
        verbose=args.verbose,
    )
    gc.collect()

    LOG.info("Benchmarking DLA-only engine...")
    dla_result = benchmark_engine(
        engine_dla,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        dla_core=args.dla_core,
        cuda_graph=False,
        verbose=args.verbose,
    )
    gc.collect()

    LOG.info("Benchmarking AxoNN engine...")
    axonn_result = benchmark_engine(
        engine_axonn,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        cuda_graph=False,
        verbose=args.verbose,
    )
    gc.collect()

    # Step 6: Print results
    _print_results(
        model_name,
        imgsz,
        gpu_result,
        dla_result,
        axonn_result,
        axonn_build,
        iterations=args.iterations,
        energy_ratio=args.energy_ratio,
        max_transitions=args.max_transitions,
    )


def _print_results(
    model_name: str,
    imgsz: int,
    gpu_result: object,
    dla_result: object,
    axonn_result: object,
    axonn_build: tuple[float, float, int, int, int],
    *,
    iterations: int,
    energy_ratio: float,
    max_transitions: int,
) -> None:
    gpu_lat_ms = gpu_result.latency.mean * 1000  # type: ignore[union-attr]
    dla_lat_ms = dla_result.latency.mean * 1000  # type: ignore[union-attr]
    axonn_lat_ms = axonn_result.latency.mean * 1000  # type: ignore[union-attr]

    gpu_energy_mj = gpu_result.energy.mean  # type: ignore[union-attr]
    dla_energy_mj = dla_result.energy.mean  # type: ignore[union-attr]
    axonn_energy_mj = axonn_result.energy.mean  # type: ignore[union-attr]

    gpu_fps = 1.0 / gpu_result.latency.mean if gpu_result.latency.mean > 0 else 0.0  # type: ignore[union-attr]
    dla_fps = 1.0 / dla_result.latency.mean if dla_result.latency.mean > 0 else 0.0  # type: ignore[union-attr]
    axonn_fps = 1.0 / axonn_result.latency.mean if axonn_result.latency.mean > 0 else 0.0  # type: ignore[union-attr]

    gpu_power_w = gpu_result.power_draw.mean / 1000  # type: ignore[union-attr]
    dla_power_w = dla_result.power_draw.mean / 1000  # type: ignore[union-attr]
    axonn_power_w = axonn_result.power_draw.mean / 1000  # type: ignore[union-attr]

    _, _, axonn_transitions, axonn_gpu_layers, axonn_dla_layers = axonn_build

    def _pct(axonn_val: float, other_val: float) -> float:
        if other_val == 0:
            return 0.0
        return ((axonn_val - other_val) / other_val) * 100

    lat_vs_gpu = _pct(axonn_lat_ms, gpu_lat_ms)
    energy_vs_gpu = _pct(axonn_energy_mj, gpu_energy_mj)
    lat_vs_dla = _pct(axonn_lat_ms, dla_lat_ms)
    energy_vs_dla = _pct(axonn_energy_mj, dla_energy_mj)

    sep = "=" * 60
    dash = "-" * 60
    LOG.info(f"\n{sep}")
    LOG.info("AxoNN Benchmark Results")
    LOG.info(sep)
    LOG.info(f"Model: {model_name} ({imgsz}x{imgsz})")
    LOG.info(
        f"Iterations: {iterations} | "
        f"Energy ratio: {energy_ratio:.2f} | "
        f"Max transitions: {max_transitions}"
    )
    LOG.info(dash)
    LOG.info(f"{'Metric':<20s}{'GPU-only':>12s}{'DLA-only':>12s}{'AxoNN':>12s}")
    LOG.info(dash)
    LOG.info(f"{'Latency (ms)':<20s}{gpu_lat_ms:>12.3f}{dla_lat_ms:>12.3f}{axonn_lat_ms:>12.3f}")
    LOG.info(
        f"{'Energy (mJ)':<20s}{gpu_energy_mj:>12.3f}{dla_energy_mj:>12.3f}{axonn_energy_mj:>12.3f}"
    )
    LOG.info(f"{'Throughput (FPS)':<20s}{gpu_fps:>12.1f}{dla_fps:>12.1f}{axonn_fps:>12.1f}")
    LOG.info(f"{'Power (W)':<20s}{gpu_power_w:>12.2f}{dla_power_w:>12.2f}{axonn_power_w:>12.2f}")
    LOG.info(f"{'GPU Layers':<20s}{'all':>12s}{'--':>12s}{axonn_gpu_layers:>12d}")
    LOG.info(f"{'DLA Layers':<20s}{'0':>12s}{'--':>12s}{axonn_dla_layers:>12d}")
    LOG.info(f"{'Transitions':<20s}{'0':>12s}{'1':>12s}{axonn_transitions:>12d}")
    LOG.info(dash)
    LOG.info(f"AxoNN vs GPU:  {lat_vs_gpu:+.1f}% latency | {energy_vs_gpu:+.1f}% energy")
    LOG.info(f"AxoNN vs DLA:  {lat_vs_dla:+.1f}% latency | {energy_vs_dla:+.1f}% energy")
    LOG.info(sep)


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
    build_parser.add_argument(
        "--onnx",
        required=True,
        help="Path to the ONNX model file.",
    )
    build_parser.add_argument(
        "--output",
        required=True,
        help="Path to save the AxoNN-optimized engine.",
    )
    # AxoNN parameters
    build_parser.add_argument(
        "--energy_target",
        type=float,
        default=None,
        help="Energy Consumption Target (ECT) in millijoules. If not set, uses energy_ratio.",
    )
    build_parser.add_argument(
        "--energy_ratio",
        type=float,
        default=0.8,
        help="Ratio of GPU-only energy to use as ECT (0.0-1.0). Default is 0.8.",
    )
    build_parser.add_argument(
        "--max_transitions",
        type=int,
        default=3,
        help="Maximum GPU<->DLA transitions. Default is 3.",
    )
    build_parser.add_argument(
        "--dla_core",
        type=int,
        default=0,
        help="DLA core to use (0 or 1). Default is 0.",
    )
    # Profiling parameters
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
    # TRT build parameters
    build_parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="Workspace size in GB. Default is 4.0.",
    )
    build_parser.add_argument(
        "--timing_cache",
        default=None,
        help="Path to timing cache file.",
    )
    build_parser.add_argument(
        "--calibration_cache",
        default=None,
        help="Path to calibration cache file.",
    )
    build_parser.add_argument(
        "--optimization_level",
        type=int,
        choices=range(6),
        default=3,
        help="TensorRT builder optimization level (0-5). Default is 3.",
    )
    # Calibration data
    build_parser.add_argument(
        "--calibration_dir",
        default=None,
        help="Directory with calibration images. If not provided, uses synthetic data.",
    )
    build_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for calibration. Default is 8.",
    )
    build_parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of synthetic batches (when no calibration_dir). Default is 10.",
    )
    build_parser.add_argument(
        "--cuda_graph",
        action="store_true",
        help="Enable CUDA graph capture for GPU profiling.",
    )
    build_parser.set_defaults(func=_axonn_build)

    # -- benchmark subcommand --
    benchmark_parser = axonn_subparsers.add_parser(
        "benchmark",
        help="Full GPU vs DLA vs AxoNN comparison benchmark.",
        parents=parents,
    )
    benchmark_parser.add_argument(
        "--model",
        type=str,
        default="resnet34",
        help="Model name to benchmark (default: resnet34). Use trtutils download --list_models.",
    )
    benchmark_parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size. Default: model's default.",
    )
    benchmark_parser.add_argument(
        "--data_dir",
        type=str,
        default="axonn_benchmark_data",
        help="Directory for storing engines and ONNX files. Default is axonn_benchmark_data.",
    )
    benchmark_parser.add_argument(
        "--energy_ratio",
        type=float,
        default=0.8,
        help="AxoNN energy target ratio (0.0-1.0). Default is 0.8.",
    )
    benchmark_parser.add_argument(
        "--max_transitions",
        type=int,
        default=3,
        help="Max GPU/DLA transitions. Default is 3.",
    )
    benchmark_parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Benchmark iterations. Default is 1000.",
    )
    benchmark_parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=50,
        help="Warmup iterations. Default is 50.",
    )
    benchmark_parser.add_argument(
        "--profile_iterations",
        type=int,
        default=1000,
        help="AxoNN profiling iterations. Default is 1000.",
    )
    benchmark_parser.add_argument(
        "--dla_core",
        type=int,
        default=0,
        help="DLA core to use (0 or 1). Default is 0.",
    )
    benchmark_parser.add_argument(
        "--cuda_graph",
        action="store_true",
        help="Enable CUDA graph capture for GPU engine benchmarking.",
    )
    benchmark_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild all engines.",
    )
    benchmark_parser.set_defaults(func=_axonn_benchmark)
