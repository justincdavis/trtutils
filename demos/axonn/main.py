# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""AxoNN benchmark comparing GPU-only, DLA-only, and AxoNN-optimized engines."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx

from trtutils import FLAGS, build_engine
from trtutils.builder import SyntheticBatcher
from trtutils.compat._libs import trt
from trtutils.download import download, get_supported_models
from trtutils.jetson import benchmark_engine
from trtutils.research.axonn import build_engine as axonn_build_engine

if TYPE_CHECKING:
    from trtutils.jetson import JetsonBenchmarkResult

_DATA_DIR = Path(__file__).parent / "data"


def _get_input_info(onnx_path: Path) -> tuple[str, tuple[int, ...]]:
    model = onnx.load(str(onnx_path))
    inp = model.graph.input[0]
    name = inp.name
    dims = tuple(d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim)
    return name, dims


def _get_batcher(imgsz: int) -> SyntheticBatcher:
    return SyntheticBatcher(
        shape=(imgsz, imgsz, 3),
        dtype=np.float32,
        batch_size=8,
        num_batches=10,
        data_range=(0.0, 1.0),
        order="NCHW",
    )


def _print_results(
    model_name: str,
    imgsz: int,
    gpu_result: JetsonBenchmarkResult,
    dla_result: JetsonBenchmarkResult,
    axonn_result: JetsonBenchmarkResult,
    axonn_build: tuple[float, float, int, int, int],
    *,
    iterations: int,
    energy_ratio: float,
    max_transitions: int,
) -> None:
    # Units: latency in seconds, power_draw in milliwatts, energy in mW*s (=mJ)
    gpu_lat_ms = gpu_result.latency.mean * 1000
    axonn_lat_ms = axonn_result.latency.mean * 1000

    gpu_energy_mj = gpu_result.energy.mean
    axonn_energy_mj = axonn_result.energy.mean

    gpu_fps = 1.0 / gpu_result.latency.mean if gpu_result.latency.mean > 0 else 0.0
    axonn_fps = 1.0 / axonn_result.latency.mean if axonn_result.latency.mean > 0 else 0.0

    gpu_power_w = gpu_result.power_draw.mean / 1000
    axonn_power_w = axonn_result.power_draw.mean / 1000

    _, _, axonn_transitions, axonn_gpu_layers, axonn_dla_layers = axonn_build

    has_dla = dla_result is not None
    if has_dla:
        dla_lat_ms = dla_result.latency.mean * 1000
        dla_energy_mj = dla_result.energy.mean
        dla_fps = 1.0 / dla_result.latency.mean if dla_result.latency.mean > 0 else 0.0
        dla_power_w = dla_result.power_draw.mean / 1000
    else:
        dla_lat_ms = dla_energy_mj = dla_fps = dla_power_w = 0.0

    # Comparison percentages
    def _pct(axonn_val: float, other_val: float) -> float:
        if other_val == 0:
            return 0.0
        return ((axonn_val - other_val) / other_val) * 100

    lat_vs_gpu = _pct(axonn_lat_ms, gpu_lat_ms)
    energy_vs_gpu = _pct(axonn_energy_mj, gpu_energy_mj)

    dla_col = f"{'DLA-only':>12s}" if has_dla else f"{'N/A':>12s}"
    sep = "=" * 60
    dash = "-" * 60
    print(f"\n{sep}")
    print("AxoNN Benchmark Results")
    print(sep)
    print(f"Model: {model_name} ({imgsz}x{imgsz})")
    print(
        f"Iterations: {iterations} | "
        f"Energy ratio: {energy_ratio:.2f} | "
        f"Max transitions: {max_transitions}"
    )
    print(dash)
    print(f"{'Metric':<20s}{'GPU-only':>12s}{dla_col}{'AxoNN':>12s}")
    print(dash)
    dla_lat_s = f"{dla_lat_ms:>12.3f}" if has_dla else f"{'--':>12s}"
    dla_energy_s = f"{dla_energy_mj:>12.3f}" if has_dla else f"{'--':>12s}"
    dla_fps_s = f"{dla_fps:>12.1f}" if has_dla else f"{'--':>12s}"
    dla_pow_s = f"{dla_power_w:>12.2f}" if has_dla else f"{'--':>12s}"
    print(f"{'Latency (ms)':<20s}{gpu_lat_ms:>12.3f}{dla_lat_s}{axonn_lat_ms:>12.3f}")
    print(f"{'Energy (mJ)':<20s}{gpu_energy_mj:>12.3f}{dla_energy_s}{axonn_energy_mj:>12.3f}")
    print(f"{'Throughput (FPS)':<20s}{gpu_fps:>12.1f}{dla_fps_s}{axonn_fps:>12.1f}")
    print(f"{'Power (W)':<20s}{gpu_power_w:>12.2f}{dla_pow_s}{axonn_power_w:>12.2f}")
    print(f"{'GPU Layers':<20s}{'all':>12s}{'--':>12s}{axonn_gpu_layers:>12d}")
    print(f"{'DLA Layers':<20s}{'0':>12s}{'--':>12s}{axonn_dla_layers:>12d}")
    print(f"{'Transitions':<20s}{'0':>12s}{'--':>12s}{axonn_transitions:>12d}")
    print(dash)
    print(f"AxoNN vs GPU:  {lat_vs_gpu:+.1f}% latency | {energy_vs_gpu:+.1f}% energy")
    if has_dla:
        lat_vs_dla = _pct(axonn_lat_ms, dla_lat_ms)
        energy_vs_dla = _pct(axonn_energy_mj, dla_energy_mj)
        print(f"AxoNN vs DLA:  {lat_vs_dla:+.1f}% latency | {energy_vs_dla:+.1f}% energy")
    print(sep)


def _main() -> None:
    supported = get_supported_models()
    parser = argparse.ArgumentParser(
        description="AxoNN benchmark: GPU vs DLA vs AxoNN-optimized.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet34",
        help=f"Model name to benchmark (default: resnet34). Supported: {', '.join(sorted(supported))}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size (default: model's default, e.g. 224 for classifiers, 640 for detectors)",
    )
    parser.add_argument(
        "--energy-ratio",
        type=float,
        default=0.8,
        help="AxoNN energy target ratio (default: 0.8)",
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=3,
        help="Max GPU/DLA transitions (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Benchmark iterations (default: 1000)",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=50,
        help="Warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=1000,
        help="AxoNN profiling iterations (default: 1000)",
    )
    parser.add_argument(
        "--dla-core",
        type=int,
        default=0,
        help="DLA core to use, 0 or 1 (default: 0)",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA graph capture for GPU engine benchmarking",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild all engines",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    if not FLAGS.IS_JETSON:
        sys.stderr.write("This demo is only supported on Jetson systems.\n")
        sys.exit(1)

    model_name: str = args.model
    if model_name not in supported:
        parser.error(f"Unknown model '{model_name}'. Supported: {', '.join(sorted(supported))}")

    data_dir = _DATA_DIR / model_name
    data_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = data_dir / f"{model_name}.onnx"
    engine_gpu = data_dir / f"{model_name}_gpu.engine"
    engine_dla = data_dir / f"{model_name}_dla.engine"
    engine_axonn = data_dir / f"{model_name}_axonn.engine"

    # download the ONNX model
    download(
        model=model_name,
        output=onnx_path,
        imgsz=args.imgsz,
        simplify=True,
        make_static=True,
        verbose=args.verbose,
    )
    input_name, input_dims = _get_input_info(onnx_path)
    imgsz = input_dims[-1]
    shapes = [(input_name, input_dims)]

    # build the GPU engine
    if not engine_gpu.exists() or args.rebuild:
        build_engine(
            onnx=onnx_path,
            output=engine_gpu,
            fp16=True,
            shapes=shapes,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            verbose=args.verbose,
        )

    # build the DLA engine
    if not engine_dla.exists() or args.rebuild:
        build_engine(
            onnx=onnx_path,
            output=engine_dla,
            default_device=trt.DeviceType.DLA,
            gpu_fallback=True,
            fp16=True,
            int8=True,
            data_batcher=_get_batcher(imgsz),
            dla_core=args.dla_core,
            shapes=shapes,
            profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
            verbose=args.verbose,
        )

    # build the AxoNN engine
    if engine_axonn.exists() and not args.rebuild:
        axonn_build = (0.0, 0.0, 0, 0, 0)
    else:
        axonn_build = axonn_build_engine(
            onnx=onnx_path,
            output=engine_axonn,
            calibration_batcher=_get_batcher(imgsz),
            energy_ratio=args.energy_ratio,
            max_transitions=args.max_transitions,
            profile_iterations=args.profile_iterations,
            warmup_iterations=args.warmup_iterations,
            dla_core=args.dla_core,
            shapes=shapes,
            verbose=args.verbose,
        )

    # benchmark each engine
    results = {}
    for engine, tag in [(engine_gpu, "gpu"), (engine_dla, "dla"), (engine_axonn, "axonn")]:
        use_cuda_graph = bool(args.cuda_graph) if tag == "gpu" else False
        results[tag] = benchmark_engine(
            engine,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            cuda_graph=use_cuda_graph,
            verbose=args.verbose,
        )
        gc.collect()

    # print the results
    _print_results(
        model_name,
        imgsz,
        results["gpu"],
        results["dla"],
        results["axonn"],
        axonn_build,
        iterations=args.iterations,
        energy_ratio=args.energy_ratio,
        max_transitions=args.max_transitions,
    )


if __name__ == "__main__":
    _main()
