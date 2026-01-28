# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: T201
"""Benchmark different optimization combinations for Detector."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from model_utils import build_model, ensure_model_available

# Paths
REPO_DIR = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data"
IMAGE_PATH = REPO_DIR / "data" / "horse.jpg"

# Model config
MODEL_TO_DIR = {
    "yolov10n": "yolov10",
    "yolov10s": "yolov10",
    "yolov10m": "yolov10",
}


def get_detector(
    engine_path: Path,
    *,
    preprocessor: str,
    cuda_graph: bool,
    pagelocked_mem: bool,
    unified_mem: bool,
    warmup_iters: int,
):
    """Create a Detector with specified options."""
    from trtutils.image import Detector  # noqa: PLC0415

    return Detector(
        engine_path=engine_path,
        warmup=True,
        warmup_iterations=warmup_iters,
        preprocessor=preprocessor,
        cuda_graph=cuda_graph,
        pagelocked_mem=pagelocked_mem,
        unified_mem=unified_mem,
        verbose=False,
    )


def benchmark_config(
    engine_path: Path,
    image: np.ndarray,
    *,
    preprocessor: str,
    cuda_graph: bool,
    pagelocked_mem: bool,
    unified_mem: bool,
    warmup_iters: int,
    bench_iters: int,
) -> dict[str, float]:
    """Benchmark a single configuration and return timing stats."""
    detector = get_detector(
        engine_path,
        preprocessor=preprocessor,
        cuda_graph=cuda_graph,
        pagelocked_mem=pagelocked_mem,
        unified_mem=unified_mem,
        warmup_iters=warmup_iters,
    )

    timings = []
    for _ in tqdm(range(bench_iters), desc=f"prep={preprocessor} graph={cuda_graph} pl={pagelocked_mem} um={unified_mem}"):
        t0 = time.perf_counter()
        detector.end2end(image)
        t1 = time.perf_counter()
        timings.append(t1 - t0)

    del detector

    return {
        "mean_ms": statistics.mean(timings) * 1000,
        "median_ms": statistics.median(timings) * 1000,
        "std_ms": statistics.stdev(timings) * 1000 if len(timings) > 1 else 0.0,
        "min_ms": min(timings) * 1000,
        "max_ms": max(timings) * 1000,
    }


def main() -> None:
    """Run optimization benchmarks."""
    parser = argparse.ArgumentParser("Benchmark Detector optimization options.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov10n",
        choices=list(MODEL_TO_DIR.keys()),
        help="Model to benchmark.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of benchmark iterations.",
    )
    args = parser.parse_args()

    # Ensure image exists
    if not IMAGE_PATH.exists():
        err_msg = f"Image not found: {IMAGE_PATH}"
        raise FileNotFoundError(err_msg)

    image = cv2.imread(str(IMAGE_PATH))

    # Ensure model exists
    model_path = ensure_model_available(
        args.model, args.imgsz, MODEL_TO_DIR, auto_download=True
    )
    engine_path = model_path.with_suffix(".engine")

    if not engine_path.exists():
        print(f"Building engine: {engine_path}")
        build_model(model_path, engine_path, args.imgsz, opt_level=1)

    preprocessors = ["cpu", "cuda", "trt"]

    # Define configurations to test
    configs = []
    for prep in preprocessors:
        for cuda_graph in [False, True]:
            for pagelocked in [False, True]:
                for unified in [False, True]:
                    if pagelocked and unified:
                        continue
                    configs.append({
                        "preprocessor": prep,
                        "cuda_graph": cuda_graph,
                        "pagelocked_mem": pagelocked,
                        "unified_mem": unified,
                    })

    # Run benchmarks
    results = []
    print(f"\nBenchmarking {args.model} @ {args.imgsz}x{args.imgsz}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}\n")

    for cfg in configs:
        try:
            stats = benchmark_config(
                engine_path,
                image,
                preprocessor=cfg["preprocessor"],
                cuda_graph=cfg["cuda_graph"],
                pagelocked_mem=cfg["pagelocked_mem"],
                unified_mem=cfg["unified_mem"],
                warmup_iters=args.warmup,
                bench_iters=args.iterations,
            )
            results.append({**cfg, **stats})
        except Exception as e:
            print(f"FAILED: {cfg} - {e}")
            continue

    # Print results table
    print("\n" + "=" * 80)
    print(f"{'Preprocessor':<12} {'CUDAGraph':<10} {'Pagelocked':<10} {'Unified':<10} {'Mean(ms)':<10} {'Std(ms)':<10} {'Min(ms)':<10}")
    print("=" * 80)

    # Sort by mean time
    results.sort(key=lambda x: x["mean_ms"])

    for r in results:
        print(
            f"{r['preprocessor']:<12} "
            f"{str(r['cuda_graph']):<10} "
            f"{str(r['pagelocked_mem']):<10} "
            f"{str(r['unified_mem']):<10} "
            f"{r['mean_ms']:<10.3f} "
            f"{r['std_ms']:<10.3f} "
            f"{r['min_ms']:<10.3f}"
        )

    # Print speedup summary
    if len(results) >= 2:
        slowest = results[-1]["mean_ms"]
        fastest = results[0]["mean_ms"]
        speedup = slowest / fastest if fastest > 0 else 0
        print("=" * 80)
        print(f"Fastest: {results[0]['preprocessor']}, graph={results[0]['cuda_graph']}, pl={results[0]['pagelocked_mem']}, um={results[0]['unified_mem']}")
        print(f"Max speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
