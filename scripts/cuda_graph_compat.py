# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: T201
"""Test CUDA Graph compatibility for each concrete model class."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from trtutils.models import (
    DEIM,
    DFINE,
    RFDETR,
    DEIMv2,
    RTDETRv1,
    RTDETRv2,
    RTDETRv3,
    YOLO3,
    YOLO5,
    YOLO7,
    YOLO8,
    YOLO9,
    YOLO10,
    YOLO11,
    YOLO12,
    YOLO13,
    YOLOX,
)

REPO_DIR = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data" / "cuda_graph_compat"
IMAGE_PATH = REPO_DIR / "data" / "horse.jpg"

MODELS: list[tuple[type, str, int]] = [
    (YOLOX, "yoloxn", 640),
    (YOLO3, "yolov3tu", 640),
    (YOLO5, "yolov5nu", 640),
    (YOLO7, "yolov7t", 640),
    (YOLO8, "yolov8n", 640),
    (YOLO9, "yolov9t", 640),
    (YOLO10, "yolov10n", 640),
    (YOLO11, "yolov11n", 640),
    (YOLO12, "yolov12n", 640),
    (YOLO13, "yolov13n", 640),
    (RTDETRv1, "rtdetrv1_r18", 640),
    (RTDETRv2, "rtdetrv2_r18", 640),
    (RTDETRv3, "rtdetrv3_r18", 640),
    (DFINE, "dfine_n", 640),
    (DEIM, "deim_dfine_n", 640),
    (DEIMv2, "deimv2_atto", 320),
    (RFDETR, "rfdetr_n", 384),
]


def benchmark_model(
    cls: type,
    model_name: str,
    imgsz: int,
    warmup_iters: int,
    bench_iters: int,
    image: np.ndarray,
) -> tuple[str, float | None, float | None, float | str]:
    class_name = cls.__name__
    onnx_dir = DATA_DIR / class_name
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"{model_name}.onnx"
    engine_path = onnx_dir / f"{model_name}.engine"

    # download
    if not onnx_path.exists():
        print(f"  Downloading {model_name}...")
        cls.download(model_name, onnx_path, imgsz=imgsz, accept=True)

    # build
    if not engine_path.exists():
        print(f"  Building engine...")
        cls.build(onnx_path, engine_path, imgsz=imgsz)

    # baseline (no graph)
    print(f"  Benchmarking baseline (no CUDA Graph)...")
    detector = cls(
        engine_path,
        warmup=True,
        warmup_iterations=warmup_iters,
        cuda_graph=False,
        preprocessor="cuda",
        pagelocked_mem=True,
    )
    baseline_times: list[float] = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        detector.end2end(image)
        baseline_times.append(time.perf_counter() - t0)
    baseline_median = statistics.median(baseline_times)
    del detector

    # cuda graph
    print(f"  Benchmarking CUDA Graph...")
    try:
        detector = cls(
            engine_path,
            warmup=True,
            warmup_iterations=warmup_iters,
            cuda_graph=True,
            preprocessor="cuda",
            pagelocked_mem=True,
        )
        graph_times: list[float] = []
        for _ in range(bench_iters):
            t0 = time.perf_counter()
            detector.end2end(image)
            graph_times.append(time.perf_counter() - t0)
        graph_median = statistics.median(graph_times)
        del detector
        speedup = baseline_median / graph_median
        return class_name, baseline_median, graph_median, speedup
    except Exception as e:
        print(f"  CUDA Graph FAILED: {e}")
        return class_name, baseline_median, None, "False"


def main() -> None:
    parser = argparse.ArgumentParser("CUDA Graph compatibility test for all model classes.")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=200, help="Benchmark iterations.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(IMAGE_PATH))

    results: list[tuple[str, float | None, float | None, float | str]] = []

    for cls, model_name, imgsz in MODELS:
        class_name = cls.__name__
        print(f"\n{'='*60}")
        print(f"Testing {class_name} ({model_name}, imgsz={imgsz})")
        print(f"{'='*60}")
        try:
            result = benchmark_model(cls, model_name, imgsz, args.warmup, args.iterations, image)
            results.append(result)
        except Exception as e:
            print(f"  FAILED entirely: {e}")
            results.append((class_name, None, None, "Error"))

    # summary table
    print(f"\n\n{'='*70}")
    print("CUDA Graph Compatibility Summary")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Baseline (ms)':>15} {'Graph (ms)':>15} {'Speedup':>15}")
    print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    for name, baseline, graph, speedup in results:
        b_str = f"{baseline * 1000:.2f}" if baseline is not None else "N/A"
        g_str = f"{graph * 1000:.2f}" if graph is not None else "N/A"
        if isinstance(speedup, float):
            s_str = f"{speedup:.2f}x"
        else:
            s_str = str(speedup)
        print(f"{name:<15} {b_str:>15} {g_str:>15} {s_str:>15}")


if __name__ == "__main__":
    main()
