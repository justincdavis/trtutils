# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S603, S607, T201
"""Benchmark throughput scaling across batch sizes."""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import subprocess
import time
import warnings
from pathlib import Path

import shutil

import cv2
import nvtx
from tqdm import tqdm

from model_utils import build_model, ensure_model_available

# Paths
REPO_DIR = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data" / "batch"
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_PATH = str((REPO_DIR / "data" / "horse.jpg").resolve())

# Model info
MODEL_TO_DIR: dict[str, str] = {
    "yolov10n": "yolov10",
    "yolov10s": "yolov10",
    "yolov10m": "yolov10",
    "yolov11n": "yolov11",
    "yolov11s": "yolov11",
    "yolov11m": "yolov11",
    "yolov9t": "yolov9",
    "yolov9s": "yolov9",
    "yolov8n": "yolov8",
    "yolov8s": "yolov8",
}

ULTRALYTICS_MODELS = [
    "yolov11n",
    "yolov11s",
    "yolov11m",
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov9t",
    "yolov9s",
    "yolov8n",
    "yolov8s",
]

FRAMEWORKS = [
    "trtutils",
    "trtutils(graph)",
    "ultralytics(trt)",
    "ultralytics(torch)",
]


def _export_batch_onnx(
    model_name: str,
    imgsz: int,
    batch_size: int,
    output_dir: Path,
) -> Path:
    """Export an ONNX model with a specific batch size using yolo export.

    Uses the ultralytics CLI to natively export the model with the given batch
    size, so all internal reshape operations are computed correctly.
    Returns the path to the batch-specific ONNX file.
    """
    onnx_path = output_dir / f"{model_name}_{imgsz}_b{batch_size}.onnx"
    if onnx_path.exists():
        return onnx_path

    print(f"Exporting {model_name} ONNX with batch={batch_size}...")

    # ultralytics auto-downloads .pt weights
    pt_dir = REPO_DIR / "data" / "ultralytics"
    pt_dir.mkdir(parents=True, exist_ok=True)
    pt_path = pt_dir / f"{model_name}.pt"

    subprocess.run(
        [
            "yolo",
            "export",
            f"model={pt_path}",
            "format=onnx",
            f"imgsz={imgsz}",
            f"batch={batch_size}",
            "opset=17",
        ],
        check=True,
        capture_output=True,
    )

    # yolo export puts the .onnx next to the .pt file
    exported_path = pt_path.with_suffix(".onnx")
    if not exported_path.exists():
        err_msg = f"ONNX export failed, file not found: {exported_path}"
        raise FileNotFoundError(err_msg)

    shutil.move(str(exported_path), str(onnx_path))
    print(f"Exported batch ONNX: {onnx_path.name}")
    return onnx_path


def get_results(data: list[float], batch_size: int) -> dict[str, float]:
    """Compute timing stats and throughput from raw timings (seconds)."""
    mean_s = statistics.mean(data)
    std_s = statistics.stdev(data) if len(data) > 1 else 0.0
    ci95_s = 1.96 * (std_s / (len(data) ** 0.5)) if len(data) > 1 else 0.0
    mean_throughput = batch_size / mean_s if mean_s > 0 else 0.0
    return {
        "mean": mean_s * 1000.0,
        "std": std_s * 1000.0,
        "ci95": ci95_s * 1000.0,
        "median": statistics.median(data) * 1000.0,
        "min": min(data) * 1000.0,
        "max": max(data) * 1000.0,
        "throughput": mean_throughput,
    }


def get_data(device: str) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Load existing batch benchmark data for a device."""
    file_path = DATA_DIR / f"{device}.json"
    if file_path.exists():
        with file_path.open("r") as f:
            data = json.load(f)
            for fw in FRAMEWORKS:
                if data.get(fw) is None:
                    data[fw] = {}
            return data
    return {fw: {} for fw in FRAMEWORKS}


def write_data(
    device: str, data: dict[str, dict[str, dict[str, dict[str, float]]]]
) -> None:
    """Write batch benchmark data for a device."""
    file_path = DATA_DIR / f"{device}.json"
    with file_path.open("w") as f:
        json.dump(data, f, indent=4)


def _engine_path_for_batch(onnx_path: Path, imgsz: int, batch_size: int) -> Path:
    """Get engine path with batch size in the name."""
    model_name = onnx_path.stem.rsplit("_", 1)[0]
    return onnx_path.parent / f"{model_name}_{imgsz}_b{batch_size}.engine"


def benchmark_trtutils(
    device: str,
    model_name: str,
    imgsz: int,
    batch_sizes: list[int],
    warmup_iters: int,
    bench_iters: int,
    *,
    overwrite: bool,
) -> None:
    """Benchmark trtutils Detector at various batch sizes."""
    from trtutils.image import Detector

    image = cv2.imread(IMAGE_PATH)
    data = get_data(device)

    # Ensure batch=1 ONNX model available
    try:
        onnx_path = ensure_model_available(
            model_name, imgsz, MODEL_TO_DIR, auto_download=True,
        )
    except Exception as e:
        warnings.warn(f"Could not get model {model_name} @ {imgsz}: {e}")
        return

    # Directory where batch-specific ONNX files are stored
    onnx_dir = onnx_path.parent

    modes: list[tuple[str, bool]] = [
        ("trtutils", False),
        ("trtutils(graph)", True),
    ]

    for framework, cuda_graph in modes:
        if model_name not in data[framework]:
            data[framework][model_name] = {}

        for bs in batch_sizes:
            bs_key = str(bs)

            # Skip if already benchmarked
            with contextlib.suppress(KeyError):
                if data[framework][model_name][bs_key] is not None and not overwrite:
                    print(f"Skipping {framework} batch={bs} (already exists)")
                    continue

            print(f"Benchmarking {framework} batch={bs}...")

            # For batch=1, use existing ONNX; for batch>1, re-export natively
            if bs == 1:
                batch_onnx = onnx_path
            else:
                batch_onnx = _export_batch_onnx(model_name, imgsz, bs, onnx_dir)

            engine_path = _engine_path_for_batch(onnx_path, imgsz, bs)
            if not engine_path.exists():
                print(f"\tBuilding engine for batch={bs}...")
                build_model(
                    onnx=batch_onnx,
                    output=engine_path,
                    imgsz=imgsz,
                    batch_size=bs,
                    model_name=model_name,
                )

            try:
                detector = Detector(
                    engine_path=engine_path,
                    warmup_iterations=warmup_iters,
                    warmup=True,
                    preprocessor="cuda",
                    pagelocked_mem=True,
                    cuda_graph=cuda_graph,
                    verbose=False,
                )

                # Create a batch of images
                images = [image] * bs

                timings: list[float] = []
                with nvtx.annotate(f"{framework} batch={bs}", color="blue"):
                    for _ in tqdm(range(bench_iters), desc=f"{framework} b={bs}"):
                        t0 = time.perf_counter()
                        detector.end2end(images)
                        timings.append(time.perf_counter() - t0)

                del detector

                results = get_results(timings, bs)
                data[framework][model_name][bs_key] = results
                write_data(device, data)

                print(
                    f"\t{framework} batch={bs}: "
                    f"{results['mean']:.2f}ms ±{results['ci95']:.2f}, "
                    f"{results['throughput']:.1f} img/s"
                )
            except Exception as e:
                warnings.warn(f"Failed {framework} batch={bs}: {e}")
                continue


def benchmark_ultralytics(
    device: str,
    model_name: str,
    imgsz: int,
    batch_sizes: list[int],
    warmup_iters: int,
    bench_iters: int,
    *,
    overwrite: bool,
) -> None:
    """Benchmark ultralytics YOLO at various batch sizes."""
    from ultralytics import YOLO

    image = cv2.imread(IMAGE_PATH)
    data = get_data(device)

    # Ultralytics model paths
    ultralytics_dir = REPO_DIR / "data" / "ultralytics"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    pt_path = (ultralytics_dir / f"{model_name}.pt").resolve()

    # --- ultralytics(torch) ---
    framework = "ultralytics(torch)"
    if model_name not in data[framework]:
        data[framework][model_name] = {}

    for bs in batch_sizes:
        bs_key = str(bs)

        with contextlib.suppress(KeyError):
            if data[framework][model_name][bs_key] is not None and not overwrite:
                print(f"Skipping {framework} batch={bs} (already exists)")
                continue

        print(f"Benchmarking {framework} batch={bs}...")

        try:
            yolo = YOLO(model=pt_path, task="detect", verbose=False)

            images = [image] * bs

            for _ in range(warmup_iters):
                yolo(images, imgsz=imgsz, verbose=False)

            timings: list[float] = []
            with nvtx.annotate(f"{framework} batch={bs}", color="green"):
                for _ in tqdm(range(bench_iters), desc=f"{framework} b={bs}"):
                    t0 = time.perf_counter()
                    yolo(images, imgsz=imgsz, verbose=False)
                    timings.append(time.perf_counter() - t0)

            del yolo

            results = get_results(timings, bs)
            data[framework][model_name][bs_key] = results
            write_data(device, data)

            print(
                f"\t{framework} batch={bs}: "
                f"{results['mean']:.2f}ms ±{results['ci95']:.2f}, "
                f"{results['throughput']:.1f} img/s"
            )
        except Exception as e:
            warnings.warn(f"Failed {framework} batch={bs}: {e}")
            continue

    # --- ultralytics(trt) ---
    framework = "ultralytics(trt)"
    if model_name not in data[framework]:
        data[framework][model_name] = {}

    for bs in batch_sizes:
        bs_key = str(bs)

        with contextlib.suppress(KeyError):
            if data[framework][model_name][bs_key] is not None and not overwrite:
                print(f"Skipping {framework} batch={bs} (already exists)")
                continue

        print(f"Benchmarking {framework} batch={bs}...")

        # Build ultralytics TRT engine for this batch size
        engine_path = ultralytics_dir / f"{model_name}_{imgsz}_b{bs}.engine"
        base_engine_path = pt_path.with_suffix(".engine")

        if not engine_path.exists():
            print(f"\tBuilding ultralytics TRT engine for batch={bs}...")
            subprocess.run(
                [
                    "yolo",
                    "export",
                    f"model={pt_path}",
                    "format=engine",
                    f"imgsz={imgsz}",
                    f"batch={bs}",
                    "half",
                ],
                check=True,
                capture_output=True,
            )

            if not base_engine_path.exists():
                err_msg = f"Ultralytics TRT engine not found: {base_engine_path}"
                raise FileNotFoundError(err_msg)

            base_engine_path.rename(engine_path)

        try:
            yolo = YOLO(model=engine_path, task="detect", verbose=False)

            images = [image] * bs

            for _ in range(warmup_iters):
                yolo(images, imgsz=imgsz, verbose=False)

            timings = []
            with nvtx.annotate(f"{framework} batch={bs}", color="orange"):
                for _ in tqdm(range(bench_iters), desc=f"{framework} b={bs}"):
                    t0 = time.perf_counter()
                    yolo(images, imgsz=imgsz, verbose=False)
                    timings.append(time.perf_counter() - t0)

            del yolo

            results = get_results(timings, bs)
            data[framework][model_name][bs_key] = results
            write_data(device, data)

            print(
                f"\t{framework} batch={bs}: "
                f"{results['mean']:.2f}ms ±{results['ci95']:.2f}, "
                f"{results['throughput']:.1f} img/s"
            )
        except Exception as e:
            warnings.warn(f"Failed {framework} batch={bs}: {e}")
            continue


def main() -> None:
    """Run batch size throughput benchmarks."""
    parser = argparse.ArgumentParser("Benchmark throughput scaling across batch sizes.")
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device name (e.g. 5080). Results go to data/batch/{device}.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov10n",
        help="Model to benchmark.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="Batch sizes to test.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--trtutils",
        action="store_true",
        help="Run trtutils benchmarks.",
    )
    parser.add_argument(
        "--ultralytics",
        action="store_true",
        help="Run ultralytics benchmarks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data.",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Enable NVTX profiling annotations.",
    )
    args = parser.parse_args()

    if args.nvtx:
        from trtutils import enable_nvtx
        enable_nvtx()

    if args.model not in MODEL_TO_DIR:
        err_msg = f"Unknown model: {args.model}. Supported: {list(MODEL_TO_DIR.keys())}"
        raise ValueError(err_msg)

    print(f"Batch benchmark: {args.model} @ {args.imgsz}x{args.imgsz}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}\n")

    if args.trtutils:
        benchmark_trtutils(
            args.device,
            args.model,
            args.imgsz,
            args.batch_sizes,
            args.warmup,
            args.iterations,
            overwrite=args.overwrite,
        )

    if args.ultralytics:
        if args.model not in ULTRALYTICS_MODELS:
            warnings.warn(
                f"{args.model} is not a supported ultralytics model, skipping"
            )
        else:
            benchmark_ultralytics(
                args.device,
                args.model,
                args.imgsz,
                args.batch_sizes,
                args.warmup,
                args.iterations,
                overwrite=args.overwrite,
            )

    # Print summary
    data = get_data(args.device)
    print("\n" + "=" * 70)
    print(f"Summary: {args.model} @ {args.imgsz}")
    print(
        f"{'Framework':<22} {'Batch':<8} {'Mean(ms)':<10} {'CI95(ms)':<10} {'Throughput(img/s)':<18}",
    )
    print("=" * 70)

    for fw in FRAMEWORKS:
        model_data = data.get(fw, {}).get(args.model, {})
        for bs in args.batch_sizes:
            entry = model_data.get(str(bs))
            if entry:
                print(
                    f"{fw:<22} {bs:<8} "
                    f"{entry['mean']:<10.2f} {entry.get('ci95', 0.0):<10.2f} {entry['throughput']:<18.1f}"
                )
    print("=" * 70)


if __name__ == "__main__":
    main()
