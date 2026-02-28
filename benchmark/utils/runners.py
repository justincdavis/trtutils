# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S603, S607, T201
"""Benchmark runner functions for model, batch, SAHI, and optimization benchmarks."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import cv2
from tqdm import tqdm

from .config import (
    BATCH_FRAMEWORKS,
    DATA_DIR,
    IMAGE_PATH,
    MODEL_FRAMEWORKS,
    MODEL_TO_DIR,
    MODEL_TO_IMGSIZES,
    REPO_DIR,
    SAHI_IMAGE_PATH,
)
from .data import get_data, write_data
from .models import build_model, ensure_model_available
from .timing import benchmark_loop, compute_results


# ---------------------------------------------------------------------------
# Model comparison benchmarks (from run.py)
# ---------------------------------------------------------------------------


def benchmark_trtutils_models(
    device: str,
    model_name: str,
    image_sizes: list[int],
    warmup_iters: int,
    bench_iters: int,
    *,
    overwrite: bool = False,
) -> None:
    """Benchmark trtutils Detector and raw TensorRT inference across image sizes."""
    from trtutils import TRTEngine
    from trtutils.image import Detector

    image = cv2.imread(IMAGE_PATH)
    data = get_data(device, "models", MODEL_FRAMEWORKS)

    modes: list[tuple[str, bool]] = [
        ("trtutils", False),
        ("trtutils(graph)", True),
        ("tensorrt", False),
        ("tensorrt(graph)", True),
    ]

    for framework, cuda_graph in modes:
        for imgsz in image_sizes:
            with contextlib.suppress(KeyError):
                if data[framework][model_name][str(imgsz)] is not None and not overwrite:
                    continue

            try:
                onnx_path = ensure_model_available(model_name, imgsz, MODEL_TO_DIR)
            except Exception as e:
                warnings.warn(f"Could not get {model_name} @ {imgsz}: {e}")
                continue

            engine_path = onnx_path.with_suffix(".engine")
            if not engine_path.exists():
                print(f"\tBuilding engine...")
                build_model(onnx=onnx_path, output=engine_path, imgsz=imgsz)
                if not engine_path.exists():
                    err_msg = f"Engine build failed: {engine_path}"
                    raise FileNotFoundError(err_msg)

            print(f"Benchmarking {framework} {model_name} @ {imgsz}...")

            if framework.startswith("trtutils"):
                detector = Detector(
                    engine_path=engine_path,
                    warmup_iterations=warmup_iters,
                    warmup=True,
                    preprocessor="cuda",
                    pagelocked_mem=True,
                    cuda_graph=cuda_graph,
                    verbose=True,
                )
                timings = benchmark_loop(
                    lambda: detector.end2end(image),
                    warmup_iters=0,
                    bench_iters=bench_iters,
                    desc=f"{framework} {imgsz}",
                )
                del detector
            else:
                engine = TRTEngine(
                    engine_path=engine_path,
                    warmup_iterations=warmup_iters,
                    warmup=True,
                    cuda_graph=cuda_graph,
                    verbose=True,
                )
                input_ptrs = [b.allocation for b in engine.input_bindings]
                if cuda_graph:
                    exec_fn = lambda: engine.graph_exec(debug=True)
                else:
                    exec_fn = lambda: engine.raw_exec(
                        input_ptrs, debug=True, no_warn=True,
                    )
                timings = benchmark_loop(
                    exec_fn, 0, bench_iters, f"{framework} {imgsz}",
                )
                del engine

            results = compute_results(timings)
            data[framework].setdefault(model_name, {})[str(imgsz)] = results
            write_data(device, "models", data)


def benchmark_ultralytics_models(
    device: str,
    model_name: str,
    image_sizes: list[int],
    warmup_iters: int,
    bench_iters: int,
    *,
    overwrite: bool = False,
) -> None:
    """Benchmark ultralytics YOLO (torch and TRT) across image sizes."""
    from ultralytics import YOLO

    image = cv2.imread(IMAGE_PATH)
    data = get_data(device, "models", MODEL_FRAMEWORKS)

    ultralytics_dir = REPO_DIR / "data" / "ultralytics"
    pt_path = (ultralytics_dir / f"{model_name}.pt").resolve()

    methods: list[tuple[str, Path, bool]] = [
        ("ultralytics(torch)", pt_path, False),
        ("ultralytics(trt)", pt_path, True),
    ]

    for framework, weight_path, compile_engine in methods:
        for imgsz in image_sizes:
            with contextlib.suppress(KeyError):
                if data[framework][model_name][str(imgsz)] is not None and not overwrite:
                    continue

            if compile_engine:
                engine_path = ultralytics_dir / f"{model_name}_{imgsz}.engine"
                if not engine_path.exists():
                    print(f"\tBuilding ultralytics TRT engine for {imgsz}...")
                    base_engine = pt_path.with_suffix(".engine")
                    subprocess.run(
                        [
                            "yolo", "export", f"model={pt_path}",
                            "format=engine", f"imgsz={imgsz}", "half",
                        ],
                        check=True,
                        capture_output=True,
                    )
                    if not base_engine.exists():
                        err_msg = f"Ultralytics TRT engine not found: {base_engine}"
                        raise FileNotFoundError(err_msg)
                    base_engine.rename(engine_path)
                model_path = engine_path
            else:
                model_path = weight_path

            print(f"Benchmarking {framework} {model_name} @ {imgsz}...")
            yolo = YOLO(model=model_path, task="detect", verbose=False)
            timings = benchmark_loop(
                lambda: yolo(image, imgsz=imgsz, verbose=False),
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
                desc=f"{framework} {imgsz}",
            )
            del yolo

            results = compute_results(timings)
            data[framework].setdefault(model_name, {})[str(imgsz)] = results
            write_data(device, "models", data)


# ---------------------------------------------------------------------------
# SAHI comparison benchmarks (from run.py)
# ---------------------------------------------------------------------------


def benchmark_sahi(
    device: str,
    model_name: str,
    warmup_iters: int,
    bench_iters: int,
    *,
    overwrite: bool = False,
) -> None:
    """Benchmark trtutils SAHI against the official SAHI package."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from trtutils.compat.sahi import TRTDetectionModel
    from trtutils.image import SAHI, Detector

    from .sahi_compat import UltralyticsTRTDetector

    image = cv2.imread(SAHI_IMAGE_PATH)
    imgsz = 640
    overlap = 0.2
    conf_thres = 0.25

    data = get_data(device, "models", MODEL_FRAMEWORKS)
    sahi_key = f"sahi_{model_name}"
    if sahi_key not in data:
        data[sahi_key] = {}

    # Resolve model paths
    try:
        trt_weight_path = ensure_model_available(model_name, imgsz, MODEL_TO_DIR)
    except Exception as e:
        err_msg = f"Could not get {model_name} @ {imgsz}: {e}"
        raise FileNotFoundError(err_msg) from e

    trt_path = trt_weight_path.with_suffix(".engine")
    if not trt_path.exists():
        err_msg = f"Engine not found: {trt_path}. Run model benchmark first."
        raise FileNotFoundError(err_msg)

    ultralytics_dir = REPO_DIR / "data" / "ultralytics"
    pt_path = (ultralytics_dir / f"{model_name}.pt").resolve()
    utrt_path = ultralytics_dir / f"{model_name}_{imgsz}.engine"
    if not utrt_path.exists():
        err_msg = (
            f"Ultralytics engine not found: {utrt_path}. "
            "Run ultralytics benchmark first."
        )
        raise FileNotFoundError(err_msg)

    # --- trtutils SAHI ---
    if "trtutils" not in data[sahi_key] or overwrite:
        print("\tBenchmarking trtutils SAHI...")
        detector = Detector(
            trt_path,
            warmup=True,
            warmup_iterations=warmup_iters,
            preprocessor="trt",
            verbose=False,
        )
        sahi_obj = SAHI(
            detector,
            slice_size=(imgsz, imgsz),
            slice_overlap=(overlap, overlap),
            verbose=False,
        )

        timings: list[float] = []
        detection_counts: list[int] = []
        for _ in tqdm(range(bench_iters)):
            t0 = time.perf_counter()
            detections = sahi_obj.end2end(
                image, conf_thres=conf_thres, verbose=False,
            )
            timings.append(time.perf_counter() - t0)
            detection_counts.append(len(detections))

        del sahi_obj, detector

        results = compute_results(timings)
        avg_detections = sum(detection_counts) // len(detection_counts)
        data[sahi_key]["trtutils"] = {
            "timing": results,
            "detections": avg_detections,
        }
        print(
            f"\t\tTRTUtils SAHI: {results['mean']:.2f}ms, "
            f"{avg_detections} detections",
        )
        write_data(device, "models", data)

    # --- Official SAHI backends ---
    sahi_backends = [
        ("sahi(ultralytics)(torch)", "ultralytics_torch"),
        ("sahi(ultralytics)(trt)", "ultralytics_trt"),
        ("sahi(trtutils)", "trtutils_sahi"),
    ]

    for sahi_type_key, backend_tag in sahi_backends:
        if sahi_type_key in data[sahi_key] and not overwrite:
            continue

        print(f"\tBenchmarking {sahi_type_key}...")

        if backend_tag == "ultralytics_torch":
            detection_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(pt_path),
                confidence_threshold=conf_thres,
                device="cuda",
            )
        elif backend_tag == "ultralytics_trt":
            detection_model = UltralyticsTRTDetector(
                model_path=str(utrt_path),
                confidence_threshold=conf_thres,
                device="cuda",
            )
        else:
            detection_model = TRTDetectionModel(
                model_path=str(trt_path),
                confidence_threshold=conf_thres,
            )

        for _ in range(warmup_iters):
            get_sliced_prediction(
                SAHI_IMAGE_PATH,
                detection_model,
                slice_height=imgsz,
                slice_width=imgsz,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                verbose=0,
            )

        timings = []
        detection_counts = []
        for _ in tqdm(range(bench_iters)):
            t0 = time.perf_counter()
            result = get_sliced_prediction(
                SAHI_IMAGE_PATH,
                detection_model,
                slice_height=imgsz,
                slice_width=imgsz,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                verbose=0,
            )
            timings.append(time.perf_counter() - t0)
            detection_counts.append(len(result.object_prediction_list))

        results = compute_results(timings)
        avg_detections = sum(detection_counts) // len(detection_counts)
        data[sahi_key][sahi_type_key] = {
            "timing": results,
            "detections": avg_detections,
        }
        print(
            f"\t\t{sahi_type_key}: {results['mean']:.2f}ms, "
            f"{avg_detections} detections",
        )
        write_data(device, "models", data)


# ---------------------------------------------------------------------------
# Batch throughput benchmarks (from batch.py)
# ---------------------------------------------------------------------------


def _export_batch_onnx(
    model_name: str,
    imgsz: int,
    batch_size: int,
    output_dir: Path,
) -> Path:
    """Export an ONNX model with a specific batch size using the yolo CLI."""
    onnx_path = output_dir / f"{model_name}_{imgsz}_b{batch_size}.onnx"
    if onnx_path.exists():
        return onnx_path

    print(f"Exporting {model_name} ONNX with batch={batch_size}...")

    pt_dir = REPO_DIR / "data" / "ultralytics"
    pt_dir.mkdir(parents=True, exist_ok=True)
    pt_path = pt_dir / f"{model_name}.pt"

    subprocess.run(
        [
            "yolo", "export", f"model={pt_path}",
            "format=onnx", f"imgsz={imgsz}",
            f"batch={batch_size}", "opset=17",
        ],
        check=True,
        capture_output=True,
    )

    exported_path = pt_path.with_suffix(".onnx")
    if not exported_path.exists():
        err_msg = f"ONNX export failed, file not found: {exported_path}"
        raise FileNotFoundError(err_msg)

    shutil.move(str(exported_path), str(onnx_path))
    print(f"Exported batch ONNX: {onnx_path.name}")
    return onnx_path


def _engine_path_for_batch(
    onnx_path: Path,
    imgsz: int,
    batch_size: int,
) -> Path:
    """Get the engine path for a batch-specific model."""
    model_name = onnx_path.stem.rsplit("_", 1)[0]
    return onnx_path.parent / f"{model_name}_{imgsz}_b{batch_size}.engine"


def benchmark_trtutils_batch(
    device: str,
    model_name: str,
    imgsz: int,
    batch_sizes: list[int],
    warmup_iters: int,
    bench_iters: int,
    *,
    overwrite: bool = False,
) -> None:
    """Benchmark trtutils Detector throughput across batch sizes."""
    import nvtx
    from trtutils.image import Detector

    image = cv2.imread(IMAGE_PATH)
    data = get_data(device, "batch", BATCH_FRAMEWORKS)

    try:
        onnx_path = ensure_model_available(
            model_name, imgsz, MODEL_TO_DIR, auto_download=True,
        )
    except Exception as e:
        warnings.warn(f"Could not get {model_name} @ {imgsz}: {e}")
        return

    onnx_dir = onnx_path.parent

    modes: list[tuple[str, bool]] = [
        ("trtutils", False),
        ("trtutils(graph)", True),
    ]

    for framework, cuda_graph in modes:
        data[framework].setdefault(model_name, {})

        for bs in batch_sizes:
            bs_key = str(bs)

            with contextlib.suppress(KeyError):
                if data[framework][model_name][bs_key] is not None and not overwrite:
                    print(f"Skipping {framework} batch={bs} (already exists)")
                    continue

            print(f"Benchmarking {framework} batch={bs}...")

            batch_onnx = (
                onnx_path if bs == 1
                else _export_batch_onnx(model_name, imgsz, bs, onnx_dir)
            )
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
                images = [image] * bs

                with nvtx.annotate(f"{framework} batch={bs}", color="blue"):
                    timings = benchmark_loop(
                        lambda: detector.end2end(images),
                        0, bench_iters, f"{framework} b={bs}",
                    )

                del detector

                results = compute_results(timings, batch_size=bs)
                data[framework][model_name][bs_key] = results
                write_data(device, "batch", data)

                print(
                    f"\t{framework} batch={bs}: "
                    f"{results['mean']:.2f}ms ±{results['ci95']:.2f}, "
                    f"{results['throughput']:.1f} img/s",
                )
            except Exception as e:
                warnings.warn(f"Failed {framework} batch={bs}: {e}")
                continue


def benchmark_ultralytics_batch(
    device: str,
    model_name: str,
    imgsz: int,
    batch_sizes: list[int],
    warmup_iters: int,
    bench_iters: int,
    *,
    overwrite: bool = False,
) -> None:
    """Benchmark ultralytics YOLO throughput across batch sizes."""
    import nvtx
    from ultralytics import YOLO

    image = cv2.imread(IMAGE_PATH)
    data = get_data(device, "batch", BATCH_FRAMEWORKS)

    ultralytics_dir = REPO_DIR / "data" / "ultralytics"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    pt_path = (ultralytics_dir / f"{model_name}.pt").resolve()

    # --- ultralytics(torch) ---
    framework = "ultralytics(torch)"
    data[framework].setdefault(model_name, {})

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

            with nvtx.annotate(f"{framework} batch={bs}", color="green"):
                timings = benchmark_loop(
                    lambda: yolo(images, imgsz=imgsz, verbose=False),
                    warmup_iters, bench_iters, f"{framework} b={bs}",
                )

            del yolo

            results = compute_results(timings, batch_size=bs)
            data[framework][model_name][bs_key] = results
            write_data(device, "batch", data)

            print(
                f"\t{framework} batch={bs}: "
                f"{results['mean']:.2f}ms ±{results['ci95']:.2f}, "
                f"{results['throughput']:.1f} img/s",
            )
        except Exception as e:
            warnings.warn(f"Failed {framework} batch={bs}: {e}")
            continue

    # --- ultralytics(trt) ---
    framework = "ultralytics(trt)"
    data[framework].setdefault(model_name, {})

    for bs in batch_sizes:
        bs_key = str(bs)

        with contextlib.suppress(KeyError):
            if data[framework][model_name][bs_key] is not None and not overwrite:
                print(f"Skipping {framework} batch={bs} (already exists)")
                continue

        print(f"Benchmarking {framework} batch={bs}...")

        engine_path = ultralytics_dir / f"{model_name}_{imgsz}_b{bs}.engine"
        base_engine_path = pt_path.with_suffix(".engine")

        if not engine_path.exists():
            print(f"\tBuilding ultralytics TRT engine for batch={bs}...")
            subprocess.run(
                [
                    "yolo", "export", f"model={pt_path}",
                    "format=engine", f"imgsz={imgsz}",
                    f"batch={bs}", "half",
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

            with nvtx.annotate(f"{framework} batch={bs}", color="orange"):
                timings = benchmark_loop(
                    lambda: yolo(images, imgsz=imgsz, verbose=False),
                    warmup_iters, bench_iters, f"{framework} b={bs}",
                )

            del yolo

            results = compute_results(timings, batch_size=bs)
            data[framework][model_name][bs_key] = results
            write_data(device, "batch", data)

            print(
                f"\t{framework} batch={bs}: "
                f"{results['mean']:.2f}ms ±{results['ci95']:.2f}, "
                f"{results['throughput']:.1f} img/s",
            )
        except Exception as e:
            warnings.warn(f"Failed {framework} batch={bs}: {e}")
            continue


# ---------------------------------------------------------------------------
# Optimization grid benchmarks (from optimizations.py)
# ---------------------------------------------------------------------------


def benchmark_optimizations(
    device: str,
    model_name: str,
    imgsz: int,
    warmup_iters: int,
    bench_iters: int,
) -> None:
    """Benchmark Detector with all combinations of optimization flags."""
    from trtutils.image import Detector

    image = cv2.imread(str(IMAGE_PATH))

    onnx_path = ensure_model_available(model_name, imgsz, MODEL_TO_DIR)
    engine_path = onnx_path.with_suffix(".engine")
    if not engine_path.exists():
        print(f"Building engine: {engine_path}")
        build_model(onnx_path, engine_path, imgsz, opt_level=1)

    # Generate config grid (skip invalid pagelocked+unified combo)
    configs: list[dict] = []
    for prep in ["cpu", "cuda", "trt"]:
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

    results: list[dict] = []
    print(f"\nBenchmarking {model_name} @ {imgsz}x{imgsz}")
    print(f"Warmup: {warmup_iters}, Iterations: {bench_iters}\n")

    for cfg in configs:
        desc = (
            f"prep={cfg['preprocessor']} graph={cfg['cuda_graph']} "
            f"pl={cfg['pagelocked_mem']} um={cfg['unified_mem']}"
        )
        try:
            detector = Detector(
                engine_path=engine_path,
                warmup=True,
                warmup_iterations=warmup_iters,
                preprocessor=cfg["preprocessor"],
                cuda_graph=cfg["cuda_graph"],
                pagelocked_mem=cfg["pagelocked_mem"],
                unified_mem=cfg["unified_mem"],
                verbose=False,
            )
            timings = benchmark_loop(
                lambda: detector.end2end(image), 0, bench_iters, desc,
            )
            del detector

            stats = compute_results(timings)
            results.append({**cfg, **stats})
        except Exception as e:
            print(f"FAILED: {desc} - {e}")
            continue

    # Print results table sorted by mean latency
    results.sort(key=lambda x: x["mean"])

    print("\n" + "=" * 90)
    print(
        f"{'Preprocessor':<12} {'CUDAGraph':<10} {'Pagelocked':<10} "
        f"{'Unified':<10} {'Mean(ms)':<10} {'Std(ms)':<10} {'Min(ms)':<10}",
    )
    print("=" * 90)

    for r in results:
        print(
            f"{r['preprocessor']:<12} {str(r['cuda_graph']):<10} "
            f"{str(r['pagelocked_mem']):<10} {str(r['unified_mem']):<10} "
            f"{r['mean']:<10.3f} {r['std']:<10.3f} {r['min']:<10.3f}",
        )

    if len(results) >= 2:
        speedup = (
            results[-1]["mean"] / results[0]["mean"]
            if results[0]["mean"] > 0 else 0
        )
        print("=" * 90)
        print(
            f"Fastest: {results[0]['preprocessor']}, "
            f"graph={results[0]['cuda_graph']}, "
            f"pl={results[0]['pagelocked_mem']}, "
            f"um={results[0]['unified_mem']}",
        )
        print(f"Max speedup: {speedup:.2f}x")

    # Write results
    out_dir = DATA_DIR / "optimizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{device}.json"
    payload = {
        "device": device,
        "model": model_name,
        "imgsz": imgsz,
        "warmup": warmup_iters,
        "iterations": bench_iters,
        "results": results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")


# ---------------------------------------------------------------------------
# Model bootstrap (from run.py)
# ---------------------------------------------------------------------------


def bootstrap_models(models: list[str]) -> None:
    """Download all models for their configured image sizes."""
    total_downloads = sum(
        len(MODEL_TO_IMGSIZES.get(model, [])) for model in models
    )

    print(f"\nBootstrapping models: {', '.join(models)}")
    print(f"Total downloads: {total_downloads}\n")

    failed: list[tuple[str, int]] = []
    for model in models:
        model_sizes = MODEL_TO_IMGSIZES.get(model, [])
        if not model_sizes:
            warnings.warn(f"No image sizes configured for {model}, skipping")
            continue

        for imgsz in model_sizes:
            try:
                ensure_model_available(
                    model, imgsz, MODEL_TO_DIR, auto_download=True,
                )
                print(f"SUCCESS - {model} @ {imgsz}")
            except Exception as e:
                print(f"FAILED - {model} @ {imgsz}: {e}")
                failed.append((model, imgsz))

    if failed:
        print(f"\nFAILED - Failed to download {len(failed)} model(s)")
        for model, imgsz in failed:
            print(f"\t{model} @ {imgsz}")
    else:
        print("\nSUCCESS - All models downloaded successfully")
