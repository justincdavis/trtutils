# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S603, S607, T201
"""Benchmark runner functions."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import time
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from tqdm import tqdm

from .config import (
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

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def _export_batch_onnx(
    model_name: str,
    imgsz: int,
    batch_size: int,
    output_dir: Path,
) -> Path:
    onnx_path = output_dir / f"{model_name}_{imgsz}_b{batch_size}.onnx"
    if onnx_path.exists():
        return onnx_path

    print(f"Exporting {model_name} ONNX with batch={batch_size}...")

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
    model_name = onnx_path.stem.rsplit("_", 1)[0]
    return onnx_path.parent / f"{model_name}_{imgsz}_b{batch_size}.engine"


def _run_benchmarks(
    device: str,
    model_name: str,
    configs: list[tuple[str, int, int]],
    bench_iters: int,
    data_subdir: str,
    frameworks: list[str],
    modes: list[tuple[str, Any]],
    runner_factory: Callable,
    loop_warmup: int = 0,
    *,
    overwrite: bool = False,
) -> None:
    data = get_data(device, data_subdir, frameworks)

    for framework, flag in modes:
        if framework not in frameworks:
            continue
        data[framework].setdefault(model_name, {})

        for data_key, imgsz, bs in configs:
            with contextlib.suppress(KeyError):
                if data[framework][model_name][data_key] is not None and not overwrite:
                    print(f"Skipping {framework} {data_key} (already exists)")
                    continue

            print(f"Benchmarking {framework} {model_name} imgsz={imgsz} batch={bs}...")

            try:
                with runner_factory(flag, imgsz, bs) as exec_fn:
                    timings = benchmark_loop(
                        exec_fn,
                        loop_warmup,
                        bench_iters,
                        f"{framework} {data_key}",
                    )

                results = compute_results(timings, batch_size=bs)
                data[framework][model_name][data_key] = results
                write_data(device, data_subdir, data)

                print(
                    f"\t{framework} {data_key}: "
                    f"{results['mean']:.2f}ms ±{results['ci95']:.2f}, "
                    f"{results['throughput']:.1f} img/s",
                )
            except Exception as e:
                warnings.warn(f"Failed {framework} {data_key}: {e}")
                continue


@contextlib.contextmanager
def _runner_trtutils(mode, imgsz, bs, *, model_name, warmup_iters, image):
    from trtutils.image import Detector

    cuda_graph = mode in ("detector_graph", "raw_graph")
    is_raw = mode in ("raw", "raw_graph")

    onnx_path = ensure_model_available(
        model_name,
        imgsz,
        MODEL_TO_DIR,
        auto_download=True,
    )
    if bs > 1:
        batch_onnx = _export_batch_onnx(model_name, imgsz, bs, onnx_path.parent)
        engine_path = _engine_path_for_batch(onnx_path, imgsz, bs)
    else:
        batch_onnx = onnx_path
        engine_path = onnx_path.with_suffix(".engine")
    if not engine_path.exists():
        print(f"\tBuilding engine for {imgsz}x{bs}...")
        build_model(
            onnx=batch_onnx,
            output=engine_path,
            imgsz=imgsz,
            batch_size=bs,
            model_name=model_name,
        )
    detector = Detector(
        engine_path=engine_path,
        warmup_iterations=warmup_iters,
        warmup=True,
        preprocessor="cuda",
        pagelocked_mem=True,
        cuda_graph=cuda_graph,
        verbose=False,
    )
    if is_raw:
        engine = detector.engine
        if cuda_graph:
            exec_fn = lambda: engine.graph_exec(debug=True)
        else:
            input_ptrs = [b.allocation for b in engine.input_bindings]
            exec_fn = lambda: engine.raw_exec(input_ptrs, debug=True, no_warn=True)
    else:
        images = [image] * bs
        exec_fn = lambda: detector.end2end(images)

    try:
        yield exec_fn
    finally:
        del detector


@contextlib.contextmanager
def _runner_ultralytics(compile_engine, imgsz, bs, *, model_name, image):
    from ultralytics import YOLO

    ultralytics_dir = REPO_DIR / "data" / "ultralytics"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    pt_path = (ultralytics_dir / f"{model_name}.pt").resolve()

    if compile_engine:
        if bs > 1:
            engine_path = ultralytics_dir / f"{model_name}_{imgsz}_b{bs}.engine"
        else:
            engine_path = ultralytics_dir / f"{model_name}_{imgsz}.engine"
        base_engine = pt_path.with_suffix(".engine")
        if not engine_path.exists():
            print(f"\tBuilding ultralytics TRT engine for {imgsz}x{bs}...")
            export_cmd = [
                "yolo",
                "export",
                f"model={pt_path}",
                "format=engine",
                f"imgsz={imgsz}",
            ]
            if bs > 1:
                export_cmd.append(f"batch={bs}")
            export_cmd.append("half")
            subprocess.run(export_cmd, check=True, capture_output=True)
            if not base_engine.exists():
                err_msg = f"Ultralytics TRT engine not found: {base_engine}"
                raise FileNotFoundError(err_msg)
            base_engine.rename(engine_path)
        model_path = engine_path
    else:
        model_path = pt_path
    yolo = YOLO(model=model_path, task="detect", verbose=False)
    images = [image] * bs
    try:
        yield lambda: yolo(images, imgsz=imgsz, verbose=False)
    finally:
        del yolo


@contextlib.contextmanager
def _runner_sahi(
    backend_tag,
    *,
    trt_path,
    utrt_path,
    image,
    imgsz,
    overlap,
    conf_thres,
    warmup_iters,
):
    from sahi.predict import get_sliced_prediction

    from trtutils.compat.sahi import TRTDetectionModel
    from trtutils.image import SAHI, Detector

    from .sahi_compat import UltralyticsTRTDetector

    if backend_tag == "native":
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
        try:
            yield lambda: len(
                sahi_obj.end2end(image, conf_thres=conf_thres, verbose=False),
            )
        finally:
            del sahi_obj, detector
    elif backend_tag in ("trtutils_sahi", "ultralytics_trt"):
        if backend_tag == "trtutils_sahi":
            detection_model = TRTDetectionModel(
                model_path=str(trt_path),
                confidence_threshold=conf_thres,
            )
        else:
            detection_model = UltralyticsTRTDetector(
                model_path=str(utrt_path),
                confidence_threshold=conf_thres,
                device="cuda",
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
        try:
            yield lambda: len(
                get_sliced_prediction(
                    SAHI_IMAGE_PATH,
                    detection_model,
                    slice_height=imgsz,
                    slice_width=imgsz,
                    overlap_height_ratio=overlap,
                    overlap_width_ratio=overlap,
                    verbose=0,
                ).object_prediction_list,
            )
        finally:
            del detection_model
    else:
        err_msg = f"Unknown SAHI backend: {backend_tag}"
        raise ValueError(err_msg)


def run_benchmark(
    kind: str,
    device: str,
    model_name: str,
    warmup_iters: int,
    bench_iters: int,
    *,
    configs: list[tuple[str, int, int]] | None = None,
    data_subdir: str = "models",
    frameworks: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    if kind == "trtutils" or kind == "ultralytics":
        image = cv2.imread(IMAGE_PATH)
        if kind == "trtutils":
            runner = partial(
                _runner_trtutils,
                model_name=model_name,
                warmup_iters=warmup_iters,
                image=image,
            )
            modes = modes=[
                ("trtutils", "detector"),
                ("trtutils(graph)", "detector_graph"),
                ("tensorrt", "raw"),
                ("tensorrt(graph)", "raw_graph"),
            ]
        else:
            runner = partial(
                _runner_ultralytics,
                model_name=model_name,
                image=image,
            )
            modes=[("ultralytics(torch)", False), ("ultralytics(trt)", True)],
        _run_benchmarks(
            device,
            model_name,
            configs,
            bench_iters,
            data_subdir,
            frameworks,
            modes,
            runner_factory=runner,
            loop_warmup=warmup_iters,
            overwrite=overwrite,
        )
    elif kind == "sahi":
        image = cv2.imread(SAHI_IMAGE_PATH)
        imgsz = 640
        overlap = 0.2
        conf_thres = 0.25

        data = get_data(device, "models", MODEL_FRAMEWORKS)
        sahi_key = f"sahi_{model_name}"
        if sahi_key not in data:
            data[sahi_key] = {}

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
        utrt_path = ultralytics_dir / f"{model_name}_{imgsz}.engine"
        if not utrt_path.exists():
            err_msg = f"Ultralytics engine not found: {utrt_path}. Run ultralytics benchmark first."
            raise FileNotFoundError(err_msg)

        runner = partial(
            _runner_sahi,
            trt_path=trt_path,
            utrt_path=utrt_path,
            image=image,
            imgsz=imgsz,
            overlap=overlap,
            conf_thres=conf_thres,
            warmup_iters=warmup_iters,
        )

        modes = [
            ("trtutils", "native"),
            ("sahi(trtutils)", "trtutils_sahi"),
            ("sahi(ultralytics)(trt)", "ultralytics_trt"),
        ]

        for mode_key, backend_tag in modes:
            if mode_key in data[sahi_key] and not overwrite:
                print(f"Skipping {mode_key} (already exists)")
                continue

            print(f"\tBenchmarking {mode_key}...")

            try:
                with runner(backend_tag) as exec_fn:
                    timings: list[float] = []
                    detection_counts: list[int] = []
                    for _ in tqdm(range(bench_iters)):
                        t0 = time.perf_counter()
                        count = exec_fn()
                        timings.append(time.perf_counter() - t0)
                        detection_counts.append(count)

                results = compute_results(timings)
                avg_detections = sum(detection_counts) // len(detection_counts)
                data[sahi_key][mode_key] = {
                    "timing": results,
                    "detections": avg_detections,
                }
                print(
                    f"\t\t{mode_key}: {results['mean']:.2f}ms, {avg_detections} detections",
                )
                write_data(device, "models", data)
            except Exception as e:
                warnings.warn(f"Failed {mode_key}: {e}")
                continue
    else:
        err_msg = f"Unknown benchmark kind: {kind}"
        raise ValueError(err_msg)


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

    configs: list[dict] = []
    for prep in ["cpu", "cuda", "trt"]:
        for cuda_graph in [False, True]:
            for pagelocked in [False, True]:
                for unified in [False, True]:
                    if pagelocked and unified:
                        continue
                    configs.append(
                        {
                            "preprocessor": prep,
                            "cuda_graph": cuda_graph,
                            "pagelocked_mem": pagelocked,
                            "unified_mem": unified,
                        }
                    )

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
                lambda: detector.end2end(image),
                0,
                bench_iters,
                desc,
            )
            del detector

            stats = compute_results(timings)
            results.append({**cfg, **stats})
        except Exception as e:
            print(f"FAILED: {desc} - {e}")
            continue

    results.sort(key=lambda x: x["mean"])

    print("\n" + "=" * 90)
    print(
        f"{'Preprocessor':<12} {'CUDAGraph':<10} {'Pagelocked':<10} "
        f"{'Unified':<10} {'Mean(ms)':<10} {'Std(ms)':<10} {'Min(ms)':<10}",
    )
    print("=" * 90)

    for r in results:
        print(
            f"{r['preprocessor']:<12} {r['cuda_graph']!s:<10} "
            f"{r['pagelocked_mem']!s:<10} {r['unified_mem']!s:<10} "
            f"{r['mean']:<10.3f} {r['std']:<10.3f} {r['min']:<10.3f}",
        )

    if len(results) >= 2:
        speedup = results[-1]["mean"] / results[0]["mean"] if results[0]["mean"] > 0 else 0
        print("=" * 90)
        print(
            f"Fastest: {results[0]['preprocessor']}, "
            f"graph={results[0]['cuda_graph']}, "
            f"pl={results[0]['pagelocked_mem']}, "
            f"um={results[0]['unified_mem']}",
        )
        print(f"Max speedup: {speedup:.2f}x")

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


def bootstrap_models(models: list[str]) -> None:
    """Download all models for their configured image sizes."""
    total_downloads = sum(len(MODEL_TO_IMGSIZES.get(model, [])) for model in models)

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
                    model,
                    imgsz,
                    MODEL_TO_DIR,
                    auto_download=True,
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
