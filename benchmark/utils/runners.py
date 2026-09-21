# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S603, S607, T201
"""Benchmark runner functions."""

from __future__ import annotations

import contextlib
import itertools
import json
import subprocess
import time
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from tqdm import tqdm

import trtutils
from trtutils.download import download

from .config import (
    DATA_DIR,
    IMAGE_PATH,
    MODEL_FRAMEWORKS,
    MODEL_TO_DIR,
    MODEL_TO_IMGSIZES,
    REPO_DIR,
    SAHI_IMAGE_PATH,
    get_timing_cache_path,
)
from .data import get_data, write_data
from .models import build_model, ensure_model_available
from .timing import benchmark_loop, compute_results

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

# alternating resolutions for the resolution-switch composition; see
# benchmark_optimizations
_RESOLUTION_SWITCH_SIZES = [(1280, 720), (1920, 1080)]


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

    # the download tool builds its own deterministic venv per model family,
    # so the benchmark environment does not need the upstream frameworks
    # installed. families whose exporters cannot set a batch size raise
    # NotImplementedError rather than silently exporting at batch 1.
    output_dir.mkdir(parents=True, exist_ok=True)
    download(
        model=model_name,
        output=onnx_path,
        opset=17,
        imgsz=imgsz,
        batch=batch_size,
        verbose=False,
    )

    if not onnx_path.exists():
        err_msg = f"ONNX export failed, file not found: {onnx_path}"
        raise FileNotFoundError(err_msg)

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

    def build_engine():
        print(f"\tBuilding engine for {imgsz}x{bs}...")
        build_model(
            onnx=batch_onnx,
            output=engine_path,
            imgsz=imgsz,
            batch_size=bs,
            model_name=model_name,
            timing_cache=get_timing_cache_path(),
        )

    def create_detector() -> Detector:
        return Detector(
            engine_path=engine_path,
            warmup_iterations=warmup_iters,
            warmup=True,
            preprocessor="cuda",
            pagelocked_mem=True,
            cuda_graph=cuda_graph,
            verbose=False,
        )

    if not engine_path.exists():
        build_engine()

    try:
        detector = create_detector()
    except RuntimeError as e:
        if "Failed to deserialize" not in str(e):
            raise
        print(f"\tEngine incompatible, rebuilding: {engine_path}")
        engine_path.unlink(missing_ok=True)
        build_engine()
        detector = create_detector()

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

    if bs > 1:
        engine_path = ultralytics_dir / f"{model_name}_{imgsz}_b{bs}.engine"
    else:
        engine_path = ultralytics_dir / f"{model_name}_{imgsz}.engine"
    base_engine = pt_path.with_suffix(".engine")

    def build_engine():
        print(f"\tBuilding ultralytics TRT engine for {imgsz}x{bs}...")
        export_cmd = [
            "yolo",
            "export",
            f"model={pt_path}",
            "format=engine",
            f"imgsz={imgsz}",
            "half=True",
        ]
        if bs > 1:
            export_cmd.append(f"batch={bs}")
        subprocess.run(export_cmd, check=True, capture_output=True)
        if not base_engine.exists():
            err_msg = f"Ultralytics TRT engine not found: {base_engine}"
            raise FileNotFoundError(err_msg)
        base_engine.rename(engine_path)

    def create_yolo() -> YOLO:
        return YOLO(model=model_path, task="detect", verbose=False)

    if compile_engine:
        model_path = engine_path
    else:
        model_path = pt_path

    if not engine_path.exists():
        build_engine()

    try:
        yolo = create_yolo()
    except Exception as e:
        if "Failed to deserialize" not in str(e):
            raise
        print(f"\tEngine incompatible, rebuilding: {engine_path}")
        engine_path.unlink(missing_ok=True)
        build_engine()
        yolo = create_yolo()

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
            modes = modes = [
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
            modes = ([("ultralytics(torch)", False), ("ultralytics(trt)", True)],)
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


def _trtutils_git_sha() -> str | None:
    """Git SHA of the trtutils checkout actually imported, or None outside a repo."""
    src_dir = Path(trtutils.__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=src_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return result.stdout.strip()


def write_stage_snapshot(device: str, stage: str, section: str, payload: dict) -> Path:
    """
    Merge one benchmark section into data/perf-series/<device>/stage-<stage>.json.

    ``optimize`` and ``batch`` each own a section of the same stage file, so
    running them in either order (or re-running one) never discards the other.
    """
    out_dir = DATA_DIR / "perf-series" / device
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stage-{stage}.json"
    snapshot: dict = {}
    if out_path.exists():
        with out_path.open("r") as f:
            snapshot = json.load(f)
    snapshot.update({"device": device, "stage": stage, "trtutils_sha": _trtutils_git_sha()})
    snapshot[section] = payload
    with out_path.open("w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Wrote stage snapshot: {out_path} [{section}]")
    return out_path


def benchmark_optimizations(
    device: str,
    model_name: str,
    imgsz: int,
    warmup_iters: int,
    bench_iters: int,
    stage: str | None = None,
) -> None:
    """Benchmark Detector with all combinations of optimization flags."""
    from trtutils.image import Detector

    image = cv2.imread(str(IMAGE_PATH))

    onnx_path = ensure_model_available(model_name, imgsz, MODEL_TO_DIR)

    def _engine_for_batch(batch: int) -> Path:
        """Static engine matching a batch size, built on first use."""
        if batch == 1:
            path = onnx_path.with_suffix(".engine")
            source = onnx_path
        else:
            source = _export_batch_onnx(model_name, imgsz, batch, onnx_path.parent)
            path = _engine_path_for_batch(onnx_path, imgsz, batch)
        if not path.exists():
            print(f"Building engine for batch {batch}: {path.name}")
            build_model(
                onnx=source,
                output=path,
                imgsz=imgsz,
                batch_size=batch,
                model_name=model_name,
                opt_level=1,
                timing_cache=get_timing_cache_path(),
            )
        return path

    # Input composition matters as much as the flags. A batch of identically
    # sized images never exercises the heterogeneous staging path, a single
    # image never exercises batching at all, and neither exercises a shape
    # that changes between successive calls, so a grid over flags alone
    # leaves all three invisible.
    inputs: dict[str, list] = {
        "single": [image],
        "homogeneous8": [cv2.resize(image, (1280, 720))] * 8,
        "heterogeneous8": [
            cv2.resize(image, size)
            for size in [
                (640, 480),
                (1280, 720),
                (800, 600),
                (1920, 1080),
                (512, 512),
                (1024, 768),
                (640, 640),
                (1600, 900),
            ]
        ],
        # single-image calls that alternate resolution every iteration; batch
        # size stays 1, only the shape submitted to the detector changes
        "resolution-switch": [cv2.resize(image, size) for size in _RESOLUTION_SWITCH_SIZES],
    }

    configs: list[dict] = []
    for prep in ["cpu", "cuda", "trt"]:
        for cuda_graph in [False, True]:
            for pagelocked in [False, True]:
                for unified in [False, True]:
                    if pagelocked and unified:
                        continue
                    for inputs_key in inputs:
                        configs.append(
                            {
                                "preprocessor": prep,
                                "cuda_graph": cuda_graph,
                                "pagelocked_mem": pagelocked,
                                "unified_mem": unified,
                                "inputs": inputs_key,
                            }
                        )

    results: list[dict] = []
    print(f"\nBenchmarking {model_name} @ {imgsz}x{imgsz}")
    print(f"Warmup: {warmup_iters}, Iterations: {bench_iters}\n")

    for cfg in configs:
        desc = (
            f"prep={cfg['preprocessor']} graph={cfg['cuda_graph']} "
            f"pl={cfg['pagelocked_mem']} um={cfg['unified_mem']} "
            f"in={cfg['inputs']}"
        )
        is_resolution_switch = cfg["inputs"] == "resolution-switch"
        batch = 1 if is_resolution_switch else len(inputs[cfg["inputs"]])
        try:
            engine_path = _engine_for_batch(batch)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {desc} - no engine for batch {batch}: {e}")
            continue
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
            if is_resolution_switch:
                images_cycle = itertools.cycle(inputs["resolution-switch"])
                exec_fn = lambda: detector.end2end(next(images_cycle))  # noqa: B023
            else:
                batch_images = inputs[cfg["inputs"]]
                exec_fn = lambda imgs=batch_images: detector.end2end(imgs)
            timings = benchmark_loop(
                exec_fn,
                0,
                bench_iters,
                desc,
            )
            del detector

            stats = compute_results(timings, batch_size=batch)
            results.append({**cfg, **stats})
        except Exception as e:
            print(f"FAILED: {desc} - {e}")
            continue

    results.sort(key=lambda x: x["mean"])

    print("\n" + "=" * 110)
    print(
        f"{'Preprocessor':<12} {'Inputs':<16} {'CUDAGraph':<10} {'Pagelocked':<10} "
        f"{'Unified':<10} {'Mean(ms)':<10} {'Std(ms)':<10} {'Min(ms)':<10}",
    )
    print("=" * 110)

    for r in results:
        print(
            f"{r['preprocessor']:<12} {r['inputs']:<16} {r['cuda_graph']!s:<10} "
            f"{r['pagelocked_mem']!s:<10} {r['unified_mem']!s:<10} "
            f"{r['mean']:<10.3f} {r['std']:<10.3f} {r['min']:<10.3f}",
        )

    if len(results) >= 2:
        speedup = results[-1]["mean"] / results[0]["mean"] if results[0]["mean"] > 0 else 0
        print("=" * 90)
        print(
            f"Fastest: {results[0]['preprocessor']}, "
            f"inputs={results[0]['inputs']}, "
            f"graph={results[0]['cuda_graph']}, "
            f"pl={results[0]['pagelocked_mem']}, "
            f"um={results[0]['unified_mem']}",
        )
        print(f"Max speedup: {speedup:.2f}x")

    payload = {
        "device": device,
        "model": model_name,
        "imgsz": imgsz,
        "warmup": warmup_iters,
        "iterations": bench_iters,
        "results": results,
    }
    if stage is not None:
        write_stage_snapshot(device, stage, "optimize", payload)
        return
    out_dir = DATA_DIR / "optimizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{device}.json"
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
