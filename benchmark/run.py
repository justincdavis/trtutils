# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S603, S607, T201
"""Benchmark trtutils against popular frameworks."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import statistics
import subprocess
import warnings
import time
from pathlib import Path

import cv2
from tqdm import tqdm


# global paths
REPO_DIR = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_PATH = str((REPO_DIR / "data" / "horse.jpg").resolve())
MODELNAME = "yolov10n"
MODELNAMES = [
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov7t",
    "yolov7m",
    "yoloxt",
    "yoloxn",
    "yoloxs",
    "yoloxm",
]
ULTRALYTICS_MODELS = [
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov8n",
    "yolov8s",
    "yolov8m",
]
MODEL_DIRS = [
    "yolov10",
    "yolov9",
    "yolov8",
    "yolov7",
    "yolox",
]
ONNX_DIR = REPO_DIR / "data" / "yolov10"

# global vars
IMAGE_SIZES = [160, 320, 480, 640, 800, 960, 1120, 1280]
FRAMEWORKS = [
    "ultralytics(torch)",
    "ultralytics(trt)",
    "trtutils(cpu)",
    "trtutils(cuda)",
    "trtutils(trt)",
    "tensorrt",
]


def get_results(data: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(data) * 1000.0,
        "median": statistics.median(data) * 1000.0,
        "min": min(data) * 1000.0,
        "max": max(data) * 1000.0,
    }


def get_data(device: str) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    file_path = DATA_DIR / f"{device}.json"
    if file_path.exists():
        with file_path.open("r") as f:
            data = json.load(f)
            for f in FRAMEWORKS:
                if data.get(f) is None:
                    data[f] = {}
            return data
    else:
        return {f: {} for f in FRAMEWORKS}


def write_data(
    device: str, data: dict[str, dict[str, dict[str, dict[str, float]]]]
) -> None:
    file_path = DATA_DIR / f"{device}.json"
    with file_path.open("w") as f:
        json.dump(data, f, indent=4)


def benchmark_trtutils(
    device: str, warmup_iters: int, bench_iters: int, *, overwrite: bool
) -> None:
    from trtutils import TRTEngine, FLAGS
    from trtutils.impls.yolo import YOLO
    from trtutils.builder import build_engine

    # resolve paths
    image = cv2.imread(IMAGE_PATH)

    # get initial data
    data = get_data(device)

    for preprocessor in ["cpu", "cuda", "trt"]:
        if preprocessor == "trt" and not FLAGS.TRT_HAS_UINT8:
            continue

        framework = f"trtutils({preprocessor})"
        for imgsz in IMAGE_SIZES:
            # if we can find the model nested, then we can skip
            with contextlib.suppress(KeyError):
                if data[framework][MODELNAME][str(imgsz)] is not None and not overwrite:
                    continue

            print(f"Processing {framework} on {MODELNAME} for imgsz={imgsz}...")

            # resolve paths
            weight_path = ONNX_DIR / f"{MODELNAME}_{imgsz}.onnx"
            trt_path = weight_path.with_suffix(".engine")

            if not trt_path.exists():
                print("\tBuilding trtutils engine...")
                build_engine(
                    onnx=weight_path,
                    output=trt_path,
                    fp16=True,
                    timing_cache=str(Path(__file__).parent / "timing.cache"),
                    shapes=[("images", (1, 3, imgsz, imgsz))]
                    if "yolov9" in MODELNAME
                    else None,
                )

                # verify
                if not trt_path.exists():
                    err_msg = f"trtutils TensorRT engine not found: {trt_path}"
                    raise FileNotFoundError(err_msg)

            print("\tBenchmarking trtutils engine...")
            trt_yolo = YOLO(
                engine_path=trt_path,
                warmup_iterations=warmup_iters,
                warmup=True,
                preprocessor=preprocessor,
                pagelocked_mem=True,
                verbose=True,
            )
            t_timing = []
            for _ in tqdm(range(bench_iters)):
                t0 = time.perf_counter()
                # trt_yolo.end2end(image)
                trt_yolo.end2end(image, verbose=True)  # add verbose for debugging
                t_timing.append(time.perf_counter() - t0)
            del trt_yolo

            trt_results = get_results(t_timing)

            if MODELNAME not in data[framework]:
                data[framework][MODELNAME] = {}
            data[framework][MODELNAME][str(imgsz)] = trt_results
            write_data(device, data)

            # add the 'raw' engine execution when using cpu preprocessor
            if preprocessor == "cpu":
                print("\tBenchmarking tensorrt engine...")
                base_engine = TRTEngine(
                    engine_path=trt_path,
                    warmup_iterations=warmup_iters,
                    warmup=True,
                    verbose=True,
                )
                input_ptrs = [
                    binding.allocation for binding in base_engine.input_bindings
                ]
                r_timing = []
                for _ in tqdm(range(bench_iters)):
                    t00 = time.perf_counter()
                    # use the debug flag so a stream sync is completed
                    base_engine.raw_exec(input_ptrs, debug=True, no_warn=True)
                    r_timing.append(time.perf_counter() - t00)
                del base_engine

                raw_results = get_results(r_timing)

                if MODELNAME not in data["tensorrt"]:
                    data["tensorrt"][MODELNAME] = {}
                data["tensorrt"][MODELNAME][str(imgsz)] = raw_results
                write_data(device, data)


def benchmark_ultralytics(
    device: str, warmup_iters: int, bench_iters: int, *, overwrite: bool
) -> None:
    from ultralytics import YOLO

    # resolve paths
    ultralytics_weight_path = (
        REPO_DIR / "data" / "ultralytics" / f"{MODELNAME}.pt"
    ).resolve()
    utrt_base_path = ultralytics_weight_path.with_suffix(".engine")
    image = cv2.imread(IMAGE_PATH)

    # read initial data
    data = get_data(device)

    # handle both Torch and TensorRT backed
    for imgsz in IMAGE_SIZES:
        # resolve paths
        # make a "smarter" path
        utrt_path = (
            ultralytics_weight_path.parent
            / f"{ultralytics_weight_path.stem}_{imgsz}.engine"
        )
        compile_engine = False
        methods: list[tuple[str, Path, bool]] = [
            ("ultralytics(torch)", ultralytics_weight_path, False),
            ("ultralytics(trt)", utrt_path, True),
        ]

        for utag, upath, compile_engine in methods:
            # if we can find the model nested, then we can skip
            with contextlib.suppress(KeyError):
                if data[utag][MODELNAME][str(imgsz)] is not None and not overwrite:
                    continue

            print(f"Processing ultralytics on {MODELNAME} for imgsz={imgsz}...")
            # chekc if the engine file is already built
            if not upath.exists() and compile_engine:
                # build ultralytics tensorrt engine
                print("\tBuilding ONNX and Ultralytics engine...")
                subprocess.run(
                    [
                        "yolo",
                        "export",
                        f"model={ultralytics_weight_path}",
                        "format=engine",
                        f"imgsz={imgsz}",
                        "half",
                    ],
                    check=True,
                    capture_output=True,
                )

                # verify the build
                if not utrt_base_path.exists():
                    err_msg = f"Ultralytics TensorRT engine not found: {upath}"
                    raise FileNotFoundError(err_msg)

                # copy the utrt_base_path to upath
                utrt_base_path.rename(upath)

            # perform actual benchmark
            print("\tBenchmarking ultralytics engine...")
            u_yolo = YOLO(model=upath, task="detect", verbose=False)
            for _ in range(warmup_iters):
                u_yolo(image, imgsz=imgsz, verbose=False)
            u_timing = []
            for _ in tqdm(range(bench_iters)):
                t0 = time.perf_counter()
                u_yolo(image, imgsz=imgsz, verbose=False)
                u_timing.append(time.perf_counter() - t0)
            del u_yolo

            u_results = get_results(u_timing)

            if MODELNAME not in data[utag]:
                data[utag][MODELNAME] = {}
            data[utag][MODELNAME][str(imgsz)] = u_results
            write_data(device, data)


def main() -> None:
    """Run the benchmarking."""
    parser = argparse.ArgumentParser("Run benchmarking against popular frameworks.")
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="The name of the device you are generating a benchmark on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov10n",
        help="Which model to benchmark, special option: all",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="The number of warmup iterations to perform.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="The number of iterations to perform and measure for benchmarking.",
    )
    parser.add_argument(
        "--trtutils",
        action="store_true",
        help="Run the benchmarks on the trtutils frameworks.",
    )
    parser.add_argument(
        "--ultralytics",
        action="store_true",
        help="Run the benchmarks on the ultralytics framework.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data by rerunning benchmarks.",
    )
    args = parser.parse_args()

    # check if iterating over all possible models
    models: list[str] = []
    if args.model == "all":
        models = copy.deepcopy(MODELNAMES)
    else:
        models.append(args.model)

    # process each model
    for modelname in models:
        # solve for MODELNAME and ONNX_DIR
        global MODELNAME
        if modelname not in MODELNAMES:
            err_msg = f"Could not find: {modelname}"
            raise ValueError(err_msg)
        MODELNAME = modelname

        # now solve for new onnx_dir
        yolo_version = (
            copy.copy(MODELNAME)
            .replace("t", "")
            .replace("n", "")
            .replace("s", "")
            .replace("m", "")
        )
        global ONNX_DIR
        ONNX_DIR = REPO_DIR / "data" / yolo_version

        # trtutils benchmark
        if args.trtutils:
            try:
                benchmark_trtutils(
                    args.device,
                    args.warmup,
                    args.iterations,
                    overwrite=args.overwrite,
                )
            except Exception as e:
                warnings.warn(f"Failed to process {MODELNAME} with trtutils: {e}")
                continue

        # ultralytics benchmark
        if args.ultralytics:
            try:
                if MODELNAME in ULTRALYTICS_MODELS:
                    benchmark_ultralytics(
                        args.device,
                        args.warmup,
                        args.iterations,
                        overwrite=args.overwrite,
                    )
                else:
                    warnings.warn(
                        f"Could not process: {MODELNAME}, since it is not a valid ultralytics model"
                    )
            except Exception as e:
                warnings.warn(f"Failed to process {MODELNAME} with ultralytics: {e}")
                continue


if __name__ == "__main__":
    main()
