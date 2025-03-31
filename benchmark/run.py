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
            return json.load(f)
    else:
        # NOTE: Add framework name when another is being added for comparision
        return {"ultralytics": {}, "trtutils": {}}


def write_data(device: str, data: dict[str, dict[str, dict[str, dict[str, float]]]]) -> None:
    file_path = DATA_DIR / f"{device}.json"
    with file_path.open("w") as f:
        json.dump(data, f, indent=4)


def benchmark_trtutils(device: str, warmup_iters: int, bench_iters: int, *, overwrite: bool) -> None:
    from trtutils.impls.yolo import YOLO
    from trtutils.trtexec import build_engine

    # resolve paths
    image = cv2.imread(IMAGE_PATH)

    # get initial data
    data = get_data(device)

    for imgsz in IMAGE_SIZES:
        # if we can find the model nested, then we can skip
        with contextlib.suppress(KeyError):
            if data["trtutils"][MODELNAME][str(imgsz)] is not None and not overwrite:
                continue

        print(f"Processing trtutils on {MODELNAME} for imgsz={imgsz}...")

        # resolve paths
        weight_path = ONNX_DIR / f"{MODELNAME}_{imgsz}.onnx"
        trt_path = weight_path.with_suffix(".engine")

        if not trt_path.exists():
            print("\tBuilding trtutils engine...")
            build_engine(
                weight_path,
                trt_path,
                fp16=True,
                # patch for compiling the yolov9 exported onnx
                shapes=[("images", (1, 3, imgsz, imgsz))] if "yolov9" in MODELNAME else None,
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

        if MODELNAME not in data["trtutils"]:
            data["trtutils"][MODELNAME] = {}
        data["trtutils"][MODELNAME][str(imgsz)] = trt_results
        write_data(device, data)


def benchmark_ultralytics(device: str, warmup_iters: int, bench_iters: int, *, overwrite: bool) -> None:
    from ultralytics import YOLO

    # resolve paths
    ultralytics_weight_path = (REPO_DIR / "data" / "ultralytics" / f"{MODELNAME}.pt").resolve()
    utrt_base_path = ultralytics_weight_path.with_suffix(".engine")
    image = cv2.imread(IMAGE_PATH)

    # read initial data
    data = get_data(device)

    for imgsz in IMAGE_SIZES:
        # if we can find the model nested, then we can skip
        with contextlib.suppress(KeyError):
            if data["ultralytics"][MODELNAME][str(imgsz)] is not None and not overwrite:
                continue

        # resolve paths
        # make a "smarter" path
        utrt_path = ultralytics_weight_path.parent / f"{ultralytics_weight_path.stem}_{imgsz}.engine"

        print(f"Processing ultralytics on {MODELNAME} for imgsz={imgsz}...")
        # chekc if the engine file is already built
        if not utrt_path.exists():
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
                err_msg = f"Ultralytics TensorRT engine not found: {utrt_path}"
                raise FileNotFoundError(err_msg)
            
            # copy the utrt_base_path to utrt_path
            utrt_base_path.rename(utrt_path)

        # perform actual benchmark
        print("\tBenchmarking ultralytics engine...")
        u_yolo = YOLO(model=utrt_path, task="detect", verbose=False)
        for _ in range(warmup_iters):
            u_yolo(image, imgsz=imgsz, verbose=False)
        u_timing = []
        for _ in tqdm(range(bench_iters)):
            t0 = time.perf_counter()
            u_yolo(image, imgsz=imgsz, verbose=False)
            u_timing.append(time.perf_counter() - t0)
        del u_yolo

        u_results = get_results(u_timing)

        if MODELNAME not in data["ultralytics"]:
            data["ultralytics"][MODELNAME] = {}
        data["ultralytics"][MODELNAME][str(imgsz)] = u_results
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
        yolo_version = copy.copy(MODELNAME).replace("t", "").replace("n", "").replace("s", "").replace("m", "")
        global ONNX_DIR
        ONNX_DIR = REPO_DIR / "data" / yolo_version

        # trtutils benchmark
        if args.trtutils:
            benchmark_trtutils(
                args.device,
                args.warmup,
                args.iterations,
                overwrite=args.overwrite,
            )

        # ultralytics benchmark
        if args.ultralytics:
            if MODELNAME in ULTRALYTICS_MODELS:
                benchmark_ultralytics(
                    args.device,
                    args.warmup,
                    args.iterations,
                    overwrite=args.overwrite,
                )
            else:
                warnings.warn(f"Could not process: {MODELNAME}, since it is not a valid ultralytics model")


if __name__ == "__main__":
    main()
