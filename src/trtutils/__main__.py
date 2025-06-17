# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Main entry point for trtutils CLI."""

from __future__ import annotations

import argparse
import operator
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import cv2ext
import numpy as np

import trtutils
from trtutils._log import LOG
from trtutils.trtexec._cli import cli_trtexec

if TYPE_CHECKING:
    from types import SimpleNamespace

    from ._benchmark import Metric


def _benchmark(args: SimpleNamespace) -> None:
    mpath = Path(args.engine)
    if not mpath.exists():
        err_msg = f"Cannot find provided engine: {mpath}"
        raise FileNotFoundError(err_msg)

    latency: Metric
    energy: Metric | None = None
    power: Metric | None = None
    if args.jetson:
        jresult = trtutils.jetson.benchmark_engine(
            engine=mpath,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            tegra_interval=1,
            dla_core=args.dla_core,
            warmup=True,
            verbose=args.verbose,
        )
        latency = jresult.latency
        energy = jresult.energy
        power = jresult.power_draw
    else:
        result = trtutils.benchmark_engine(
            engine=mpath,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            dla_core=args.dla_core,
            warmup=True,
            verbose=args.verbose,
        )
        latency = result.latency

    latency_in_ms = {
        "mean": latency.mean * 1000.0,
        "median": latency.median * 1000.0,
        "min": latency.min * 1000.0,
        "max": latency.max * 1000.0,
    }
    LOG.info(f"Benchmarking result for: {mpath.stem}")
    LOG.info("=" * 40)
    LOG.info("Latency (ms):")
    LOG.info(f"  Mean   : {latency_in_ms['mean']:.2f}")
    LOG.info(f"  Median : {latency_in_ms['median']:.2f}")
    LOG.info(f"  Min    : {latency_in_ms['min']:.2f}")
    LOG.info(f"  Max    : {latency_in_ms['max']:.2f}")

    if energy is not None:
        energy_in_joules = {
            "mean": energy.mean / 1000.0,
            "median": energy.median / 1000.0,
            "min": energy.min / 1000.0,
            "max": energy.max / 1000.0,
        }
        LOG.info("Energy Consumption (J):")
        LOG.info(f"  Mean   : {energy_in_joules['mean']:.3f}")
        LOG.info(f"  Median : {energy_in_joules['median']:.3f}")
        LOG.info(f"  Min    : {energy_in_joules['min']:.3f}")
        LOG.info(f"  Max    : {energy_in_joules['max']:.3f}")

    if power is not None:
        power_in_watts = {
            "mean": power.mean / 1000.0,
            "median": power.median / 1000.0,
            "min": power.min / 1000.0,
            "max": power.max / 1000.0,
        }
        LOG.info("Power Draw (W):")
        LOG.info(f"  Mean   : {power_in_watts['mean']:.3f}")
        LOG.info(f"  Median : {power_in_watts['median']:.3f}")
        LOG.info(f"  Min    : {power_in_watts['min']:.3f}")
        LOG.info(f"  Max    : {power_in_watts['max']:.3f}")

    LOG.info("=" * 40)


def _build(args: SimpleNamespace) -> None:
    if args.int8:
        LOG.warning("Build API is unstable and experimental with INT8 quantization.")

    # Create ImageBatcher if calibration directory is provided
    batcher = None
    if args.calibration_dir is not None:
        if args.input_shape is None:
            err_msg = "Input shape must be provided when using calibration directory"
            raise ValueError(err_msg)
        if args.input_dtype is None:
            err_msg = "Input dtype must be provided when using calibration directory"
            raise ValueError(err_msg)

        batcher = trtutils.builder.ImageBatcher(
            image_dir=args.calibration_dir,
            shape=args.input_shape,
            dtype=args.input_dtype,
            batch_size=args.batch_size,
            order=args.data_order,
            max_images=args.max_images,
            resize_method=args.resize_method,
            input_scale=args.input_scale,
        )

    # actual call
    trtutils.build_engine(
        onnx=Path(args.onnx),
        output=Path(args.output),
        timing_cache=args.timing_cache,
        workspace=args.workspace,
        dla_core=args.dla_core,
        calibration_cache=args.calibration_cache,
        data_batcher=batcher,
        gpu_fallback=args.gpu_fallback,
        direct_io=args.direct_io,
        prefer_precision_constraints=args.prefer_precision_constraints,
        reject_empty_algorithms=args.reject_empty_algorithms,
        ignore_timing_mismatch=args.ignore_timing_mismatch,
        fp16=args.fp16,
        int8=args.int8,
        verbose=args.verbose,
    )


def _can_run_on_dla(args: SimpleNamespace) -> None:
    full_dla, chunks = trtutils.builder.can_run_on_dla(
        onnx=Path(args.onnx),
        verbose_layers=args.verbose_layers,
        verbose_chunks=args.verbose_chunks,
    )
    # compute portion layers compatible
    all_layers = 0
    compat_layers = 0
    for chunk in chunks:
        chunk_size = chunk[2] - chunk[1] + 1
        if chunk[-1]:
            compat_layers += chunk_size
        all_layers += chunk_size
    portion_compat = round((compat_layers / all_layers) * 100.0, 2)
    LOG.info(
        f"ONNX: {args.onnx}, Fully DLA Compatible: {full_dla}, Layers: {compat_layers} / {all_layers} ({portion_compat} % Compatible)"
    )


def _build_dla(args: SimpleNamespace) -> None:
    dtype: np.dtype = np.dtype("float32")
    if args.dtype == "float16":
        dtype = np.dtype("float16")
    elif args.dtype == "int8":
        dtype = np.dtype("int8")

    batcher = trtutils.builder.ImageBatcher(
        image_dir=args.image_dir,
        shape=args.shape,
        dtype=dtype,
        batch_size=args.batch_size,
        order=args.order,
        max_images=args.max_images,
        resize_method=args.resize_method,
        input_scale=args.input_scale,
        verbose=args.verbose,
    )
    trtutils.builder.build_dla_engine(
        onnx=Path(args.onnx),
        output_path=Path(args.output),
        data_batcher=batcher,
        dla_core=args.dla_core,
        max_chunks=args.max_chunks,
        min_layers=args.min_layers,
        timing_cache=args.timing_cache,
        verbose=args.verbose,
    )


def _run_yolo(args: SimpleNamespace) -> None:
    img_extensions = [".jpg", ".jpeg", ".png"]
    video_extensions = [".mp4", ".avi", ".mov"]

    input_path = Path(args.input)
    is_image = input_path.suffix in img_extensions
    is_video = input_path.suffix in video_extensions
    if not is_image and not is_video:
        err_msg = f"Invalid input file: {input_path}"
        raise ValueError(err_msg)

    yolo = trtutils.impls.yolo.YOLO(
        engine_path=args.engine,
        warmup_iterations=args.warmup_iterations,
        input_range=args.input_range,
        preprocessor=args.preprocessor,
        resize_method=args.resize_method,
        conf_thres=args.conf_thres,
        dla_core=args.dla_core,
        warmup=args.warmup,
        verbose=args.verbose,
    )
    times: dict[str, list[float]] = {
        "pre": [],
        "run": [],
        "post": [],
        "det": [],
    }

    def run(
        img: np.ndarray,
    ) -> tuple[
        list[tuple[tuple[int, int, int, int], float, int]], float, float, float, float
    ]:
        t0 = time.perf_counter()
        tensor, ratios, pads = yolo.preprocess(img, no_copy=True)
        t1 = time.perf_counter()
        results = yolo.run(tensor, preprocessed=True, postprocess=False, no_copy=True)
        t2 = time.perf_counter()
        p_results = yolo.postprocess(results, ratios, pads, no_copy=True)
        t3 = time.perf_counter()
        dets = yolo.get_detections(p_results)
        t4 = time.perf_counter()
        return (
            dets,
            round(1000 * (t1 - t0), 2),
            round(1000 * (t2 - t1), 2),
            round(1000 * (t3 - t2), 2),
            round(1000 * (t4 - t3), 2),
        )

    def log(
        dets: list[tuple[tuple[int, int, int, int], float, int]],
        pre_t: float,
        run_t: float,
        post_t: float,
        det_t: float,
    ) -> None:
        LOG.info(f"Found {len(dets)} detections")
        LOG.info(f"Preprocessing time: {pre_t} ms")
        LOG.info(f"Run time: {run_t} ms")
        LOG.info(f"Postprocessing time: {post_t} ms")
        LOG.info(f"Detection time: {det_t} ms")

    def process_image(
        img: np.ndarray,
    ) -> tuple[
        list[tuple[int, int, int, int]],
        list[float],
        list[int],
        float,
        float,
        float,
        float,
    ]:
        dets, pre_t, run_t, post_t, det_t = run(img)
        log(dets, pre_t, run_t, post_t, det_t)
        bboxes = [d[0] for d in dets]
        scores = [d[1] for d in dets]
        classes = [d[2] for d in dets]
        times["pre"].append(pre_t)
        times["run"].append(run_t)
        times["post"].append(post_t)
        times["det"].append(det_t)
        return bboxes, scores, classes, pre_t, run_t, post_t, det_t

    def draw(
        img: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
        scores: list[float],
        classes: list[int],
        pre_t: float,
        run_t: float,
        post_t: float,
        det_t: float,
    ) -> np.ndarray:
        canvas = cv2ext.bboxes.draw_bboxes(img, bboxes, scores, classes)
        canvas = cv2ext.image.draw.text(canvas, f"PRE:  {pre_t} ms", (10, 30))
        canvas = cv2ext.image.draw.text(canvas, f"RUN:  {run_t} ms", (10, 60))
        canvas = cv2ext.image.draw.text(canvas, f"POST: {post_t} ms", (10, 90))
        return cv2ext.image.draw.text(canvas, f"DET:  {det_t} ms", (10, 120))

    def _summarize_classes(
        dets: list[tuple[tuple[int, int, int, int], float, int]],
    ) -> list[tuple[int, int]]:
        class_count: dict[int, int] = {}
        for _, _, class_id in dets:
            if class_id not in class_count:
                class_count[class_id] = 0
            class_count[class_id] += 1
        return sorted(class_count.items(), key=operator.itemgetter(0))

    if is_image:
        img = cv2.imread(str(input_path))
        if img is None:
            err_msg = f"Failed to read image: {input_path}"
            raise ValueError(err_msg)

        bboxes, scores, classes, pre_t, run_t, post_t, det_t = process_image(img)

        if args.show:
            canvas = draw(img, bboxes, scores, classes, pre_t, run_t, post_t, det_t)
            cv2.imshow("YOLO", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif is_video:
        if args.show:
            display = cv2ext.Display(f"YOLO detection: {input_path.stem}")
        else:
            display = None
        for fid, frame in cv2ext.IterableVideo(input_path):
            if display is not None and display.stopped:
                break

            LOG.info(f"Processing frame {fid}")
            bboxes, scores, classes, pre_t, run_t, post_t, det_t = process_image(frame)

            if args.show:
                canvas = draw(
                    frame, bboxes, scores, classes, pre_t, run_t, post_t, det_t
                )
                if display is not None:
                    display.update(canvas)

        if display is not None:
            display.stop()

    else:
        err_msg = f"Invalid input file: {input_path}"
        raise ValueError(err_msg)

    LOG.info("Times:")
    for k, v in times.items():
        LOG.info(f"{k}: {np.mean(v):.2f} ms")


def _inspect(args: SimpleNamespace) -> None:
    engine_size, max_batch, inputs, outputs = trtutils.inspect.inspect_engine(
        Path(args.engine)
    )
    LOG.info(f"Engine Size: {engine_size / (1024 * 1024):.2f} MB")
    LOG.info(f"Max Batch Size: {max_batch}")
    LOG.info("Inputs:")
    for name, shape, dtype, fmt in inputs:
        LOG.info(f"\t{name}: shape={shape}, dtype={dtype}, format={fmt}")
    LOG.info("Outputs:")
    for name, shape, dtype, fmt in outputs:
        LOG.info(f"\t{name}: shape={shape}, dtype={dtype}, format={fmt}")


def _main() -> None:
    # common arguments parser
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--dla_core",
        type=int,
        default=None,
        help="DLA core to assign DLA layers of the engine to. Default is None.",
    )
    parent_parser.add_argument(
        "--log_level",
        choices=[
            "DEBUG",
            "debug",
            "INFO",
            "info",
            "WARNING",
            "warning",
            "ERROR",
            "error",
            "CRITICAL",
            "critical",
        ],
        default="INFO",
        help="Set the log level. Default is INFO.",
    )
    parent_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    # main parser
    parser = argparse.ArgumentParser(description="Utilities for TensorRT.")

    # create subparser for each command
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        required=True,
    )

    # benchmark command, will use benchmark engine by default, can pass jetson to use the jetson benchmarker
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark a given TensorRT engine.",
        parents=[parent_parser],
    )
    benchmark_parser.add_argument(
        "--engine",
        "-e",
        required=True,
        help="Path to the engine file.",
    )
    benchmark_parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=1000,
        help="Number of iterations to measure over.",
    )
    benchmark_parser.add_argument(
        "--warmup_iterations",
        "-wi",
        type=int,
        default=100,
        help="The number of iterations to warmup the model before measuring.",
    )
    benchmark_parser.add_argument(
        "--jetson",
        "-j",
        action="store_true",
        help="If True, will use the trtutils.jetson submodule benchmarker to record energy and pwoerdraw as well.",
    )
    benchmark_parser.set_defaults(func=_benchmark)

    # trtexec parser
    trtexec_parser = subparsers.add_parser(
        "trtexec",
        help="Run trtexec.",
    )
    trtexec_parser.set_defaults(func=cli_trtexec)

    # build_engine parser
    build_parser = subparsers.add_parser(
        "build",
        help="Build a TensorRT engine from an ONNX model.",
        parents=[parent_parser],
    )
    build_parser.add_argument(
        "--onnx",
        "-o",
        required=True,
        help="Path to the ONNX model file.",
    )
    build_parser.add_argument(
        "--output",
        "-out",
        required=True,
        help="Path to save the TensorRT engine file.",
    )
    build_parser.add_argument(
        "--device",
        "-d",
        choices=["gpu", "dla", "GPU", "DLA"],
        default="gpu",
        help="Device to use for the engine. Default is 'gpu'.",
    )
    build_parser.add_argument(
        "--timing_cache",
        "-tc",
        default=None,
        help="Path to store timing cache data. Default is 'timing.cache'.",
    )
    build_parser.add_argument(
        "--workspace",
        "-w",
        type=float,
        default=4.0,
        help="Workspace size in GB. Default is 4.0.",
    )
    build_parser.add_argument(
        "--calibration_cache",
        "-cc",
        default=None,
        help="Path to store calibration cache data. Default is 'calibration.cache'.",
    )
    build_parser.add_argument(
        "--calibration_dir",
        "-cd",
        default=None,
        help="Directory containing images for INT8 calibration.",
    )
    build_parser.add_argument(
        "--input_shape",
        "-is",
        type=int,
        nargs=3,
        help="Input shape in HWC format (height, width, channels). Required when using calibration directory.",
    )
    build_parser.add_argument(
        "--input_dtype",
        "-id",
        choices=["float32", "float16", "int8"],
        help="Input data type. Required when using calibration directory.",
    )
    build_parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=8,
        help="Batch size for calibration. Default is 8.",
    )
    build_parser.add_argument(
        "--data_order",
        "-do",
        choices=["NCHW", "NHWC"],
        default="NCHW",
        help="Data ordering expected by the network. Default is NCHW.",
    )
    build_parser.add_argument(
        "--max_images",
        "-mi",
        type=int,
        help="Maximum number of images to use for calibration.",
    )
    build_parser.add_argument(
        "--resize_method",
        "-rm",
        choices=["letterbox", "linear"],
        default="letterbox",
        help="Method to resize images. Default is letterbox.",
    )
    build_parser.add_argument(
        "--input_scale",
        "-sc",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="Input value range. Default is [0.0, 1.0].",
    )
    build_parser.add_argument(
        "--gpu_fallback",
        action="store_true",
        help="Allow GPU fallback for unsupported layers when building for DLA.",
    )
    build_parser.add_argument(
        "--direct_io",
        action="store_true",
        help="Use direct IO for the engine.",
    )
    build_parser.add_argument(
        "--prefer_precision_constraints",
        action="store_true",
        help="Prefer precision constraints.",
    )
    build_parser.add_argument(
        "--reject_empty_algorithms",
        action="store_true",
        help="Reject empty algorithms.",
    )
    build_parser.add_argument(
        "--ignore_timing_mismatch",
        action="store_true",
        help="Allow different CUDA device timing caches to be used.",
    )
    build_parser.add_argument(
        "--fp16",
        action="store_true",
        help="Quantize the engine to FP16 precision.",
    )
    build_parser.add_argument(
        "--int8",
        action="store_true",
        help="Quantize the engine to INT8 precision.",
    )
    build_parser.set_defaults(func=_build)

    # run_on_dla parser
    can_run_on_dla_parser = subparsers.add_parser(
        "can_run_on_dla",
        help="Evaluate if the model can run on a DLA.",
        parents=[parent_parser],
    )
    can_run_on_dla_parser.add_argument(
        "--onnx",
        "-o",
        required=True,
        help="Path to the ONNX model file.",
    )
    can_run_on_dla_parser.add_argument(
        "--verbose_layers",
        action="store_true",
        help="Print detailed information about each layer's DLA compatibility.",
    )
    can_run_on_dla_parser.add_argument(
        "--verbose_chunks",
        action="store_true",
        help="Print detailed information about layer chunks and their device assignments.",
    )
    can_run_on_dla_parser.set_defaults(func=_can_run_on_dla)

    # build_dla parser
    build_dla_parser = subparsers.add_parser(
        "build_dla",
        help="Build a TensorRT engine for DLA.",
        parents=[parent_parser],
    )
    build_dla_parser.add_argument(
        "--onnx",
        "-o",
        required=True,
        help="Path to the ONNX model file.",
    )
    build_dla_parser.add_argument(
        "--output",
        "-out",
        required=True,
        help="Path to save the TensorRT engine file.",
    )
    build_dla_parser.add_argument(
        "--max_chunks",
        type=int,
        default=1,
        help="Maximum number of DLA chunks to assign. Default is 1.",
    )
    build_dla_parser.add_argument(
        "--min_layers",
        type=int,
        default=20,
        help="Minimum number of layers in a chunk to be assigned to DLA. Default is 20.",
    )
    build_dla_parser.add_argument(
        "--image_dir",
        required=True,
        help="Path to the directory containing images for calibration.",
    )
    build_dla_parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=(640, 640, 3),
        help="Input shape in HWC format (height, width, channels).",
    )
    build_dla_parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "int8"],
        default="float32",
        help="Input data type. Required when using calibration directory.",
    )
    build_dla_parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for calibration. Default is 8.",
    )
    build_dla_parser.add_argument(
        "--order",
        choices=["NCHW", "NHWC"],
        default="NCHW",
        help="Data ordering expected by the network. Default is NCHW.",
    )
    build_dla_parser.add_argument(
        "--max_images",
        type=int,
        help="Maximum number of images to use for calibration.",
    )
    build_dla_parser.add_argument(
        "--resize_method",
        choices=["letterbox", "linear"],
        default="letterbox",
        help="Method to resize images. Default is letterbox.",
    )
    build_dla_parser.add_argument(
        "--input_scale",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="Input value range. Default is [0.0, 1.0].",
    )
    build_dla_parser.add_argument(
        "--timing_cache",
        "-tc",
        default=None,
        help="Path to store timing cache data. Default is 'timing.cache'.",
    )
    build_dla_parser.set_defaults(func=_build_dla)

    # yolo parser
    yolo_parser = subparsers.add_parser(
        "yolo",
        help="Run YOLO object detection on an image or video.",
        parents=[parent_parser],
    )
    yolo_parser.add_argument(
        "--engine",
        "-e",
        required=True,
        help="Path to the TensorRT engine file.",
    )
    yolo_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input image or video file.",
    )
    yolo_parser.add_argument(
        "--conf_thres",
        "-c",
        type=float,
        default=0.1,
        help="Confidence threshold for detections. Default is 0.1.",
    )
    yolo_parser.add_argument(
        "--input_range",
        "-r",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="Input value range. Default is [0.0, 1.0].",
    )
    yolo_parser.add_argument(
        "--preprocessor",
        "-p",
        choices=["cpu", "cuda", "trt"],
        default="trt",
        help="Preprocessor to use. Default is trt.",
    )
    yolo_parser.add_argument(
        "--resize_method",
        "-rm",
        choices=["letterbox", "linear"],
        default="letterbox",
        help="Method to resize images. Default is letterbox.",
    )
    yolo_parser.add_argument(
        "--warmup",
        "-w",
        action="store_true",
        help="Perform warmup iterations.",
    )
    yolo_parser.add_argument(
        "--warmup_iterations",
        "-wi",
        type=int,
        default=10,
        help="Number of warmup iterations. Default is 10.",
    )
    yolo_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the detections.",
    )
    yolo_parser.set_defaults(func=_run_yolo)

    # inspect parser
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a TensorRT engine.",
        parents=[parent_parser],
    )
    inspect_parser.add_argument(
        "--engine",
        "-e",
        required=True,
        help="Path to the TensorRT engine file.",
    )
    inspect_parser.set_defaults(func=_inspect)

    # parse args and call the function
    args, unknown = parser.parse_known_args()

    # set log level
    trtutils.set_log_level(args.log_level)

    # call function with args
    if hasattr(args, "func"):
        if args.command == "trtexec":
            args.func(unknown)
        else:
            args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
