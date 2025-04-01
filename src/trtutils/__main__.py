# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: PLC0415
"""Main entry point for trtutils CLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import cv2ext
import numpy as np

import trtutils
from trtutils.trtexec._cli import cli_trtexec

if TYPE_CHECKING:
    from types import SimpleNamespace

    from ._benchmark import Metric


_log = logging.getLogger("trtutils")


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
            warmup=True,
        )
        latency = jresult.latency
        energy = jresult.energy
        power = jresult.power_draw
    else:
        result = trtutils.benchmark_engine(
            engine=mpath,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            warmup=True,
        )
        latency = result.latency

    latency_in_ms = {
        "mean": latency.mean * 1000.0,
        "median": latency.median * 1000.0,
        "min": latency.min * 1000.0,
        "max": latency.max * 1000.0,
    }
    _log.info(f"Benchmarking result for: {mpath.stem}")
    _log.info("=" * 40)
    _log.info("Latency (ms):")
    _log.info(f"  Mean   : {latency_in_ms['mean']:.2f}")
    _log.info(f"  Median : {latency_in_ms['median']:.2f}")
    _log.info(f"  Min    : {latency_in_ms['min']:.2f}")
    _log.info(f"  Max    : {latency_in_ms['max']:.2f}")

    if energy is not None:
        energy_in_joules = {
            "mean": energy.mean / 1000.0,
            "median": energy.median / 1000.0,
            "min": energy.min / 1000.0,
            "max": energy.max / 1000.0,
        }
        _log.info("Energy Consumption (J):")
        _log.info(f"  Mean   : {energy_in_joules['mean']:.3f}")
        _log.info(f"  Median : {energy_in_joules['median']:.3f}")
        _log.info(f"  Min    : {energy_in_joules['min']:.3f}")
        _log.info(f"  Max    : {energy_in_joules['max']:.3f}")

    if power is not None:
        power_in_watts = {
            "mean": power.mean / 1000.0,
            "median": power.median / 1000.0,
            "min": power.min / 1000.0,
            "max": power.max / 1000.0,
        }
        _log.info("Power Draw (W):")
        _log.info(f"  Mean   : {power_in_watts['mean']:.3f}")
        _log.info(f"  Median : {power_in_watts['median']:.3f}")
        _log.info(f"  Min    : {power_in_watts['min']:.3f}")
        _log.info(f"  Max    : {power_in_watts['max']:.3f}")

    _log.info("=" * 40)


def _build(args: SimpleNamespace) -> None:
    if args.int8:
        _log.warning("Build API is unstable and experimental with INT8 quantization.")

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
        log_level=args.log_level,
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
        int8=args.int8,
        fp16=args.fp16,
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
    _log.info(
        f"ONNX: {args.onnx}, Fully DLA Compatible: {full_dla}, Layers: {compat_layers} / {all_layers} ({portion_compat} % Compatible)"
    )


def _build_dla(args: SimpleNamespace) -> None:
    dtype = np.float32
    if args.dtype == "float16":
        dtype = np.float16
    elif args.dtype == "int8":
        dtype = np.int8

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
        warmup=args.warmup,
        verbose=args.verbose,
    )

    if is_image:
        img = cv2.imread(str(input_path))
        if img is None:
            err_msg = f"Failed to read image: {input_path}"
            raise ValueError(err_msg)

        dets = yolo.end2end(img)
        bboxes = [d[0] for d in dets]
        scores = [d[1] for d in dets]
        classes = [d[2] for d in dets]

        canvas = cv2ext.bboxes.draw_bboxes(img, bboxes, scores, classes)
        cv2.imshow("YOLO", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif is_video:
        with cv2ext.Display(f"YOLO detection: {input_path.stem}") as display:
            for _, frame in cv2ext.IterableVideo(input_path):
                if display.stopped:
                    break

                dets = yolo.end2end(frame)
                bboxes = [d[0] for d in dets]
                scores = [d[1] for d in dets]
                classes = [d[2] for d in dets]

                canvas = cv2ext.bboxes.draw_bboxes(frame, bboxes, scores, classes)
                display.update(canvas)

    else:
        err_msg = f"Invalid input file: {input_path}"
        raise ValueError(err_msg)


def _inspect(args: SimpleNamespace) -> None:
    engine_size, max_batch, inputs, outputs = trtutils.inspect.inspect_engine(
        Path(args.engine)
    )
    _log.info(f"Engine Size: {engine_size / (1024 * 1024):.2f} MB")
    _log.info(f"Max Batch Size: {max_batch}")
    _log.info("Inputs:")
    for name, shape, dtype in inputs:
        _log.info(f"\t{name}: shape={shape}, dtype={dtype}")
    _log.info("Outputs:")
    for name, shape, dtype in outputs:
        _log.info(f"\t{name}: shape={shape}, dtype={dtype}")


def _main() -> None:
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
        "--timing_cache",
        "-tc",
        default=None,
        help="Path to store timing cache data. Default is 'timing.cache'.",
    )
    build_parser.add_argument(
        "--log_level",
        "-ll",
        type=int,
        help="Log level to use if the logger is None. Default is WARNING.",
    )
    build_parser.add_argument(
        "--workspace",
        "-w",
        type=float,
        default=4.0,
        help="Workspace size in GB. Default is 4.0.",
    )
    build_parser.add_argument(
        "--dla_core",
        type=int,
        help="Specify the DLA core. By default, the engine is built for GPU.",
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
    build_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output from can_run_on_dla.",
    )
    build_parser.set_defaults(func=_build)

    # run_on_dla parser
    can_run_on_dla_parser = subparsers.add_parser(
        "can_run_on_dla",
        help="Evaluate if the model can run on a DLA.",
    )
    can_run_on_dla_parser.add_argument(
        "--onnx",
        "-o",
        required=True,
        help="Path to the ONNX model file.",
    )
    can_run_on_dla_parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 precision to assess DLA compatibility.",
    )
    can_run_on_dla_parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision to assess DLA compatibility.",
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
        "--dla_core",
        type=int,
        default=0,
        help="Specify the DLA core. By default, the engine is built for GPU.",
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
        "--verbose",
        action="store_true",
        help="Verbose output from can_run_on_dla.",
    )
    build_dla_parser.set_defaults(func=_build_dla)

    # yolo parser
    yolo_parser = subparsers.add_parser(
        "yolo",
        help="Run YOLO object detection on an image or video.",
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
        choices=["cpu", "cuda"],
        default="cuda",
        help="Preprocessor to use. Default is cuda.",
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
        "--verbose",
        action="store_true",
        help="Output additional debugging information.",
    )
    yolo_parser.set_defaults(func=_run_yolo)

    # inspect parser
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a TensorRT engine.",
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
    if hasattr(args, "func"):
        if args.command == "trtexec":
            args.func(unknown)
        else:
            args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    trtutils.set_log_level("INFO")
    _main()
