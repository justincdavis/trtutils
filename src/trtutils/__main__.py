# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Main entry point for trtutils CLI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import cv2ext
import numpy as np
from typing_extensions import TypeGuard

import trtutils
from trtutils._log import LOG
from trtutils.core import cache as caching_tools
from trtutils.trtexec._cli import cli_trtexec

if TYPE_CHECKING:
    from types import SimpleNamespace

    from ._benchmark import Metric


def _is_raw_outputs(
    outputs: list[np.ndarray] | list[list[np.ndarray]],
) -> TypeGuard[list[np.ndarray]]:
    return not outputs or isinstance(outputs[0], np.ndarray)


def _benchmark(args: SimpleNamespace) -> None:
    """
    Benchmark a TensorRT engine.

    Measures and reports performance metrics for a TensorRT engine, including latency,
    and optionally energy consumption and power draw on Jetson devices.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:
        - engine : str
            Path to the TensorRT engine file.
        - iterations : int
            Number of iterations to benchmark.
        - warmup_iterations : int
            Number of warmup iterations.
        - jetson : bool
            Whether to use Jetson-specific benchmarking for energy/power metrics.
        - tegra_interval : int
            Milliseconds between tegrastats samples (Jetson only).
        - dla_core : int or None
            DLA core to use, if any.
        - warmup : bool
            Whether to perform warmup.
        - verbose : bool
            Enable verbose output.

    Raises
    ------
    FileNotFoundError
        If the engine file does not exist.

    """
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
            tegra_interval=args.tegra_interval,
            dla_core=args.dla_core,
            warmup=True,
            verbose=args.verbose,
        )
        latency = jresult.latency  # type: ignore[assignment]
        energy = jresult.energy  # type: ignore[assignment]
        power = jresult.power_draw  # type: ignore[assignment]
    else:
        result = trtutils.benchmark_engine(
            engine=mpath,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            dla_core=args.dla_core,
            warmup=True,
            verbose=args.verbose,
        )
        latency = result.latency  # type: ignore[assignment]

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


def _parse_shapes_arg(
    shape_args: list[str] | None,
) -> list[tuple[str, tuple[int, ...]]] | None:
    """
    Parse shape specification arguments from the command line.

    Converts string specifications in the format "NAME:dim1,dim2,dim3,..." into
    a list of tuples containing the binding name and its dimensions.

    Parameters
    ----------
    shape_args : list[str] or None
        List of shape specifications in format "NAME:dim1,dim2[,dim3...]".
        Example: ["input:1,3,224,224", "mask:1,1,224,224"]

    Returns
    -------
    list[tuple[str, tuple[int, ...]]] or None
        List of (name, dimensions) tuples, or None if shape_args is empty.

    Raises
    ------
    ValueError
        If shape specification is invalid or dimensions cannot be parsed.

    """
    if not shape_args:
        return None
    parsed: list[tuple[str, tuple[int, ...]]] = []
    for spec in shape_args:
        if ":" not in spec:
            err_msg = "Invalid --shapes specification. Expected NAME:dim1,dim2[,dim3...]"
            raise ValueError(err_msg)
        name, dims_str = spec.split(":", 1)
        try:
            dims = tuple(int(x) for x in dims_str.split(",") if x)
        except ValueError as exc:
            err_msg = f"Invalid dimension in --shapes: {dims_str}"
            raise ValueError(err_msg) from exc
        if len(dims) == 0:
            err_msg = "No dimensions provided in --shapes"
            raise ValueError(err_msg)
        parsed.append((name, dims))
    return parsed


def _build(args: SimpleNamespace, *, add_yolo_hook: bool = False) -> None:
    """
    Build a TensorRT engine from an ONNX model.

    Creates a TensorRT engine with specified precision (FP32, FP16, INT8) and
    optimization settings. Optionally includes YOLO-specific NMS hooks.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:
        - onnx : str
            Path to the ONNX model file.
        - output : str
            Path to save the TensorRT engine.
        - int8 : bool
            Enable INT8 quantization.
        - fp16 : bool
            Enable FP16 precision.
        - calibration_dir : str or None
            Directory with calibration images for INT8.
        - input_shape : tuple[int, int, int] or None
            Input shape in HWC format for calibration.
        - input_dtype : str or None
            Input data type for calibration.
        - batch_size : int
            Batch size for calibration.
        - data_order : str
            Data ordering (NCHW or NHWC).
        - max_images : int or None
            Maximum images for calibration.
        - resize_method : str
            Image resize method (letterbox or linear).
        - input_scale : list[float]
            Input value range scaling.
        - shape : list[str] or None
            Fixed input shapes specification.
        - timing_cache : str or None
            Path to timing cache file.
        - workspace : float
            Workspace size in GB.
        - dla_core : int or None
            DLA core to use.
        - calibration_cache : str or None
            Path to calibration cache.
        - gpu_fallback : bool
            Allow GPU fallback for DLA.
        - direct_io : bool
            Use direct IO.
        - prefer_precision_constraints : bool
            Prefer precision constraints.
        - reject_empty_algorithms : bool
            Reject empty algorithms.
        - ignore_timing_mismatch : bool
            Ignore timing cache mismatches.
        - cache : bool
            Cache the engine.
        - verbose : bool
            Enable verbose output.
    add_yolo_hook : bool, optional
        Whether to add YOLO NMS hook, by default False.

    Raises
    ------
    ValueError
        If required parameters are missing for INT8 calibration.

    """
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

        input_shape = args.input_shape
        input_dtype = args.input_dtype

        batcher = trtutils.builder.ImageBatcher(
            image_dir=args.calibration_dir,
            shape=input_shape,
            dtype=np.dtype(input_dtype).type,
            batch_size=args.batch_size,
            order=args.data_order,
            max_images=args.max_images,
            resize_method=args.resize_method,
            input_scale=tuple(args.input_scale) if args.input_scale is not None else None,  # type: ignore[arg-type]
            verbose=args.verbose,
        )

    # handle hooks
    hooks = []
    if add_yolo_hook:
        hooks.append(
            trtutils.builder.hooks.yolo_efficient_nms_hook(
                num_classes=args.num_classes,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                top_k=args.top_k,
                class_agnostic=args.class_agnostic,
                box_coding=args.box_coding,
            )
        )

    shapes = _parse_shapes_arg(getattr(args, "shape", None))

    # actual call
    trtutils.build_engine(
        onnx=Path(args.onnx),
        output=Path(args.output),
        timing_cache=args.timing_cache,
        workspace=args.workspace,
        dla_core=args.dla_core,
        calibration_cache=args.calibration_cache,
        data_batcher=batcher,
        shapes=shapes,
        optimization_level=args.optimization_level,
        gpu_fallback=args.gpu_fallback,
        direct_io=args.direct_io,
        prefer_precision_constraints=args.prefer_precision_constraints,
        reject_empty_algorithms=args.reject_empty_algorithms,
        ignore_timing_mismatch=args.ignore_timing_mismatch,
        fp16=args.fp16,
        int8=args.int8,
        hooks=hooks,
        cache=args.cache,
        verbose=args.verbose,
    )


def _build_yolo(args: SimpleNamespace) -> None:
    """
    Build a TensorRT engine from an ONNX model with YOLO NMS injection.

    Wrapper around _build that automatically injects efficient NMS operations
    for YOLO object detection models.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments (same as _build), plus YOLO-specific parameters:
        - num_classes : int
            Number of object classes.
        - conf_threshold : float
            Confidence threshold for NMS.
        - iou_threshold : float
            IOU threshold for NMS.
        - top_k : int
            Maximum number of boxes to keep.
        - box_coding : str
            Box encoding format ('corner' or 'center_size').
        - class_agnostic : bool
            Use class-agnostic NMS.

    """
    _build(args, add_yolo_hook=True)


def _can_run_on_dla(args: SimpleNamespace) -> None:
    """
    Evaluate if a model can run on DLA and report layer compatibility.

    Analyzes an ONNX model to determine which layers are compatible with
    NVIDIA Deep Learning Accelerator (DLA) and provides detailed statistics.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:
        - onnx : str
            Path to the ONNX model file.
        - verbose_layers : bool
            Print detailed per-layer compatibility information.
        - verbose_chunks : bool
            Print detailed chunk assignment information.

    """
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
    portion_compat = round((compat_layers / all_layers) * 100.0, 2) if all_layers else 0.0
    LOG.info(
        f"ONNX: {args.onnx}, Fully DLA Compatible: {full_dla}, Layers: {compat_layers} / {all_layers} ({portion_compat} % Compatible)"
    )


def _build_dla(args: SimpleNamespace) -> None:
    """
    Build a TensorRT engine for DLA with automatic mixed GPU/DLA layer assignments.

    Automatically analyzes the model and assigns compatible layers to DLA while
    keeping incompatible layers on GPU. Requires INT8 precision and calibration data.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:
        - onnx : str
            Path to the ONNX model file.
        - output : str
            Path to save the TensorRT engine.
        - calibration_dir : str
            Directory with calibration images (required).
        - input_shape : tuple[int, int, int]
            Input shape in HWC format (required).
        - input_dtype : str
            Input data type (required).
        - batch_size : int
            Batch size for calibration.
        - data_order : str
            Data ordering (NCHW or NHWC).
        - max_images : int or None
            Maximum images for calibration.
        - resize_method : str
            Image resize method.
        - input_scale : list[float]
            Input value range scaling.
        - dla_core : int or None
            DLA core to use (defaults to 0 if not provided).
        - max_chunks : int
            Maximum number of DLA chunks.
        - min_layers : int
            Minimum layers per chunk.
        - workspace : float
            Workspace size in GB.
        - calibration_cache : str or None
            Path to calibration cache.
        - timing_cache : str or None
            Path to timing cache.
        - shape : list[str] or None
            Fixed input shapes specification.
        - direct_io : bool
            Use direct IO.
        - prefer_precision_constraints : bool
            Prefer precision constraints.
        - reject_empty_algorithms : bool
            Reject empty algorithms.
        - ignore_timing_mismatch : bool
            Ignore timing cache mismatches.
        - cache : bool
            Cache the engine.
        - verbose : bool
            Enable verbose output.

    Raises
    ------
    ValueError
        If required DLA build parameters are missing.

    """
    # require calibration data for dla builds
    if args.calibration_dir is None:
        err_msg = "Calibration directory is required for DLA builds"
        raise ValueError(err_msg)
    if args.input_shape is None:
        err_msg = "Input shape is required for DLA builds"
        raise ValueError(err_msg)
    if args.input_dtype is None:
        err_msg = "Input dtype is required for DLA builds"
        raise ValueError(err_msg)

    input_shape = args.input_shape
    input_dtype = args.input_dtype

    # Set default dla_core to 0 if not provided
    if args.dla_core is None:
        args.dla_core = 0

    batcher = trtutils.builder.ImageBatcher(
        image_dir=args.calibration_dir,
        shape=input_shape,
        dtype=np.dtype(input_dtype).type,
        batch_size=args.batch_size,
        order=args.data_order,
        max_images=args.max_images,
        resize_method=args.resize_method,
        input_scale=tuple(args.input_scale) if args.input_scale is not None else None,  # type: ignore[arg-type]
        verbose=args.verbose,
    )

    shapes = _parse_shapes_arg(getattr(args, "shape", None))

    trtutils.builder.build_dla_engine(
        onnx=Path(args.onnx),
        output_path=Path(args.output),
        data_batcher=batcher,
        dla_core=args.dla_core,
        max_chunks=args.max_chunks,
        min_layers=args.min_layers,
        workspace=args.workspace,
        calibration_cache=args.calibration_cache,
        timing_cache=args.timing_cache,
        shapes=shapes,
        optimization_level=args.optimization_level,
        direct_io=args.direct_io,
        prefer_precision_constraints=args.prefer_precision_constraints,
        reject_empty_algorithms=args.reject_empty_algorithms,
        ignore_timing_mismatch=args.ignore_timing_mismatch,
        cache=args.cache,
        verbose=args.verbose,
    )


def _detect(args: SimpleNamespace) -> None:
    """
    Run object detection on images or videos using a TensorRT engine.

    Performs YOLO-style object detection on input images or videos and optionally
    displays results with bounding boxes and timing information.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:
        - engine : str
            Path to the TensorRT engine file.
        - input : str
            Path to input image or video file.
        - warmup_iterations : int
            Number of warmup iterations.
        - input_range : list[float]
            Input value range.
        - preprocessor : str
            Preprocessor type ('cpu', 'cuda', or 'trt').
        - resize_method : str
            Image resize method ('letterbox' or 'linear').
        - conf_thres : float
            Confidence threshold for detections.
        - nms_iou_thres : float
            IOU threshold for NMS.
        - dla_core : int or None
            DLA core to use.
        - warmup : bool
            Perform warmup iterations.
        - pagelocked_mem : bool
            Use pagelocked memory.
        - unified_mem : bool
            Use unified memory.
        - extra_nms : bool
            Apply additional CPU NMS.
        - agnostic_nms : bool
            Use class-agnostic NMS.
        - no_warn : bool
            Suppress warnings.
        - verbose : bool
            Enable verbose output.
        - show : bool
            Display results in window.

    Raises
    ------
    ValueError
        If input file is invalid or cannot be read.

    """
    img_extensions = [".jpg", ".jpeg", ".png"]
    video_extensions = [".mp4", ".avi", ".mov"]

    input_path = Path(args.input)
    is_image = input_path.suffix in img_extensions
    is_video = input_path.suffix in video_extensions
    if not is_image and not is_video:
        err_msg = f"Invalid input file: {input_path}"
        raise ValueError(err_msg)

    detector = trtutils.image.Detector(
        engine_path=args.engine,
        warmup_iterations=args.warmup_iterations,
        input_range=tuple(args.input_range) if args.input_range is not None else None,  # type: ignore[arg-type]
        preprocessor=args.preprocessor,
        resize_method=args.resize_method,
        conf_thres=args.conf_thres,
        nms_iou_thres=args.nms_iou_thres,
        dla_core=args.dla_core,
        warmup=args.warmup,
        pagelocked_mem=args.pagelocked_mem,
        unified_mem=args.unified_mem,
        extra_nms=args.extra_nms,
        agnostic_nms=args.agnostic_nms,
        no_warn=args.no_warn,
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
    ) -> tuple[list[tuple[tuple[int, int, int, int], float, int]], float, float, float, float]:
        t0 = time.perf_counter()
        tensor, ratios, pads = detector.preprocess([img], no_copy=True, verbose=args.verbose)
        t1 = time.perf_counter()
        results = detector.run(
            [tensor],
            preprocessed=True,
            postprocess=False,
            no_copy=True,
            verbose=args.verbose,
        )
        t2 = time.perf_counter()
        if not _is_raw_outputs(results):
            err_msg = "Expected raw detector outputs before postprocess."
            raise RuntimeError(err_msg)
        p_results = detector.postprocess(results, ratios, pads, no_copy=True, verbose=args.verbose)
        t3 = time.perf_counter()
        dets: list[tuple[tuple[int, int, int, int], float, int]] = detector.get_detections(
            p_results, verbose=args.verbose
        )[0]  # type: ignore[assignment]
        t4 = time.perf_counter()
        return (
            dets,
            float(round(1000 * (t1 - t0), 2)),
            float(round(1000 * (t2 - t1), 2)),
            float(round(1000 * (t3 - t2), 2)),
            float(round(1000 * (t4 - t3), 2)),
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

    if is_image:
        img = cv2.imread(str(input_path))
        if img is None:
            err_msg = f"Failed to read image: {input_path}"
            raise ValueError(err_msg)

        bboxes, scores, classes, pre_t, run_t, post_t, det_t = process_image(img)

        if args.show:
            canvas = draw(img, bboxes, scores, classes, pre_t, run_t, post_t, det_t)
            cv2.imshow("DETECT", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif is_video:
        display = cv2ext.Display(f"DETECT: {input_path.stem}") if args.show else None

        for fid, frame in cv2ext.IterableVideo(input_path):
            if display is not None and display.stopped:
                break

            LOG.info(f"Processing frame {fid}")
            bboxes, scores, classes, pre_t, run_t, post_t, det_t = process_image(frame)

            if args.show:
                canvas = draw(frame, bboxes, scores, classes, pre_t, run_t, post_t, det_t)
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


def _classify(args: SimpleNamespace) -> None:
    """
    Run image classification on an image using a TensorRT engine.

    Performs image classification and reports the predicted class with confidence
    score and timing information.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:
        - engine : str
            Path to the TensorRT engine file.
        - input : str
            Path to input image file.
        - warmup_iterations : int
            Number of warmup iterations.
        - input_range : list[float]
            Input value range.
        - preprocessor : str
            Preprocessor type ('cpu', 'cuda', or 'trt').
        - resize_method : str
            Image resize method ('letterbox' or 'linear').
        - dla_core : int or None
            DLA core to use.
        - warmup : bool
            Perform warmup iterations.
        - pagelocked_mem : bool
            Use pagelocked memory.
        - unified_mem : bool
            Use unified memory.
        - no_warn : bool
            Suppress warnings.
        - verbose : bool
            Enable verbose output.
        - show : bool
            Display results in window.

    Raises
    ------
    ValueError
        If input file is invalid or cannot be read.

    """
    img_extensions = [".jpg", ".jpeg", ".png"]

    input_path = Path(args.input)
    is_image = input_path.suffix in img_extensions
    if not is_image:
        err_msg = f"Invalid input file: {input_path}"
        raise ValueError(err_msg)

    classifier = trtutils.image.Classifier(
        engine_path=args.engine,
        warmup_iterations=args.warmup_iterations,
        input_range=tuple(args.input_range) if args.input_range is not None else None,  # type: ignore[arg-type]
        preprocessor=args.preprocessor,
        resize_method=args.resize_method,
        dla_core=args.dla_core,
        warmup=args.warmup,
        pagelocked_mem=args.pagelocked_mem,
        unified_mem=args.unified_mem,
        no_warn=args.no_warn,
        verbose=args.verbose,
    )
    times: dict[str, list[float]] = {
        "pre": [],
        "run": [],
        "post": [],
        "cls": [],
    }

    def run(
        img: np.ndarray,
    ) -> tuple[tuple[int, float], float, float, float, float]:
        t0 = time.perf_counter()
        tensor, _, _ = classifier.preprocess([img], no_copy=True)
        t1 = time.perf_counter()
        results = classifier.run([tensor], preprocessed=True, postprocess=False, no_copy=True)
        t2 = time.perf_counter()
        if not _is_raw_outputs(results):
            err_msg = "Expected raw classifier outputs before postprocess."
            raise RuntimeError(err_msg)
        p_results = classifier.postprocess(results, no_copy=True)
        t3 = time.perf_counter()
        cls_results: list[tuple[int, float]] = classifier.get_classifications(p_results, top_k=1)[0]  # type: ignore[assignment]
        t4 = time.perf_counter()
        return (
            cls_results[0],
            float(round(1000 * (t1 - t0), 2)),
            float(round(1000 * (t2 - t1), 2)),
            float(round(1000 * (t3 - t2), 2)),
            float(round(1000 * (t4 - t3), 2)),
        )

    def log(
        cls_results: tuple[int, float],
        pre_t: float,
        run_t: float,
        post_t: float,
        det_t: float,
    ) -> None:
        LOG.info(f"Found {cls_results[0]} with confidence {cls_results[1]}")
        LOG.info(f"Preprocessing time: {pre_t} ms")
        LOG.info(f"Run time: {run_t} ms")
        LOG.info(f"Postprocessing time: {post_t} ms")
        LOG.info(f"Detection time: {det_t} ms")

    def process_image(
        img: np.ndarray,
    ) -> tuple[
        tuple[int, float],
        float,
        float,
        float,
        float,
    ]:
        cls_results, pre_t, run_t, post_t, det_t = run(img)
        log(cls_results, pre_t, run_t, post_t, det_t)
        times["pre"].append(pre_t)
        times["run"].append(run_t)
        times["post"].append(post_t)
        times["det"].append(det_t)
        return cls_results, pre_t, run_t, post_t, det_t

    def draw(
        img: np.ndarray,
        cls_results: tuple[int, float],
        pre_t: float,
        run_t: float,
        post_t: float,
        det_t: float,
    ) -> np.ndarray:
        canvas = cv2ext.image.draw.text(img, f"CLASS: {cls_results[0]}", (50, 30))
        canvas = cv2ext.image.draw.text(canvas, f"CONF: {cls_results[1]:.2f}", (50, 60))
        canvas = cv2ext.image.draw.text(canvas, f"PRE:  {pre_t} ms", (10, 30))
        canvas = cv2ext.image.draw.text(canvas, f"RUN:  {run_t} ms", (10, 60))
        canvas = cv2ext.image.draw.text(canvas, f"POST: {post_t} ms", (10, 90))
        return cv2ext.image.draw.text(canvas, f"CLS:  {det_t} ms", (10, 120))

    if is_image:
        img = cv2.imread(str(input_path))
        if img is None:
            err_msg = f"Failed to read image: {input_path}"
            raise ValueError(err_msg)

        cls_results, pre_t, run_t, post_t, det_t = process_image(img)

        if args.show:
            canvas = draw(img, cls_results, pre_t, run_t, post_t, det_t)
            cv2.imshow("CLASSIFY", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        err_msg = f"Invalid input file: {input_path}"
        raise ValueError(err_msg)

    LOG.info("Times:")
    for k, v in times.items():
        LOG.info(f"{k}: {np.mean(v):.2f} ms")


def _inspect(args: SimpleNamespace) -> None:
    """
    Inspect a TensorRT engine and report its properties.

    Displays information about the engine including size, batch configuration,
    and input/output tensor specifications.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:
        - engine : str
            Path to the TensorRT engine file.
        - verbose : bool
            Enable verbose output.

    """
    engine_size, max_batch, inputs, outputs = trtutils.inspect.inspect_engine(
        Path(args.engine),
        verbose=args.verbose,
    )
    LOG.info(f"Engine Size: {engine_size / (1024 * 1024):.2f} MB")
    LOG.info(f"Max Batch Size: {max_batch}")
    LOG.info("Inputs:")
    for name, shape, dtype, fmt in inputs:
        LOG.info(f"\t{name}: shape={shape}, dtype={dtype}, format={fmt}")
    LOG.info("Outputs:")
    for name, shape, dtype, fmt in outputs:
        LOG.info(f"\t{name}: shape={shape}, dtype={dtype}, format={fmt}")


def _profile(args: SimpleNamespace) -> None:
    """
    Profile a TensorRT engine layer-by-layer.

    Runs inference multiple times and collects per-layer execution times,
    then writes the results to a JSON file and optionally displays verbose output.
    On Jetson devices with --jetson flag, also measures power and energy.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:
        - engine : str
            Path to the TensorRT engine file.
        - output : str
            Path to save the JSON profiling results.
        - iterations : int
            Number of profiling iterations to run.
        - warmup_iterations : int
            Number of warmup iterations before profiling.
        - dla_core : int or None
            DLA core to use, if any.
        - warmup : bool
            Whether to perform warmup iterations.
        - verbose : bool
            Enable verbose output.
        - save_raw : bool
            Whether to include raw timing values in the JSON output.
        - jetson : bool
            Whether to use Jetson-specific profiling with power/energy metrics.
        - tegra_interval : int
            Milliseconds between tegrastats samples (Jetson only).

    Raises
    ------
    FileNotFoundError
        If the engine file does not exist.

    """
    engine_path = Path(args.engine)
    if not engine_path.exists():
        err_msg = f"Cannot find provided engine: {engine_path}"
        raise FileNotFoundError(err_msg)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = trtutils.profile_engine(
        engine=engine_path,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        dla_core=args.dla_core,
        tegra_interval=args.tegra_interval if hasattr(args, "tegra_interval") else 5,
        jetson=args.jetson if hasattr(args, "jetson") else False,
        warmup=args.warmup,
        verbose=args.verbose,
    )

    total_time_dict = {
        "name": result.total_time.name,
        "mean": result.total_time.mean,
        "median": result.total_time.median,
        "min": result.total_time.min,
        "max": result.total_time.max,
    }
    if args.save_raw:
        total_time_dict["raw"] = result.total_time.raw

    layer_dicts = []
    for layer in result.layers:
        layer_dict = {
            "name": layer.name,
            "mean": layer.mean,
            "median": layer.median,
            "min": layer.min,
            "max": layer.max,
        }
        if args.save_raw:
            layer_dict["raw"] = layer.raw
        layer_dicts.append(layer_dict)

    json_data = {
        "iterations": result.iterations,
        "total_time": total_time_dict,
        "layers": layer_dicts,
    }

    # Add power and energy data if this is a Jetson result
    power_draw = getattr(result, "power_draw", None)
    energy = getattr(result, "energy", None)
    if power_draw is not None and energy is not None:
        power_dict = {
            "mean": power_draw.mean,
            "median": power_draw.median,
            "min": power_draw.min,
            "max": power_draw.max,
        }
        energy_dict = {
            "mean": energy.mean,
            "median": energy.median,
            "min": energy.min,
            "max": energy.max,
        }
        if args.save_raw:
            power_dict["raw"] = power_draw.raw
            energy_dict["raw"] = energy.raw

        json_data["power_draw"] = power_dict
        json_data["energy"] = energy_dict

    # Write JSON output
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    # Log summary
    LOG.info(f"Profiling result for: {engine_path.stem}")
    LOG.info("=" * 40)
    LOG.info("Total Time (ms):")
    LOG.info(f"  Mean   : {result.total_time.mean:.3f}")
    LOG.info(f"  Median : {result.total_time.median:.3f}")
    LOG.info(f"  Min    : {result.total_time.min:.3f}")
    LOG.info(f"  Max    : {result.total_time.max:.3f}")
    LOG.info(f"Layers profiled: {len(result.layers)}")

    # Log power and energy if available
    if power_draw is not None and energy is not None:
        LOG.info("=" * 40)
        LOG.info("Power Draw (mW):")
        LOG.info(f"  Mean   : {power_draw.mean:.1f}")
        LOG.info(f"  Median : {power_draw.median:.1f}")
        LOG.info(f"  Min    : {power_draw.min:.1f}")
        LOG.info(f"  Max    : {power_draw.max:.1f}")
        LOG.info("Energy (mJ):")
        LOG.info(f"  Mean   : {energy.mean:.3f}")
        LOG.info(f"  Median : {energy.median:.3f}")
        LOG.info(f"  Min    : {energy.min:.3f}")
        LOG.info(f"  Max    : {energy.max:.3f}")

    LOG.info(f"Results written to: {output_path}")

    # Verbose output: show per-layer timings
    if args.verbose:
        LOG.info("=" * 40)
        LOG.info("Layer timings (sorted by mean time, descending):")
        for layer in result.layers:
            LOG.info(f"  {layer.name}:")
            LOG.info(f"    Mean   : {layer.mean:.3f} ms")
            LOG.info(f"    Median : {layer.median:.3f} ms")
            LOG.info(f"    Min    : {layer.min:.3f} ms")
            LOG.info(f"    Max    : {layer.max:.3f} ms")

    LOG.info("=" * 40)


def _download(args: SimpleNamespace) -> None:
    """
    Download a model from a remote source and convert it to ONNX format.

    Downloads pre-trained models and automatically converts them to ONNX format
    for use with TensorRT. Includes license acceptance flow for model usage.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:
        - model : str
            Name of the model to download.
        - output : Path
            Path to save the ONNX model file.
        - opset : int
            ONNX opset version to use.
        - imgsz : int
            Image size for the model.
        - requirements_export : Path | None
            Optional path to export the created virtual environment's requirements file.
        - accept : bool
            Whether to accept license terms automatically.
        - verbose : bool
            Enable verbose output.
        - no_cache : bool
            Disable caching of downloaded weights, repos, and uv packages.

    """
    if args.list_models:
        configs = trtutils.download.load_model_configs()
        LOG.info("Supported models:\n")
        for family in sorted(configs):
            model_names = sorted(configs[family].keys())
            LOG.info(f"  {family}:")
            LOG.info(f"    {', '.join(model_names)}")
            LOG.info("")
        return

    if args.model is None or args.output is None:
        LOG.error("--model and --output are required for downloading.")
        return

    if not args.accept:
        LOG.info(
            f"You are about to download model '{args.model}' which may have license restrictions."
        )
        LOG.info("Please ensure you comply with the model's license terms.")
        response = input("Do you accept the license terms? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            LOG.info("License not accepted. Aborting download.")
            return

    trtutils.download.download(
        args.model,
        args.output,
        args.opset,
        args.imgsz,
        requirements_export=args.requirements_export,
        accept=True,
        verbose=args.verbose,
        no_cache=args.no_cache,
        no_uv_cache=args.no_cache,
    )


def _clear_cache(args: SimpleNamespace) -> None:
    """
    Clear the trtutils engine cache.

    Removes all cached TensorRT engines from the trtutils cache directory.

    Parameters
    ----------
    args : SimpleNamespace
        Command-line arguments containing:
        - no_warn : bool
            Whether to suppress the warning about clearing the cache.

    """
    caching_tools.clear(no_warn=args.no_warn)
    LOG.info("Engine cache cleared.")


def _main() -> None:
    """
    Set up the main entry point for the trtutils command-line interface.

    Sets up argument parsing for all CLI commands, processes arguments,
    configures logging, and dispatches to the appropriate command handler.

    Commands
    --------
    benchmark
        Benchmark a TensorRT engine for performance metrics.
    trtexec
        Run trtexec with provided options.
    build
        Build a TensorRT engine from an ONNX model.
    build_yolo
        Build a TensorRT engine with YOLO NMS injection.
    can_run_on_dla
        Evaluate DLA compatibility of a model.
    build_dla
        Build an engine with automatic DLA layer assignments.
    detect
        Run object detection on images or videos.
    classify
        Run image classification on images.
    inspect
        Inspect a TensorRT engine's properties.
    profile
        Profile a TensorRT engine layer-by-layer.
    download
        Download and convert models to ONNX.
    """
    # general arguments parser (for all commands)
    general_parser = argparse.ArgumentParser(add_help=False)
    general_parser.add_argument(
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
    general_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    # dla arguments parser (for commands that support DLA)
    dla_parser = argparse.ArgumentParser(add_help=False)
    dla_parser.add_argument(
        "--dla_core",
        type=int,
        default=None,
        help="DLA core to assign DLA layers of the engine to. Default is None.",
    )

    # shared build arguments parser
    build_common_parser = argparse.ArgumentParser(add_help=False)
    build_common_parser.add_argument(
        "--onnx",
        "-o",
        required=True,
        help="Path to the ONNX model file.",
    )
    build_common_parser.add_argument(
        "--output",
        "-out",
        required=True,
        help="Path to save the TensorRT engine file.",
    )
    build_common_parser.add_argument(
        "--timing_cache",
        "-tc",
        default=None,
        help=(
            "Path to store timing cache data, or 'global' to use the global "
            "timing cache stored in the trtutils cache directory. Default is None."
        ),
    )
    build_common_parser.add_argument(
        "--calibration_cache",
        "-cc",
        default=None,
        help="Path to store calibration cache data. Default is None.",
    )
    build_common_parser.add_argument(
        "--workspace",
        "-w",
        type=float,
        default=4.0,
        help="Workspace size in GB. Default is 4.0.",
    )
    build_common_parser.add_argument(
        "--optimization_level",
        type=int,
        choices=range(6),
        default=3,
        help="TensorRT builder optimization level (0-5). Default is 3.",
    )
    build_common_parser.add_argument(
        "--shape",
        "-s",
        action="append",
        default=None,
        help=(
            "Fix input binding shapes. Format NAME:dim1,dim2[,dim3...]. "
            "Repeat --shapes for multiple inputs."
        ),
    )
    build_common_parser.add_argument(
        "--direct_io",
        action="store_true",
        help="Use direct IO for the engine.",
    )
    build_common_parser.add_argument(
        "--prefer_precision_constraints",
        action="store_true",
        help="Prefer precision constraints.",
    )
    build_common_parser.add_argument(
        "--reject_empty_algorithms",
        action="store_true",
        help="Reject empty algorithms.",
    )
    build_common_parser.add_argument(
        "--ignore_timing_mismatch",
        action="store_true",
        help="Allow different CUDA device timing caches to be used.",
    )
    build_common_parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache the engine in the trtutils engine cache.",
    )

    # calibration arguments parser
    calibration_parser = argparse.ArgumentParser(add_help=False)
    calibration_parser.add_argument(
        "--calibration_dir",
        "-cd",
        default=None,
        help="Directory containing images for INT8 calibration.",
    )
    calibration_parser.add_argument(
        "--input_shape",
        "-is",
        type=int,
        nargs=3,
        help="Input shape in HWC format (height, width, channels). Required when using calibration directory.",
    )
    calibration_parser.add_argument(
        "--input_dtype",
        "-id",
        choices=["float32", "float16", "int8"],
        help="Input data type. Required when using calibration directory.",
    )
    calibration_parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=8,
        help="Batch size for calibration. Default is 8.",
    )
    calibration_parser.add_argument(
        "--data_order",
        "-do",
        choices=["NCHW", "NHWC"],
        default="NCHW",
        help="Data ordering expected by the network. Default is NCHW.",
    )
    calibration_parser.add_argument(
        "--max_images",
        "-mi",
        type=int,
        help="Maximum number of images to use for calibration.",
    )
    calibration_parser.add_argument(
        "--resize_method",
        "-rm",
        choices=["letterbox", "linear"],
        default="letterbox",
        help="Method to resize images. Default is letterbox.",
    )
    calibration_parser.add_argument(
        "--input_scale",
        "-sc",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="Input value range. Default is [0.0, 1.0].",
    )

    # warmup arguments parser
    warmup_parser = argparse.ArgumentParser(add_help=False)
    warmup_parser.add_argument(
        "--warmup",
        action="store_true",
        help="Perform warmup iterations.",
    )
    warmup_parser.add_argument(
        "--warmup_iterations",
        "-wi",
        type=int,
        default=10,
        help="Number of warmup iterations. Default is 10.",
    )

    # memory arguments parser
    memory_parser = argparse.ArgumentParser(add_help=False)
    memory_parser.add_argument(
        "--pagelocked_mem",
        action="store_true",
        help="Use pagelocked memory for CUDA operations.",
    )
    memory_parser.add_argument(
        "--unified_mem",
        action="store_true",
        help="Use unified memory for CUDA operations.",
    )
    memory_parser.add_argument(
        "--no_warn",
        action="store_true",
        help="Suppress warnings from TensorRT.",
    )

    # build device/precision args (to be inherited by build and build_yolo)
    build_device_parser = argparse.ArgumentParser(add_help=False)
    build_device_parser.add_argument(
        "--device",
        "-d",
        choices=["gpu", "dla", "GPU", "DLA"],
        default="gpu",
        help="Device to use for the engine. Default is 'gpu'.",
    )
    build_device_parser.add_argument(
        "--gpu_fallback",
        action="store_true",
        help="Allow GPU fallback for unsupported layers when building for DLA.",
    )
    build_device_parser.add_argument(
        "--fp16",
        action="store_true",
        help="Quantize the engine to FP16 precision.",
    )
    build_device_parser.add_argument(
        "--int8",
        action="store_true",
        help="Quantize the engine to INT8 precision.",
    )

    # main parser
    parser = argparse.ArgumentParser(
        description="Utilities for TensorRT.",
        parents=[general_parser],
    )

    # create subparser for each command
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        required=False,
    )

    # benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark a given TensorRT engine.",
        parents=[general_parser, dla_parser, warmup_parser],
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
        "--jetson",
        "-j",
        action="store_true",
        help="If True, will use the trtutils.jetson submodule benchmarker to record energy and power draw as well.",
    )
    benchmark_parser.add_argument(
        "--tegra_interval",
        type=int,
        default=5,
        help="Milliseconds between each tegrastats sampling for Jetson benchmarking. Default is 5.",
    )
    benchmark_parser.set_defaults(func=_benchmark)

    # trtexec parser
    trtexec_parser = subparsers.add_parser(
        "trtexec",
        help="Run trtexec.",
        parents=[general_parser],
    )
    trtexec_parser.set_defaults(func=cli_trtexec)

    # build_engine parser
    build_parser = subparsers.add_parser(
        "build",
        help="Build a TensorRT engine from an ONNX model.",
        parents=[
            general_parser,
            dla_parser,
            build_common_parser,
            calibration_parser,
            build_device_parser,
        ],
    )
    build_parser.set_defaults(func=_build)

    # build_yolo parser
    build_yolo_parser = subparsers.add_parser(
        "build_yolo",
        help="Build a TensorRT engine from an ONNX model and inject NMS.",
        parents=[
            general_parser,
            dla_parser,
            build_common_parser,
            calibration_parser,
            build_device_parser,
        ],
    )
    build_yolo_parser.add_argument(
        "--num_classes",
        type=int,
        default=80,
        help="Number of classes for NMS. Default is 80.",
    )
    build_yolo_parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Score threshold for NMS. Default is 0.25.",
    )
    build_yolo_parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IOU threshold for NMS. Default is 0.5.",
    )
    build_yolo_parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Top-k boxes for NMS. Default is 100.",
    )
    build_yolo_parser.add_argument(
        "--box_coding",
        choices=["corner", "center_size"],
        default="center_size",
        help="Box coding for TRT EfficientNMS. 'corner' or 'center_size'. Default is center_size.",
    )
    build_yolo_parser.add_argument(
        "--class_agnostic",
        action="store_true",
        help="Use class-agnostic NMS.",
    )
    build_yolo_parser.set_defaults(func=_build_yolo)

    # can_run_on_dla parser
    can_run_on_dla_parser = subparsers.add_parser(
        "can_run_on_dla",
        help="Evaluate if the model can run on a DLA.",
        parents=[general_parser],
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
        help="Build a TensorRT engine for DLA with automatic layer assignments.",
        parents=[general_parser, dla_parser, build_common_parser, calibration_parser],
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
    build_dla_parser.set_defaults(func=_build_dla)

    # detect parser
    detect_parser = subparsers.add_parser(
        "detect",
        help="Run YOLO object detection on an image or video.",
        parents=[general_parser, dla_parser, warmup_parser, memory_parser],
    )
    detect_parser.add_argument(
        "--engine",
        "-e",
        required=True,
        help="Path to the TensorRT engine file.",
    )
    detect_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input image or video file.",
    )
    detect_parser.add_argument(
        "--conf_thres",
        "-c",
        type=float,
        default=0.1,
        help="Confidence threshold for detections. Default is 0.1.",
    )
    detect_parser.add_argument(
        "--nms_iou_thres",
        type=float,
        default=0.5,
        help="NMS IOU threshold for detections. Default is 0.5.",
    )
    detect_parser.add_argument(
        "--input_range",
        "-r",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="Input value range. Default is [0.0, 1.0].",
    )
    detect_parser.add_argument(
        "--preprocessor",
        "-p",
        choices=["cpu", "cuda", "trt"],
        default="trt",
        help="Preprocessor to use. Default is trt.",
    )
    detect_parser.add_argument(
        "--resize_method",
        "-rm",
        choices=["letterbox", "linear"],
        default="letterbox",
        help="Method to resize images. Default is letterbox.",
    )
    detect_parser.add_argument(
        "--extra_nms",
        action="store_true",
        help="Perform additional CPU-side NMS.",
    )
    detect_parser.add_argument(
        "--agnostic_nms",
        action="store_true",
        help="Perform class-agnostic NMS.",
    )
    detect_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the detections.",
    )
    detect_parser.set_defaults(func=_detect)

    # classify parser
    classify_parser = subparsers.add_parser(
        "classify",
        help="Run image classification on an image.",
        parents=[general_parser, dla_parser, warmup_parser, memory_parser],
    )
    classify_parser.add_argument(
        "--engine",
        "-e",
        required=True,
        help="Path to the TensorRT engine file.",
    )
    classify_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input image file.",
    )
    classify_parser.add_argument(
        "--input_range",
        "-r",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="Input value range. Default is [0.0, 1.0].",
    )
    classify_parser.add_argument(
        "--preprocessor",
        "-p",
        choices=["cpu", "cuda", "trt"],
        default="trt",
        help="Preprocessor to use. Default is trt.",
    )
    classify_parser.add_argument(
        "--show",
        action="store_true",
        help="Show the classifications.",
    )
    classify_parser.set_defaults(func=_classify)

    # inspect parser
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a TensorRT engine.",
        parents=[general_parser],
    )
    inspect_parser.add_argument(
        "--engine",
        "-e",
        required=True,
        help="Path to the TensorRT engine file.",
    )
    inspect_parser.set_defaults(func=_inspect)

    # profile parser
    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile a TensorRT engine layer-by-layer.",
        parents=[general_parser, dla_parser, warmup_parser],
    )
    profile_parser.add_argument(
        "--engine",
        "-e",
        required=True,
        help="Path to the TensorRT engine file.",
    )
    profile_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to save the JSON profiling results.",
    )
    profile_parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=100,
        help="Number of profiling iterations to run. Default is 100.",
    )
    profile_parser.add_argument(
        "--save_raw",
        action="store_true",
        help="Include raw timing values in the JSON output. Default is False.",
    )
    profile_parser.add_argument(
        "--jetson",
        action="store_true",
        help="Use Jetson-specific profiling to measure per-layer power and energy. Note: Jetson profiling defaults to 10000 iterations for better coverage.",
    )
    profile_parser.add_argument(
        "--tegra_interval",
        type=int,
        default=5,
        help="Milliseconds between tegrastats samples (Jetson only). Default is 5.",
    )
    profile_parser.set_defaults(func=_profile)

    # download parser
    download_parser = subparsers.add_parser(
        "download",
        help="Download a model from remote source and convert to ONNX.",
        parents=[general_parser],
    )
    download_parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="Name of the model to download.",
    )
    download_parser.add_argument(
        "--output",
        type=Path,
        required=False,
        default=None,
        help="Path to save the ONNX model file.",
    )
    download_parser.add_argument(
        "--list_models",
        action="store_true",
        help="List all supported models and exit.",
    )
    download_parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to use. Default is 17.",
    )
    download_parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size to use for the model. If not specified, uses the model's default.",
    )
    download_parser.add_argument(
        "--requirements_export",
        type=Path,
        default=None,
        help="Export the created virtual environment's requirements to this path using uv pip freeze.",
    )
    download_parser.add_argument(
        "--accept",
        action="store_true",
        help="Accept the license terms for the model. If not provided, you will be prompted.",
    )
    download_parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable caching of downloaded weights, repos, and uv packages.",
    )
    download_parser.set_defaults(func=_download)

    # clear_cache parser
    clear_cache_parser = subparsers.add_parser(
        "clear_cache",
        help="Clear the trtutils engine cache.",
        parents=[general_parser],
    )
    clear_cache_parser.add_argument(
        "--no_warn",
        action="store_true",
        help="Suppress the warning about clearing the cache.",
    )
    clear_cache_parser.set_defaults(func=_clear_cache)

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
