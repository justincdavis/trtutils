# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found]

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._calibrator import EngineCalibrator
from ._onnx import read_onnx
from ._utils import get_check_dla

with contextlib.suppress(AttributeError):
    from ._progress import ProgressBar

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._batcher import AbstractBatcher


def build_engine(
    onnx: Path | str,
    output: Path | str,
    default_device: trt.DeviceType | str = trt.DeviceType.GPU,
    timing_cache: Path | str | None = None,
    workspace: float = 4.0,
    dla_core: int | None = None,
    calibration_cache: Path | str | None = None,
    data_batcher: AbstractBatcher | None = None,
    layer_precision: list[tuple[int, trt.DataType | None]] | None = None,
    layer_device: list[tuple[int, trt.DeviceType | None]] | None = None,
    *,
    gpu_fallback: bool = False,
    direct_io: bool = False,
    prefer_precision_constraints: bool = False,
    reject_empty_algorithms: bool = False,
    ignore_timing_mismatch: bool = False,
    fp16: bool | None = None,
    int8: bool | None = None,
    verbose: bool | None = None,
) -> None:
    """
    Build a TensorRT engine from an ONNX model.

    Parameters
    ----------
    onnx : Path, str
        The path to the onnx model.
    output : Path, str
        The location to save the TensorRT engine.
    default_device : trt.DeviceType, str, optional
        The device to use for the engine.
        By default, trt.DeviceType.GPU.
        Options are trt.DeviceType.GPU, trt.DeviceType.DLA, or a string
        of "gpu" or "dla".
    timing_cache : Path, str, optional
        Where to store the timing cache data.
        Default is None.
    workspace : float
        The size of the workspace in gigabytes.
        Default is 4.0 GiB.
    calibration_cache : Path, str, optional
        The path to the calibration cache.
    data_batcher : AbstractBatcher, optional
        The data batcher to use for calibration.
    dla_core : int, optional
        The DLA core to build the engine for.
        By default, None or build the engine for GPU.
    layer_precision : list[tuple[int, trt.DataType | None]], optional
        The precision to use for specific layers.
        By default, None.
    layer_device : list[tuple[int, trt.DeviceType | None]], optional
        The device to use for specific layers.
        By default, None.
    gpu_fallback : bool
        Whether or not to allow GPU fallback for unsupported layers
        when building the engine for DLA.
        By default, False
    direct_io : bool
        Use direct IO for the engine.
        By default, False
    prefer_precision_constraints : bool
        Whether or not to prefer precision constraints.
        By default, False
    reject_empty_algorithms : bool
        Whether or not to reject empty algorithms.
        By default, False
    ignore_timing_mismatch : bool
        Whether or not to allow different CUDA device generated timing
        caches to be used in the building of engines.
        By default, False
    fp16 : bool, optional
        If True, quantize the engine to FP16 precision.
    int8 : bool, optional
        If True, quantize the engine to INT8 precision.
    verbose : bool, optional
        If True, print verbose output.
        By default, None or False

    Raises
    ------
    RuntimeError
        If the ONNX model cannot be parsed
    RuntimeError
        If the TensorRT engines fails to build
    ValueError
        If layer is manually assigned to DLA and DLA is not supported
        and gpu_fallback is False

    """
    # match the device
    valid_gpu = ["gpu", "GPU"]
    valid_dla = ["dla", "DLA"]
    if isinstance(default_device, str):
        if default_device not in valid_gpu + valid_dla:
            err_msg = f"Invalid default device: {default_device}. Must be one of: {valid_gpu + valid_dla}"
            raise ValueError(err_msg)
        default_device = (
            trt.DeviceType.GPU if default_device in valid_gpu else trt.DeviceType.DLA
        )
    else:
        if default_device not in [trt.DeviceType.GPU, trt.DeviceType.DLA]:
            err_msg = f"Invalid default device: {default_device}. Must be one of: {valid_gpu + valid_dla}"
            raise ValueError(err_msg)
        default_device = (
            trt.DeviceType.GPU
            if default_device == trt.DeviceType.GPU
            else trt.DeviceType.DLA
        )

    output_path = Path(output).resolve()

    # read the onnx model
    network, builder, config, _ = read_onnx(
        onnx,
        workspace,
    )

    # helper function for checking if layer can run on DLA
    check_dla: Callable[[trt.ILayer], bool] = get_check_dla(config)

    if verbose and FLAGS.BUILD_PROGRESS:
        LOG.debug("Applying ProgressBar to config")
        config.progress_monitor = ProgressBar()

    # create profile and config
    profile = builder.create_optimization_profile()
    config.add_optimization_profile(profile)

    # handle some flags
    if prefer_precision_constraints:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    if direct_io:
        config.set_flag(trt.BuilderFlag.DIRECT_IO)
    if reject_empty_algorithms:
        config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

    # load/setup the timing cache
    timing_cache_path: Path | None = (
        Path(timing_cache).resolve() if timing_cache else None
    )
    if timing_cache_path:
        buffer = b""
        if timing_cache_path.exists():
            with timing_cache_path.open("rb") as timing_cache_file:
                buffer = timing_cache_file.read()
        t_cache = config.create_timing_cache(buffer)
        config.set_timing_cache(t_cache, ignore_mismatch=ignore_timing_mismatch)

    # setup the precision sets
    if fp16 or int8:
        # want to enable fp16 for both int8 and fp16 since fp16 may be faster
        if not builder.platform_has_fast_fp16:
            LOG.warning("Platform does not have native fast FP16.")
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        if not builder.platform_has_fast_int8:
            LOG.warning("Platform does not have native fast INT8.")
        config.set_flag(trt.BuilderFlag.INT8)
        if calibration_cache is None and data_batcher is None:
            err_msg = "Neither calibration cache or data batcher passed during model building, INT8 build will not be accurate."
            LOG.warning(err_msg)
        config.int8_calibrator = EngineCalibrator(calibration_cache=calibration_cache)
        if data_batcher is not None:
            config.int8_calibrator.set_batcher(data_batcher)

    # assign the default device
    config.default_device_type = default_device

    # handle DLA assignment
    if dla_core is not None:
        config.DLA_core = dla_core
    if gpu_fallback:
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # handle individual layer precision
    if layer_precision is not None:
        # validate length
        if len(layer_precision) != network.num_layers:
            err_msg = "Layer precision list must be the same length as the number of layers in the network."
            raise ValueError(err_msg)
        # handle precision assignment
        for layer_idx, precision in layer_precision:
            if precision is None:
                continue
            layer = network.get_layer(layer_idx)
            layer.precision = precision

    # handle individual layer device
    if layer_device is not None:
        # validate length
        if len(layer_device) != network.num_layers:
            err_msg = "Layer device list must be the same length as the number of layers in the network."
            raise ValueError(err_msg)
        # handle device assignment
        for layer_idx, device in layer_device:
            if device is None:
                continue
            layer = network.get_layer(layer_idx)
            # assess if can run on DLA
            if device == trt.DeviceType.DLA and not check_dla(layer):
                err_msg = f"Layer {layer.name} (type: {layer.type}) cannot run on DLA"
                if gpu_fallback:
                    err_msg += ", using GPU fallback"
                    LOG.warning(err_msg)
                else:
                    raise ValueError(err_msg)
            else:
                config.set_device_type(layer, device)

    # build the engine
    if FLAGS.BUILD_SERIALIZED:
        engine_bytes = builder.build_serialized_network(network, config)
    else:
        engine_bytes = builder.build_engine(network, config)

    # save the timing cache
    if timing_cache_path:
        post_t_cache = config.get_timing_cache()
        with timing_cache_path.open("wb") as f:
            f.write(memoryview(post_t_cache.serialize()))

    if engine_bytes is None:
        err_msg = "Failed to build engine."
        raise RuntimeError(err_msg)

    with output_path.open("wb") as f:
        f.write(engine_bytes)
