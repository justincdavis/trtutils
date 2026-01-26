# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from trtutils._config import CONFIG
from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.compat._libs import trt
from trtutils.core import cache as caching_tools
from trtutils.core.cache import query_timing_cache, save_timing_cache_to_global

from ._calibrator import EngineCalibrator
from ._onnx import read_onnx
from ._utils import get_check_dla

ProgressBar = None
if FLAGS.BUILD_PROGRESS:
    from ._progress import ProgressBar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ._batcher import AbstractBatcher

_MIN_OPTIM_LEVEL = 0
_MAX_OPTIM_LEVEL = 5


def build_engine(
    onnx: Path | str,
    output: Path | str,
    default_device: trt.DeviceType | str = trt.DeviceType.GPU,
    timing_cache: Path | str | bool | None = None,  # noqa: FBT001
    workspace: float = 4.0,
    dla_core: int | None = None,
    calibration_cache: Path | str | None = None,
    data_batcher: AbstractBatcher | None = None,
    layer_precision: list[tuple[int, trt.DataType | None]] | None = None,
    layer_device: list[tuple[int, trt.DeviceType | None]] | None = None,
    shapes: Sequence[tuple[str, tuple[int, ...]]] | None = None,
    input_tensor_formats: list[tuple[str, trt.DataType, trt.TensorFormat]] | None = None,
    output_tensor_formats: list[tuple[str, trt.DataType, trt.TensorFormat]] | None = None,
    hooks: list[Callable[[trt.INetworkDefinition], trt.INetworkDefinition]] | None = None,
    optimization_level: int = 3,
    profiling_verbosity: trt.ProfilingVerbosity | None = None,
    tiling_optimization_level: trt.TilingOptimizationLevel | None = None,
    tiling_l2_cache_limit: int | None = None,
    *,
    gpu_fallback: bool = False,
    direct_io: bool = False,
    prefer_precision_constraints: bool = False,
    reject_empty_algorithms: bool = False,
    ignore_timing_mismatch: bool = False,
    fp16: bool | None = None,
    int8: bool | None = None,
    cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    """
    Build a TensorRT engine from an ONNX model.

    The order in which operations occur inside build_engine:

    1. Parse the ONNX model

    2. Apply any network hooks

    3. Create optimization profile and apply any manual shapes

    4. Apply builder flags (precision constraints, empty algorithms, direct I/O)

    5. Configure tensor formats if specified

    6. Configure precision (FP16, INT8)

    7. Set default device and DLA core

    8. Apply individual layer precision and device settings

    9. Set up timing cache

    10. Build the engine

    11. Save timing cache and engine

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
    timing_cache : Path, str, bool, optional
        Where to store the timing cache data.
        Can be a Path or str to a specific file, "global" or True to use
        the global timing cache stored in the trtutils cache directory,
        or None to not use a timing cache.
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
    shapes : list[tuple[str, tuple[int, ...]]], optional
        A list of (input_name, shape) pairs to specify the shapes of the input layers.
        For example, shapes=[("images", (1, 3, imgsz, imgsz))] will set the input
        “images” to a fixed shape. This shape will be used as the min, optimal,
        and max shape for the binding.
        By default, None.
    input_tensor_formats : list[tuple[str, trt.DataType, trt.TensorFormat]], optional
        A list of (name, dtype format) to allow deep specification of input layers.
        For example, input_tensor_formats=[("input", trt.DataType.UINT8, trt.TensorFormat.HWC)]
        By default, None
    output_tensor_formats : list[tuple[str, trt.DataType, trt.TensorFormat]], optional
        A list of (name, dtype format) to allow deep specification of output layers.
        For example, output_tensor_formats=[("output", trt.DataType.HALF, trt.TensorFormat.LINEAR)]
        By default, None
    hooks : list[Callable[[trt.INetworkDefinition], trt.INetworkDefinition]], optional
        An optional list of 'hook' functions to modify the TensorRT network before
        the remainder of the build phase occurs.
        By default, None
    optimization_level : int, optional
        Optimization level to apply to the TensorRT builder config (0-5).
        By default, 3.
    profiling_verbosity : trt.ProfilingVerbosity | None, optional
        Level of detail for profiling information in the built engine.
        Options are: trt.ProfilingVerbosity.NONE, trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
        trt.ProfilingVerbosity.DETAILED
        DETAILED is recommended for best layer names when using profile_engine.
        By default, None (uses TensorRT's default).
    tiling_optimization_level : int, optional
        Tiling optimization level to enable cross-kernel tiled inference.
        By default, 0 (no tiling optimization).
    tiling_l2_cache_limit : int, None, optional
        L2 cache limit (in bytes) for tiling optimization.
        By default, None (TensorRT manages the default value).
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
    cache : bool, optional
        Whether or not to cache the engine in the trtutils engine cache.
        If an existing version is found will use that.
        Uses the name of the output file to assess if the engine has been compiled before.
        As such, naming the output 'engine', 'model' or similiar will result in
        unintended caching behavior.
        By default None, will not cache the engine.
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
    # load libnvinfer plugins
    CONFIG.load_plugins()

    output_path = Path(output).resolve()

    # first thing is to check cache
    if cache:
        exists, location = caching_tools.query(output_path.stem)
        if exists:
            shutil.copy(location, output_path)
            return

    # validate and handle timing_cache parameter
    use_global_timing_cache = False
    if timing_cache is True or timing_cache == "global":
        use_global_timing_cache = True
        timing_cache_path = None
    elif timing_cache is None:
        timing_cache_path = None
    elif isinstance(timing_cache, (Path, str)):
        timing_cache_path = Path(timing_cache).resolve()
    else:
        err_msg = (
            f"Invalid timing_cache value: {timing_cache}. "
            "Must be None, Path, str, True, or 'global'."
        )
        raise ValueError(err_msg)

    # match the device
    valid_gpu = ["gpu", "GPU"]
    valid_dla = ["dla", "DLA"]
    if isinstance(default_device, str):
        if default_device not in valid_gpu + valid_dla:
            err_msg = (
                f"Invalid default device: {default_device}. Must be one of: {valid_gpu + valid_dla}"
            )
            raise ValueError(err_msg)
        default_device = trt.DeviceType.GPU if default_device in valid_gpu else trt.DeviceType.DLA
    else:
        if default_device not in [trt.DeviceType.GPU, trt.DeviceType.DLA]:
            err_msg = (
                f"Invalid default device: {default_device}. Must be one of: {valid_gpu + valid_dla}"
            )
            raise ValueError(err_msg)
        default_device = (
            trt.DeviceType.GPU if default_device == trt.DeviceType.GPU else trt.DeviceType.DLA
        )

    # read the onnx model
    network, builder, config, _ = read_onnx(
        onnx,
        workspace,
    )

    # handle all hooks to start
    if hooks is not None:
        for hook in hooks:
            network = hook(network)

    # helper function for checking if layer can run on DLA
    check_dla: Callable[[trt.ILayer], bool] = get_check_dla(config)

    if verbose and FLAGS.BUILD_PROGRESS and ProgressBar is not None:
        LOG.debug("Applying ProgressBar to config")
        config.progress_monitor = ProgressBar()

    # create profile and config
    profile = builder.create_optimization_profile()

    # handle if manual shapes were passed for inputs
    if shapes:
        for input_name, shape in shapes:
            # set the minimum, optimal, maximum to all the same
            profile.set_shape(input_name, shape, shape, shape)

    config.add_optimization_profile(profile)

    if not (_MIN_OPTIM_LEVEL <= optimization_level <= _MAX_OPTIM_LEVEL):
        err_msg = "Builder optimization level must be between 0 and 5."
        raise ValueError(err_msg)
    config.builder_optimization_level = int(optimization_level)

    # handle profiling verbosity
    if profiling_verbosity is not None:
        config.profiling_verbosity = profiling_verbosity

    # handle tiling optimization
    if tiling_optimization_level is not None:
        config.tiling_optimization_level = tiling_optimization_level
    if tiling_l2_cache_limit is not None:
        config.l2_limit_for_tiling = tiling_l2_cache_limit

    # handle some flags
    if prefer_precision_constraints:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    if reject_empty_algorithms:
        config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
    # handle custom datatype/format for input/output tensors
    if (input_tensor_formats is not None or output_tensor_formats is not None) and not direct_io:
        LOG.warning("Direct IO not enabled, but some tensor formats specified. Enabling direct IO.")
        direct_io = True
    if direct_io:
        config.set_flag(trt.BuilderFlag.DIRECT_IO)
    if input_tensor_formats is not None:
        for tensor_name, tensor_dtype, tensor_format in input_tensor_formats:
            found = False
            for idx in range(network.num_inputs):
                inp = network.get_input(idx)
                if inp.name == tensor_name:
                    inp.dtype = tensor_dtype
                    inp.allowed_formats = 1 << int(tensor_format)
                    found = True
                    break
            if not found:
                LOG.warning(f"Input tensor '{tensor_name}' not found in network")
    if output_tensor_formats is not None:
        for tensor_name, tensor_dtype, tensor_format in output_tensor_formats:
            found = False
            for idx in range(network.num_outputs):
                out = network.get_output(idx)
                if out.name == tensor_name:
                    out.dtype = tensor_dtype
                    out.allowed_formats = 1 << int(tensor_format)
                    found = True
                    break
            if not found:
                LOG.warning(f"Output tensor '{tensor_name}' not found in network")

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
        # remove the validation since we bundle the layer idx with the precision
        # # validate length
        # if len(layer_precision) != network.num_layers:
        #     err_msg = "Layer precision list must be the same length as the number of layers in the network."
        #     raise ValueError(err_msg)
        # handle precision assignment
        for layer_idx, precision in layer_precision:
            if precision is None:
                continue
            layer = network.get_layer(layer_idx)
            layer.precision = precision

    # handle individual layer device
    if layer_device is not None:
        # remove the validation since we bundle the layer idx with the device
        # # validate length
        # if len(layer_device) != network.num_layers:
        #     err_msg = (
        #         "Layer device list must be the same length as the number of layers in the network."
        #     )
        #     raise ValueError(err_msg)
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

    # load/setup the timing cache
    t_cache: trt.ITimingCache | None = None
    if use_global_timing_cache:
        # use global timing cache from cache directory
        exists, global_cache_path = query_timing_cache()
        buffer = b""
        if exists:
            with global_cache_path.open("rb") as timing_cache_file:
                buffer = timing_cache_file.read()
        t_cache = config.create_timing_cache(buffer)
        config.set_timing_cache(t_cache, ignore_mismatch=ignore_timing_mismatch)
    elif timing_cache_path:
        # use specified timing cache path
        buffer = b""
        if timing_cache_path.exists():
            with timing_cache_path.open("rb") as timing_cache_file:
                buffer = timing_cache_file.read()
        t_cache = config.create_timing_cache(buffer)
        config.set_timing_cache(t_cache, ignore_mismatch=ignore_timing_mismatch)

    # build the engine
    if FLAGS.BUILD_SERIALIZED:
        engine_bytes = builder.build_serialized_network(network, config)
    else:
        engine_bytes = builder.build_engine(network, config)

    # save the timing cache
    if use_global_timing_cache:
        # save to global timing cache in cache directory
        post_t_cache = config.get_timing_cache()
        save_timing_cache_to_global(post_t_cache, overwrite=True)
    elif timing_cache_path:
        # save to specified timing cache path
        post_t_cache = config.get_timing_cache()
        with timing_cache_path.open("wb") as f:
            f.write(memoryview(post_t_cache.serialize()))

    if engine_bytes is None:
        err_msg = "Failed to build engine."
        raise RuntimeError(err_msg)

    with output_path.open("wb") as f:
        f.write(engine_bytes)

    if cache:
        caching_tools.store(output_path, overwrite=False, clear_old=False)
