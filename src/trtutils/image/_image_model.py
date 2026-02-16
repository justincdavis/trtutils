# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import nvtx

from trtutils._engine import TRTEngine
from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._device import Device
from trtutils.core._graph import CUDAGraph
from trtutils.core._memory import memcpy_device_to_host_async, memcpy_host_to_device_async
from trtutils.core._stream import stream_synchronize

from .preprocessors import CPUPreprocessor, CUDAPreprocessor, TRTPreprocessor
from .preprocessors._image_preproc import _is_single_image

if TYPE_CHECKING:
    from typing_extensions import Self

_COLOR_CHANNELS = 3


class ImageModel:
    """Abstract base class for image models."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "trt",
        resize_method: str = "linear",
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = False,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create an ImageModel object.

        Parameters
        ----------
        engine_path : Path, str
            The Path or str to the compiled TensorRT engine.
        warmup_iterations : int
            The number of warmup iterations to perform.
            The default is 10.
        input_range : tuple[float, float]
            The range of input values which should be passed to
            the model. By default [0.0, 1.0].
        preprocessor : str
            The type of preprocessor to use.
            The options are ['cpu', 'cuda', 'trt'], default is 'trt'.
        resize_method : str
            The type of resize algorithm to use.
            The options are ['letterbox', 'linear'], default is 'linear'.
        mean : tuple[float, float, float] | None, optional
            The mean values to use for the imagenet normalization.
            By default, None, which means no normalization will be applied.
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for the imagenet normalization.
            By default, None, which means no normalization will be applied.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        device : int, optional
            The CUDA device index to use for this model. Default is None,
            which uses the current device.
        backend : str
            The execution backend to use. Options are ['auto', 'async_v3', 'async_v2'].
            Default is 'auto', which selects the best available backend.
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for underlying CUDA operations.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
        cuda_graph : bool, optional
            Whether or not to enable CUDA graph capture for optimized execution.
            When enabled, CUDA graphs are used both at the engine level and for
            end-to-end execution in the end2end() method. The first call to
            end2end() will capture a CUDA graph of the full preprocessing +
            inference pipeline, and subsequent calls will replay it. Input
            dimensions are locked after the first end2end() call.
            Only effective with async_v3 backend. Default is None (disabled).
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        Raises
        ------
        ValueError
            If input size format is incorrect
            If model does not take 3 channel input

        """
        self._tag: str = f"{Path(engine_path).stem}"
        self._verbose: bool = verbose if verbose is not None else False

        self._nvtx_tags: dict[str, str] = {
            "init": f"image_model::init [{self._tag}]",
            "preprocess": f"image_model::preprocess [{self._tag}]",
            "mock_run": f"image_model::mock_run [{self._tag}]",
            "_end2end_graph_core": f"image_model::_end2end_graph_core [{self._tag}]",
            "_copy_engine_outputs": f"image_model::_copy_engine_outputs [{self._tag}]",
            "_setup_cpu_preproc": f"image_model::_setup_cpu_preproc [{self._tag}]",
            "_setup_cuda_preproc": f"image_model::_setup_cuda_preproc [{self._tag}]",
            "_setup_trt_preproc": f"image_model::_setup_trt_preproc [{self._tag}]",
        }

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["init"])

        if self._verbose:
            LOG.debug(f"Creating ImageModel: {self._tag}")

        self._device = device
        self._pagelocked_mem = pagelocked_mem if pagelocked_mem is not None else True
        self._engine = TRTEngine(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            backend=backend,
            warmup=warmup,
            dla_core=dla_core,
            device=device,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            no_warn=no_warn,
            verbose=verbose,
        )
        self._unified_mem = self._engine.unified_mem
        self._resize_method: str = resize_method
        # Find the image input: 4D tensor with 3 channels (batch, 3, H, W)
        input_size: tuple[int, ...] | None = None
        input_spec = self._engine.input_spec[0]
        expected_input_size = 4
        for spec in self._engine.input_spec:
            shape = tuple(spec[0])
            if len(shape) == expected_input_size and shape[1] == _COLOR_CHANNELS:
                input_size = shape
                input_spec = spec
                break

        if input_size is None:
            # Fallback: use first input with original validation
            input_size = tuple(self._engine.input_spec[0][0])
            expected_input_size = 4
            if len(input_size) != expected_input_size:
                err_msg = (
                    "Expected model to have input size of form: (batch, channels, height, width)"
                )
                err_msg += f", found {input_size}"
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # init
                raise ValueError(err_msg)
            rgb_channels = 3
            if input_size[1] != rgb_channels:
                err_msg = (
                    f"Expected model to take {rgb_channels} channel input, found {input_size[1]}"
                )
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # init
                raise ValueError(err_msg)

        # Extract batch size from engine
        self._batch_size: int = input_size[0]
        self._is_dynamic_batch: bool = self._batch_size == -1
        if self._is_dynamic_batch:
            self._batch_size = 1  # Default to 1 for dynamic

        self._input_size: tuple[int, int] = (input_size[3], input_size[2])
        self._dtype = input_spec[1]
        self._input_range = input_range
        self._mean = mean
        self._std = std

        # Model-specific attributes — set by subclass _configure_model() overrides.
        # Declared here with defaults so base methods can reference them.
        # -- Detector --
        self._orig_size_dtype: np.dtype[Any] | None = None
        self._use_image_size: bool = False
        self._use_scale_factor: bool = False

        # Hook for subclasses to configure model-specific state
        # after engine is loaded but before preprocessors are created.
        self._configure_model()

        # set up the preprocessor
        self._preprocessor: CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor
        valid_preprocessors = ["cpu", "cuda", "trt"]
        if preprocessor not in valid_preprocessors:
            err_msg = f"Invalid preprocessor found, options are: {valid_preprocessors}"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # init
            raise ValueError(err_msg)
        self._preproc_cpu: CPUPreprocessor = self._setup_cpu_preproc()
        self._preproc_cuda: CUDAPreprocessor | None = None
        self._preproc_trt: TRTPreprocessor | None = None
        # change the preprocessor setup to cuda if set to trt and trt doesnt have uint8 support
        if preprocessor == "trt" and not FLAGS.TRT_HAS_UINT8:
            preprocessor = "cuda"
            LOG.warning(
                "Preprocessing method set to TensorRT, but platform doesnt have UINT8 support, fallback to CUDA."
            )
        # existing logic — guard CUDA/TRT preprocessor setup on correct device
        with Device(device):
            if preprocessor == "trt":
                self._preproc_trt = self._setup_trt_preproc()
                self._preprocessor = self._preproc_trt
            elif preprocessor == "cuda" and self._dtype == np.float32:
                self._preproc_cuda = self._setup_cuda_preproc()
                self._preprocessor = self._preproc_cuda
            else:
                self._preprocessor = self._preproc_cpu

        # basic profiler setup
        self._pre_profile: tuple[float, float] = (0.0, 0.0)
        self._infer_profile: tuple[float, float] = (0.0, 0.0)
        self._post_profile: tuple[float, float] = (0.0, 0.0)

        # E2E graph state (enabled when cuda_graph=True)
        # Note: Preprocessing runs outside the graph since H2D copies cannot be captured.
        # Only TRTEngine inference is captured in the graph.
        self._e2e_graph_enabled: bool = cuda_graph if cuda_graph is not None else False
        self._e2e_graph: CUDAGraph | None = None
        self._e2e_input_dims: tuple[int, int] | None = None
        self._e2e_batch_size: int | None = None

        # if warmup, warmup the preprocessors
        if warmup:
            self._preprocessor.warmup()

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # init

    def _setup_cpu_preproc(self: Self) -> CPUPreprocessor:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["_setup_cpu_preproc"])

        result = CPUPreprocessor(
            self._input_size,
            self._input_range,
            self._dtype,
            mean=self._mean,
            std=self._std,
            tag=self._tag,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # setup_cpu_preproc

        return result

    def _setup_cuda_preproc(self: Self) -> CUDAPreprocessor:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["_setup_cuda_preproc"])

        result = CUDAPreprocessor(
            self._input_size,
            self._input_range,
            self._dtype,
            resize=self._resize_method,
            stream=self._engine.stream,
            mean=self._mean,
            std=self._std,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
            tag=self._tag,
            orig_size_dtype=self._orig_size_dtype,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # setup_cuda_preproc

        return result

    def _setup_trt_preproc(self: Self) -> TRTPreprocessor:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["_setup_trt_preproc"])

        result = TRTPreprocessor(
            self._input_size,
            self._input_range,
            self._dtype,
            batch_size=self._batch_size,
            resize=self._resize_method,
            stream=self._engine.stream,
            mean=self._mean,
            std=self._std,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=self._unified_mem,
            tag=self._tag,
            orig_size_dtype=self._orig_size_dtype,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # setup_trt_preproc

        return result

    def _configure_model(self: Self) -> None:
        """Configure model-specific state after engine load, before preprocessor creation."""

    @property
    def engine(self: Self) -> TRTEngine:
        """Get the underlying TRTEngine."""
        return self._engine

    @property
    def name(self: Self) -> str:
        """Get the name of the engine."""
        return self._engine.name

    @property
    def input_shape(self: Self) -> tuple[int, int]:
        """Get the width, height input shape."""
        return self._input_size

    @property
    def dtype(self: Self) -> np.dtype:
        """Get the dtype required by the model."""
        return self._dtype

    @property
    def batch_size(self: Self) -> int:
        """Get the batch size of the model."""
        return self._batch_size

    @property
    def is_dynamic_batch(self: Self) -> bool:
        """Check if model has dynamic batch size."""
        return self._is_dynamic_batch

    def _update_preprocessors(self: Self) -> None:
        if self._preproc_cpu is not None:
            self._preproc_cpu = self._setup_cpu_preproc()
        if self._preproc_cuda is not None:
            self._preproc_cuda = self._setup_cuda_preproc()
        if self._preproc_trt is not None:
            self._preproc_trt = self._setup_trt_preproc()

    def update_input_range(self: Self, input_range: tuple[float, float]) -> None:
        """
        Update the input range of the model.

        This will re-create all preprocessors with the new input range.
        Only preprocessors which have been created will be re-created.

        Parameters
        ----------
        input_range : tuple[float, float]
            The new input range.

        """
        self._input_range = input_range
        self._update_preprocessors()

    def update_mean_std(
        self: Self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> None:
        """
        Update the mean and standard deviation of the model.

        Parameters
        ----------
        mean : tuple[float, float, float]
            The new mean.
        std : tuple[float, float, float]
            The new standard deviation.

        Raises
        ------
        ValueError
            If the mean or std is not a tuple of 3 floats

        """
        if len(mean) != _COLOR_CHANNELS:
            err_msg = f"{self._tag}: Mean must be a tuple of 3 floats"
            raise ValueError(err_msg)
        if len(std) != _COLOR_CHANNELS:
            err_msg = f"{self._tag}: Std must be a tuple of 3 floats"
            raise ValueError(err_msg)
        self._mean = mean
        self._std = std
        self._update_preprocessors()

    def get_random_input(
        self: Self,
    ) -> list[np.ndarray]:
        """
        Generate random images for the model.

        Returns
        -------
        list[np.ndarray]
            A list containing one random image.

        """
        return [self._engine.get_random_input()[0]]

    def mock_run(
        self: Self,
        images: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """
        Mock an execution of the model.

        Parameters
        ----------
        images : list[np.ndarray], optional
            Optional batch of images to use for execution.
            If None, random data will be generated.

        Returns
        -------
        list[np.ndarray]
            The raw outputs of the model.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["mock_run"])

        if images is not None:
            # Stack images into batch tensor if provided as list
            if len(images) == 1:
                result = self._engine.mock_execute(data=[images[0]])
            else:
                batch_tensor = np.stack(images, axis=0)
                result = self._engine.mock_execute(data=[batch_tensor])
        else:
            result = self._engine.mock_execute()

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # mock_run

        return result

    # preprocess overloads
    @overload
    def preprocess(
        self: Self,
        images: np.ndarray,
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def preprocess(
        self: Self,
        images: list[np.ndarray],
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    def preprocess(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        resize: str | None = None,
        method: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess the input images.

        Parameters
        ----------
        images : np.ndarray | list[np.ndarray]
            A single image (HWC format) or list of images to preprocess.
        resize : str
            The method to resize the images with.
            Options are [letterbox, linear].
            By default None, which will use the value passed
            during initialization.
        method : str, optional
            The underlying preprocessor to use.
            Options are 'cpu', 'cuda', or 'trt'. By default None, which
            will use the preprocessor stated in the constructor.
        no_copy : bool, optional
            If True and using CUDA, do not copy the
            data from the allocated memory. If the data
            is not copied, it WILL BE OVERWRITTEN INPLACE
            once new data is generated.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
            The preprocessed batch tensor, list of ratios per image, and list of padding per image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["preprocess"])

        # Handle single-image input
        is_single = _is_single_image(images)
        if is_single:
            images = [images]  # type: ignore[list-item]

        resize = resize if resize is not None else self._resize_method
        if verbose:
            LOG.debug(
                f"{self._tag}: Running preprocess, batch_size: {len(images)}, with method: {resize}",
            )
            LOG.debug(f"{self._tag}: Using device: {method}")
        preprocessor = self._preprocessor
        if method is not None:
            if method == "trt" and not FLAGS.TRT_HAS_UINT8:
                method = "cuda"
                LOG.warning(
                    "Preprocessing method set to TensorRT, but platform doesn't support UINT8, fallback to CUDA."
                )
            preprocessor = self._preproc_cpu
            if method == "cuda":
                if self._preproc_cuda is None:
                    self._preproc_cuda = self._setup_cuda_preproc()
                preprocessor = self._preproc_cuda
            elif method == "trt":
                if self._preproc_trt is None:
                    self._preproc_trt = self._setup_trt_preproc()
                preprocessor = self._preproc_trt
        if isinstance(preprocessor, (CUDAPreprocessor, TRTPreprocessor)):
            t0 = time.perf_counter()
            data = preprocessor(images, resize=resize, no_copy=no_copy, verbose=verbose)
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            data = preprocessor(images, resize=resize, verbose=verbose)
            t1 = time.perf_counter()
        self._pre_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # preprocess

        return data

    def _copy_engine_outputs(self: Self) -> list[np.ndarray]:
        """Copy engine outputs from device to host."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["_copy_engine_outputs"])

        outputs: list[np.ndarray] = []
        for binding in self._engine._outputs:  # noqa: SLF001
            if not (self._engine._unified_mem and self._engine._pagelocked_mem):  # noqa: SLF001
                memcpy_device_to_host_async(
                    binding.host_allocation,
                    binding.allocation,
                    self._engine.stream,
                )
            outputs.append(binding.host_allocation)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # copy_engine_outputs

        return outputs

    def _prepare_extra_engine_inputs_gpu(self: Self) -> list[int]:
        """
        Return additional GPU input pointers for GPU preprocessor path.

        Override in subclasses that need extra inputs (e.g., DETR models).

        Returns
        -------
        list[int]
            List of GPU device pointers for additional inputs.

        """
        return []

    def _prepare_extra_engine_inputs_cpu(
        self: Self,
        images: list[np.ndarray],  # noqa: ARG002
        ratios: list[tuple[float, float]],  # noqa: ARG002
    ) -> list[int]:
        """
        Return additional GPU input pointers for CPU preprocessor path.

        Override in subclasses that need extra inputs (e.g., DETR models).
        Subclasses should copy data to engine input bindings and return pointers.

        Parameters
        ----------
        images : list[np.ndarray]
            The original input images (before preprocessing).
        ratios : list[tuple[float, float]]
            The scaling ratios from preprocessing.

        Returns
        -------
        list[int]
            List of GPU device pointers for additional inputs.

        """
        return []

    def _build_graph_input_ptrs(
        self: Self,
        gpu_ptr: int,
        extra_ptrs: list[int],
    ) -> list[int]:
        """
        Build input pointer list for CUDA graph execution.

        Override in subclasses that need schema-specific input ordering.
        The default puts the image first followed by extra inputs.

        Parameters
        ----------
        gpu_ptr : int
            GPU device pointer for the main image input.
        extra_ptrs : list[int]
            Additional GPU device pointers for extra inputs.

        Returns
        -------
        list[int]
            Ordered list of GPU device pointers for engine execution.

        """
        return [gpu_ptr, *extra_ptrs]

    def _end2end_graph_core(
        self: Self,
        images: list[np.ndarray],
        *,
        verbose: bool | None = None,
    ) -> tuple[list[np.ndarray], list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Core graph-accelerated execution shared by subclasses.

        Handles dimension locking, preprocessing dispatch, graph capture/replay,
        and D2H output copy. Subclasses call this and then perform their own
        postprocessing.

        Parameters
        ----------
        images : list[np.ndarray]
            List of images to process.
        verbose : bool, optional
            Whether to log additional information.

        Returns
        -------
        tuple[list[np.ndarray], list[tuple[float, float]], list[tuple[float, float]]]
            Raw outputs, ratios, and padding for subclass postprocessing.

        Raises
        ------
        RuntimeError
            If image dimensions or batch size change after first call.
        RuntimeError
            If CUDA graph capture fails.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["_end2end_graph_core"])

        batch_size = len(images)

        # Auto-capture on first call: lock dimensions
        if self._e2e_graph is None:
            self._e2e_input_dims = (images[0].shape[0], images[0].shape[1])
            self._e2e_batch_size = batch_size

        # Validate dimensions match locked values
        img_dims = (images[0].shape[0], images[0].shape[1])
        if img_dims != self._e2e_input_dims:
            err_msg = f"Image dims {img_dims} != graph dims {self._e2e_input_dims}"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # end2end_graph_core
            raise RuntimeError(err_msg)
        if batch_size != self._e2e_batch_size:
            err_msg = f"Batch size {batch_size} != graph batch size {self._e2e_batch_size}"
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # end2end_graph_core
            raise RuntimeError(err_msg)

        # Preprocess and get GPU pointer based on preprocessor type
        if isinstance(self._preprocessor, (CUDAPreprocessor, TRTPreprocessor)):
            # GPU preprocessor: use direct_preproc for GPU pointer
            gpu_ptr, ratios, padding = self._preprocessor.direct_preproc(
                images,
                resize=self._resize_method,
                no_warn=True,
                verbose=verbose,
            )
            extra_ptrs = self._prepare_extra_engine_inputs_gpu()
            input_ptrs = self._build_graph_input_ptrs(gpu_ptr, extra_ptrs)
        else:
            # CPU preprocessor: preprocess to numpy, copy to engine binding
            # Both Classifier and Detector have compatible preprocess signatures for basic call
            tensor, ratios, padding = self.preprocess(images, no_copy=True, verbose=verbose)

            # Copy preprocessed tensor to engine's input binding (H2D)
            memcpy_host_to_device_async(
                self._engine._inputs[0].allocation,  # noqa: SLF001
                tensor,
                self._engine.stream,
            )
            gpu_ptr = self._engine._inputs[0].allocation  # noqa: SLF001

            # Add extra inputs from subclass
            extra_ptrs = self._prepare_extra_engine_inputs_cpu(images, ratios)
            input_ptrs = self._build_graph_input_ptrs(gpu_ptr, extra_ptrs)

        # Capture or replay the graph (inference only)
        if self._e2e_graph is None:
            # First call: capture the graph
            self._e2e_graph = CUDAGraph(self._engine.stream)
            with self._e2e_graph:
                self._engine.raw_exec(input_ptrs, no_warn=True)

            # Verify capture succeeded
            if not self._e2e_graph.is_captured:
                err_msg = (
                    "CUDA graph capture failed for end2end. Engine may not support graph capture."
                )
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # end2end_graph_core
                raise RuntimeError(err_msg)

            # Launch graph after capture to actually run inference
            # (capture only records operations, doesn't execute them)
            self._e2e_graph.launch()
        else:
            # Replay the captured graph
            self._e2e_graph.launch()

        # D2H copy of outputs + sync (outside the graph)
        raw_outputs = self._copy_engine_outputs()
        stream_synchronize(self._engine.stream)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # end2end_graph_core

        return raw_outputs, ratios, padding
