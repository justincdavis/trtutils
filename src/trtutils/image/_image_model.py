# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, TypeVar, overload

import numpy as np
import nvtx

from trtutils._engine import TRTEngine
from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._buffer import Buffer
from trtutils.core._device import Device
from trtutils.core._graph import CUDAGraph
from trtutils.core._memory import memcpy_host_to_device_async
from trtutils.core._stream import stream_synchronize

from .preprocessors import CPUPreprocessor, CUDAPreprocessor, TRTPreprocessor

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.image.interfaces import ImageInput

_COLOR_CHANNELS = 3
# max cached end2end CUDA graphs before the cache resets
_MAX_E2E_GRAPHS = 32

# (raw_outputs, ratios, padding, no_copy) -> postprocessed outputs per image
_PostFn = Callable[
    [List[np.ndarray], List[Tuple[float, float]], List[Tuple[float, float]], Optional[bool]],
    List[List[np.ndarray]],
]
T = TypeVar("T")


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
        cuda_graph: bool | None = None,
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
            inference pipeline, and subsequent calls will replay it. Graphs
            are cached per batch size and input buffer set, so image
            resolution and batch size may both change between calls.
            Only effective with async_v3 backend. Default is True.
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
            "run": f"image_model::run [{self._tag}]",
            "end2end": f"image_model::end2end [{self._tag}]",
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
        # whether the engine takes a single image input (no extra per-image
        # host-built inputs). Only such schemas can take the direct-GPU
        # run() path, since the extras are built from host data in
        # _engine_inputs. Detector overrides this in _configure_model.
        self._single_input_schema: bool = True

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
        self._e2e_graph_enabled: bool = cuda_graph if cuda_graph is not None else True
        # graphs cached per (batch_size, input pointer set). including the
        # pointers in the key means graphs baked against reallocated
        # preprocessor buffers naturally miss instead of replaying stale
        # addresses. bounded, cleared wholesale on overflow.
        self._e2e_graphs: dict[tuple[int, tuple[int, ...]], CUDAGraph] = {}

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

    def _validate_batch_size(self: Self, batch_size: int) -> None:
        """
        Reject a batch size a static engine cannot serve.

        A static engine's bindings are sized for exactly one batch size.
        Submitting a different one runs the engine against allocations that
        do not match the submitted data: the trailing rows are garbage at
        best, and reads run past the preprocessor's batch buffer at worst.
        Every execution path must check this before dispatching.

        Parameters
        ----------
        batch_size : int
            The number of images submitted.

        Raises
        ------
        RuntimeError
            If the engine is static and the batch size does not match.

        """
        if self._engine.is_dynamic_batch or batch_size == self._batch_size:
            return
        err_msg = (
            f"Batch size {batch_size} != engine batch size {self._batch_size}. "
            "Build the engine with a dynamic batch profile to submit variable "
            "batch sizes."
        )
        raise RuntimeError(err_msg)

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
        images: ImageInput,
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def preprocess(
        self: Self,
        images: list[ImageInput],
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    def preprocess(
        self: Self,
        images: ImageInput | list[ImageInput],
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
        images : ImageInput | list[ImageInput]
            A single image or list of images, each an HWC uint8
            ``np.ndarray`` or a ``Buffer`` (host or device) holding one. A
            device Buffer is consumed in place by the CUDA/TRT
            preprocessors (no H2D copy); the CPU preprocessor copies it to
            host once.
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
        if isinstance(images, (np.ndarray, Buffer)):
            batch_images: list[ImageInput] = [images]
        else:
            batch_images = images

        resize = resize if resize is not None else self._resize_method
        if verbose:
            LOG.debug(
                f"{self._tag}: Running preprocess, batch_size: {len(batch_images)}, with method: {resize}",
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
        # every preprocessor honors no_copy: the GPU ones hand back a view of
        # their staging binding, the CPU one a view of its reusable batch
        # tensor. Dropping it here forced a fresh copy of the whole batch on
        # the CPU path, which above 32 MB is an mmap that faults in on every
        # call rather than a cheap memcpy.
        t0 = time.perf_counter()
        data = preprocessor(batch_images, resize=resize, no_copy=no_copy, verbose=verbose)
        t1 = time.perf_counter()
        self._pre_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # preprocess

        return data

    def _stage_graphed_postprocess(
        self: Self,
        batch_size: int,  # noqa: ARG002
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Stage per-image transform data for an in-graph postprocess kernel.

        No-op in the base class. Subclasses with a GPU postprocessor
        upload the per-image transforms here (outside the graph, values
        change per call) and return identity ratios/padding so the CPU
        postprocess affine becomes a pass-through.
        """
        return ratios, padding

    def _capture_graphed_postprocess(self: Self, batch_size: int) -> None:
        """
        Enqueue the in-graph postprocess kernel, if any.

        No-op in the base class. Called inside CUDA graph capture, after
        engine execution — the launch is recorded into the graph, so
        replays run it with zero per-call launch overhead.
        """

    def _copy_engine_outputs(
        self: Self,
        output_shapes: list[tuple[int, ...]] | None = None,
    ) -> list[np.ndarray]:
        """Copy engine outputs from device to host, sized to the actual shapes."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["_copy_engine_outputs"])

        # when output_shapes is given only the prefix of each max-shape
        # allocation is valid, so copy and expose exactly that
        outputs: list[np.ndarray] = [
            binding.download(
                self._engine.stream,
                output_shapes[o_idx] if output_shapes is not None else None,
            )
            for o_idx, binding in enumerate(self._engine._outputs)  # noqa: SLF001
        ]

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
        images: list[ImageInput],  # noqa: ARG002
        ratios: list[tuple[float, float]],  # noqa: ARG002
    ) -> list[int]:
        """
        Return additional GPU input pointers for CPU preprocessor path.

        Override in subclasses that need extra inputs (e.g., DETR models).
        Subclasses should copy data to engine input bindings and return pointers.

        Parameters
        ----------
        images : list[ImageInput]
            The original input images (before preprocessing), each an HWC
            uint8 ``np.ndarray`` or a ``Buffer`` (host or device) holding
            one. Overrides that only need each image's shape can read
            ``.shape`` directly; it works the same on both types.
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
        image: Any,  # noqa: ANN401
        extras: list[Any],
    ) -> list[Any]:
        """
        Build the input list for CUDA graph execution or engine calls.

        Override in subclasses that need schema-specific input ordering.
        The default puts the image first followed by extra inputs. Inputs
        are GPU device pointers (int) on the CUDA graph path, or host
        arrays (np.ndarray) when reused to order CPU engine inputs.

        Parameters
        ----------
        image : Any
            The main image input: a GPU device pointer or a host array.
        extras : list[Any]
            Additional inputs: GPU device pointers or host arrays.

        Returns
        -------
        list[Any]
            Ordered list of inputs for engine execution.

        """
        return [image, *extras]

    def _engine_inputs(
        self: Self,
        tensor: np.ndarray,
        images: list[ImageInput],  # noqa: ARG002
        ratios: list[tuple[float, float]] | None,  # noqa: ARG002
        *,
        preprocessed: bool,  # noqa: ARG002
    ) -> list[np.ndarray]:
        """
        Build the host arrays passed to the engine call.

        Override in subclasses that need schema-specific inputs, e.g.
        Detector adds image size / scale factor arrays for DETR models.

        Parameters
        ----------
        tensor : np.ndarray
            The preprocessed batch tensor.
        images : list[ImageInput]
            The original input images, before preprocessing, each an HWC
            uint8 ``np.ndarray`` or a ``Buffer`` (host or device) holding
            one.
        ratios : list[tuple[float, float]] | None
            The scaling ratios from preprocessing.
        preprocessed : bool
            Whether the inputs were already preprocessed.

        Returns
        -------
        list[np.ndarray]
            The host arrays to pass to the engine call.

        """
        return [tensor]

    def _end2end_graph_core(
        self: Self,
        images: list[ImageInput],
        *,
        verbose: bool | None = None,
    ) -> tuple[list[np.ndarray], list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Core graph-accelerated execution shared by subclasses.

        Handles batch validation, preprocessing dispatch, graph capture/replay
        (keyed per batch size and input pointer set), and D2H output copy.
        Subclasses call this and then perform their own postprocessing.

        With per-shape staging slots the preprocessor's output pointer
        is stable per resolution, so image dimensions are no longer locked on
        the first call: a changed resolution is just a graph-cache hit or
        miss like any other pointer change.

        Parameters
        ----------
        images : list[ImageInput]
            List of images to process, each an HWC uint8 ``np.ndarray`` or
            a ``Buffer`` (host or device) holding one.
        verbose : bool, optional
            Whether to log additional information.

        Returns
        -------
        tuple[list[np.ndarray], list[tuple[float, float]], list[tuple[float, float]]]
            Raw outputs, ratios, and padding for subclass postprocessing.

        Raises
        ------
        RuntimeError
            If a static engine receives a batch size different than what
            it was built for.
        RuntimeError
            If CUDA graph capture fails.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["_end2end_graph_core"])

        batch_size = len(images)

        # resolve the context shapes for this batch size
        # dynamic engines: sets the batch dim, returns actual output shapes
        # static engines: returns None, the submitted batch must match exactly
        # (the engine would read past the preprocessor's batch buffer and the
        # trailing output rows would be garbage otherwise)
        try:
            self._validate_batch_size(batch_size)
        except RuntimeError:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # end2end_graph_core
            raise
        output_shapes = self._engine._resolve_dynamic_batch(batch_size)  # noqa: SLF001

        # Preprocess and get GPU pointer based on preprocessor type
        if isinstance(self._preprocessor, (CUDAPreprocessor, TRTPreprocessor)):
            # GPU preprocessor: use direct_preproc for GPU pointer
            t0 = time.perf_counter()
            gpu_ptr, ratios, padding = self._preprocessor.direct_preproc(
                images,
                resize=self._resize_method,
                no_warn=True,
                verbose=verbose,
            )
            t1 = time.perf_counter()
            self._pre_profile = (t0, t1)
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

        # stage in-graph GPU postprocessing (no-op unless a subclass enables
        # it): uploads per-image transforms and swaps ratios/padding for
        # identity values so downstream CPU postprocess is a pass-through
        ratios, padding = self._stage_graphed_postprocess(batch_size, ratios, padding)

        # Capture or replay the graph for this (batch, pointers) combination.
        # A graph bakes the context shapes and tensor addresses at capture
        # time, so each batch size / pointer set gets its own graph. Replays
        # are shape-independent, no per-call context work.
        graph_key = (batch_size, tuple(input_ptrs))
        e2e_graph = self._e2e_graphs.get(graph_key)
        if e2e_graph is None:
            # bounded cache: distinct keys only accumulate when buffers get
            # reallocated or many batch sizes are used, clear wholesale
            if len(self._e2e_graphs) >= _MAX_E2E_GRAPHS:
                for old_graph in self._e2e_graphs.values():
                    old_graph.invalidate()
                self._e2e_graphs = {}

            e2e_graph = CUDAGraph(self._engine.stream)
            with e2e_graph:
                self._engine.raw_exec(input_ptrs, no_warn=True)
                self._capture_graphed_postprocess(batch_size)

            # Verify capture succeeded
            if not e2e_graph.is_captured:
                err_msg = (
                    "CUDA graph capture failed for end2end. Engine may not support graph capture."
                )
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # end2end_graph_core
                raise RuntimeError(err_msg)

            self._e2e_graphs[graph_key] = e2e_graph

        # Launch the graph (capture only records operations, doesn't execute them)
        e2e_graph.launch()

        # D2H copy of outputs + sync (outside the graph), sized to the
        # actual output shapes for partial batches on dynamic engines
        raw_outputs = self._copy_engine_outputs(output_shapes)
        stream_synchronize(self._engine.stream)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # end2end_graph_core

        return raw_outputs, ratios, padding

    @staticmethod
    def _as_batch(
        outputs: list[np.ndarray] | list[list[np.ndarray]],
    ) -> tuple[list[list[np.ndarray]], bool]:
        """
        Wrap single-image postprocessed outputs into batch form.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            Postprocessed outputs for a single image or a batch.

        Returns
        -------
        tuple[list[list[np.ndarray]], bool]
            The outputs in batch form, and whether the input was single-image.

        """
        if outputs and isinstance(outputs[0], np.ndarray):
            return [outputs], True  # ty: ignore[invalid-return-type]
        return outputs, False  # ty: ignore[invalid-return-type]

    def _run_direct_gpu(
        self: Self,
        batch_images: list[ImageInput],
        *,
        postprocess: bool,
        no_copy_run: bool | None,
        verbose: bool | None,
    ) -> tuple[
        list[np.ndarray],
        list[tuple[float, float]],
        list[tuple[float, float]],
        float,
        float,
    ]:
        """
        Run the direct-GPU path of ``_run_core``.

        Feeds the preprocessed tensor's device pointer straight to the
        engine instead of round-tripping through host memory (preprocess
        to host, then back to device on execute).

        Parameters
        ----------
        batch_images : list[ImageInput]
            The batch of images to run inference on.
        postprocess : bool
            Whether postprocessing will run on the raw outputs. Raw
            outputs are copied off the engine's host allocations only when
            this is False and ``no_copy_run`` is falsy, since the caller
            is keeping them past the next inference.
        no_copy_run : bool, optional
            Whether to avoid the extra copy of raw (non-postprocessed)
            outputs.
        verbose : bool, optional
            Whether to log additional information.

        Returns
        -------
        tuple[list[np.ndarray], list[tuple[float, float]], list[tuple[float, float]], float, float]
            Raw outputs, ratios, padding, and the ``perf_counter``
            timestamps bracketing engine execution.

        """
        t0 = time.perf_counter()
        outputs, batch_ratios, batch_padding = self._end2end_graph_core(
            batch_images,
            verbose=verbose,
        )
        # raw outputs reference the engine's host allocations
        if not postprocess and not no_copy_run:
            outputs = [o.copy() for o in outputs]
        t1 = time.perf_counter()
        return outputs, batch_ratios, batch_padding, t0, t1

    def _run_core(
        self: Self,
        images: ImageInput | list[ImageInput],
        ratios: tuple[float, float] | list[tuple[float, float]] | None,
        padding: tuple[float, float] | list[tuple[float, float]] | None,
        *,
        preprocessed: bool | None,
        postprocess: bool | None,
        no_copy: bool | None,
        verbose: bool | None,
        post: _PostFn,
        needs_ratios: bool = True,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """
        Shared implementation of the public ``run`` method.

        Handles single/batch normalization, the preprocess-or-validate
        split, engine execution + timing, and optional postprocessing.
        Subclasses supply ``post`` to bridge to their own postprocess
        signature.

        Parameters
        ----------
        images : ImageInput | list[ImageInput]
            A single image or batch of images to run inference on, each an
            HWC uint8 ``np.ndarray`` or a ``Buffer`` (host or device)
            holding one.
        ratios : tuple[float, float] | list[tuple[float, float]] | None
            Scaling ratios, required when ``preprocessed`` is True and
            ``postprocess`` is True (unless ``needs_ratios`` is False).
        padding : tuple[float, float] | list[tuple[float, float]] | None
            Padding, same requirements as ``ratios``.
        preprocessed : bool, optional
            Whether ``images`` is already a preprocessed batch tensor.
            By default None, treated as False.
        postprocess : bool, optional
            Whether to run postprocessing on the raw outputs.
            By default None, treated as True.
        no_copy : bool, optional
            Whether to avoid extra copies during preprocess/run/postprocess.
        verbose : bool, optional
            Whether to log additional information.
        post : _PostFn
            Callable bridging to the subclass's postprocess signature.
        needs_ratios : bool
            Whether postprocessing requires ratios/padding to be known.
            By default True; Classifier/DepthEstimator pass False since
            their postprocess does not use ratios/padding.

        Returns
        -------
        list[np.ndarray] | list[list[np.ndarray]]
            Raw outputs if ``postprocess`` is False, else postprocessed
            outputs (unwrapped for single-image input).

        Raises
        ------
        ValueError
            If ``preprocessed`` is True but ``images`` is not a list
            containing a single batch tensor.
        RuntimeError
            If ``postprocess`` is True, ``needs_ratios`` is True, and
            ratios/padding are not available (only possible when
            ``preprocessed`` is True and none were passed in).

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["run"])

        if verbose:
            LOG.debug(f"{self._tag}: run")

        # handle single-image input
        if isinstance(images, (np.ndarray, Buffer)):
            batch_images: list[ImageInput] = [images]
            is_single = True
        else:
            batch_images = images
            is_single = False

        # normalize ratios/padding to list form
        batch_ratios: list[tuple[float, float]] | None = (
            [ratios] if isinstance(ratios, tuple) else ratios
        )
        batch_padding: list[tuple[float, float]] | None = (
            [padding] if isinstance(padding, tuple) else padding
        )

        # assign flags
        if preprocessed is None:
            preprocessed = False
        if postprocess is None:
            postprocess = True

        # assign no_copy values
        if no_copy is None and not preprocessed and postprocess:
            # remove two sets of copies when doing preprocess/run/postprocess inside
            # a single run call
            no_copy_pre: bool | None = True
            no_copy_run: bool | None = True
            no_copy_post: bool | None = False
        else:
            no_copy_pre = no_copy
            no_copy_run = no_copy
            no_copy_post = no_copy

        if verbose:
            LOG.debug(
                f"{self._tag}: Running: preprocessed: {preprocessed}, postprocess: {postprocess}",
            )

        # Direct GPU path: for single-input schemas with a GPU-resident
        # preprocessor, feed the engine the preprocessed tensor's device
        # pointer instead of copying it to host (preprocess) and back to
        # device (execute). Multi-input schemas keep the host path since
        # their extra inputs are built from per-image host data in
        # _engine_inputs.
        use_direct = (
            not preprocessed
            and self._e2e_graph_enabled
            and isinstance(self._preprocessor, (CUDAPreprocessor, TRTPreprocessor))
            and self._single_input_schema
        )
        outputs: list[np.ndarray]
        if use_direct:
            outputs, batch_ratios, batch_padding, t0, t1 = self._run_direct_gpu(
                batch_images,
                postprocess=postprocess,
                no_copy_run=no_copy_run,
                verbose=verbose,
            )
        else:
            # handle preprocessing
            if not preprocessed:
                if verbose:
                    LOG.debug("Preprocessing inputs")
                tensor, batch_ratios, batch_padding = self.preprocess(
                    batch_images, no_copy=no_copy_pre
                )
            else:
                # images is already preprocessed tensor when preprocessed=True
                if len(batch_images) != 1 or not isinstance(batch_images[0], np.ndarray):
                    err_msg = "Preprocessed inputs must be a list containing a single batch tensor."
                    if FLAGS.NVTX_ENABLED:
                        nvtx.pop_range()  # run
                    raise ValueError(err_msg)
                tensor = batch_images[0]

            batch_size = len(batch_images) if not preprocessed else tensor.shape[0]
            self._validate_batch_size(batch_size)

            engine_inputs = self._engine_inputs(
                tensor, batch_images, batch_ratios, preprocessed=preprocessed
            )

            # execute
            t0 = time.perf_counter()
            outputs = self._engine(engine_inputs, no_copy=no_copy_run)
            t1 = time.perf_counter()

        # handle postprocessing
        if postprocess:
            if verbose:
                LOG.debug("Postprocessing outputs")
            if needs_ratios and (batch_ratios is None or batch_padding is None):
                err_msg = "Must pass ratios/padding if postprocessing and passing already preprocessed inputs."
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # run
                raise RuntimeError(err_msg)
            result = post(outputs, batch_ratios or [], batch_padding or [], no_copy_post)
            self._infer_profile = (t0, t1)

            # unwrap for single-image input
            if is_single:
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # run
                return result[0]
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # run
            return result

        self._infer_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # run

        return outputs

    def _end2end_core(
        self: Self,
        images: ImageInput | list[ImageInput],
        *,
        verbose: bool | None,
        post: _PostFn,
        get: Callable[[list[list[np.ndarray]]], list[T]],
        needs_ratios: bool = True,
    ) -> T | list[T]:
        """
        Shared implementation of the public ``end2end`` method.

        Dispatches to the CUDA-graph path, the CPU-preprocessor path
        (via ``_run_core``), or the GPU-preprocessor direct-exec path,
        then extracts final results with ``get``.

        Parameters
        ----------
        images : ImageInput | list[ImageInput]
            A single image or batch of images to run inference on, each an
            HWC uint8 ``np.ndarray`` or a ``Buffer`` (host or device)
            holding one.
        verbose : bool, optional
            Whether to log additional information.
        post : _PostFn
            Callable bridging to the subclass's postprocess signature.
        get : Callable[[list[list[np.ndarray]]], list[T]]
            Callable extracting final per-image results from postprocessed
            outputs, e.g. ``get_detections`` or ``get_classifications``.
        needs_ratios : bool
            Passed through to ``_run_core`` on the CPU-preprocessor path.

        Returns
        -------
        T | list[T]
            Final results, unwrapped for single-image input.

        Raises
        ------
        RuntimeError
            If a static engine receives a batch size different than what
            it was built for, or CUDA graph capture fails.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["end2end"])

        if verbose:
            LOG.debug(f"{self._tag}: end2end")

        # handle single-image input
        if isinstance(images, (np.ndarray, Buffer)):
            batch_images: list[ImageInput] = [images]
            is_single = True
        else:
            batch_images = images
            is_single = False

        no_copy_e2e = True
        pp: list[list[np.ndarray]]
        if self._e2e_graph_enabled:
            raw, ratios, padding = self._end2end_graph_core(batch_images, verbose=verbose)
            pp = post(raw, ratios, padding, no_copy_e2e)
        elif not isinstance(self._preprocessor, (CUDAPreprocessor, TRTPreprocessor)):
            if verbose:
                LOG.debug(f"{self._tag}: end2end -> calling CPU preprocess")
            pp = self._run_core(  # ty: ignore[invalid-assignment]
                batch_images,
                None,
                None,
                preprocessed=False,
                postprocess=True,
                no_copy=True,
                verbose=verbose,
                post=post,
                needs_ratios=needs_ratios,
            )
        else:
            if verbose:
                LOG.debug(f"{self._tag}: end2end -> calling CUDA preprocess")

            # Resolve the context shapes for this batch before dispatching.
            # direct_exec feeds the engine raw GPU pointers, so the shape
            # bookkeeping execute() derives from host arrays never runs and a
            # dynamic engine would return outputs at the max profile shape
            # while ratios/padding hold one entry per submitted image.
            self._validate_batch_size(len(batch_images))
            self._engine._resolve_dynamic_batch(len(batch_images))  # noqa: SLF001

            gpu_ptr, ratios, padding = self._preprocessor.direct_preproc(
                batch_images,
                resize=self._resize_method,
                no_warn=True,
                verbose=verbose,
            )
            input_ptrs = self._build_graph_input_ptrs(
                gpu_ptr, self._prepare_extra_engine_inputs_gpu()
            )
            raw = self._engine.direct_exec(input_ptrs, no_warn=True)
            pp = post(raw, ratios, padding, no_copy_e2e)

        result = get(pp)

        # unwrap for single-image input
        if is_single:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # end2end
            return result[0]

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # end2end

        return result
