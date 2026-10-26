# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, List, Optional, Tuple, TypeVar, overload

import numpy as np
import nvtx

from trtutils._engine import TRTEngine
from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._buffer import Buffer
from trtutils.core._device import Device

from ._schema import resolve_schema
from .preprocessors import (
    CPUPreprocessor,
    CUDAPreprocessor,
    GPUImagePreprocessor,
    TRTPreprocessor,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from enum import Enum

    from typing_extensions import Self

_COLOR_CHANNELS = 3

_PREPROCESSOR_METHODS = ("cpu", "cuda", "trt")


# (raw_outputs, ratios, padding, no_copy) -> postprocessed outputs per image
_PostFn = Callable[
    [List[np.ndarray], List[Tuple[float, float]], List[Tuple[float, float]], Optional[bool]],
    List[List[np.ndarray]],
]
T = TypeVar("T")


def host_input(array: np.ndarray | Buffer) -> Buffer:
    """
    Wrap data as an engine input.

    A Buffer is bound as is. ``np.ascontiguousarray`` returns ``array``
    itself when it is already C-contiguous, so the common case is
    zero-copy; only strided or Fortran-ordered data is gathered into a new array.
    """
    if isinstance(array, Buffer):
        return array
    return Buffer.wrap(np.ascontiguousarray(array))


class ImageModel:
    """Abstract base class for image models."""

    # task models set the enum used to resolve and order the engine outputs
    _OUTPUT_SCHEMA: ClassVar[type[Enum] | None] = None

    def __init__(  # noqa: PLR0915
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
        output_schema: Enum | str | None = None,
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
            When enabled, the engine records a CUDA graph for each input
            configuration (batch size, resolution, input buffers) once it
            repeats and replays it afterwards, for run() and end2end() alike.
            Preprocessing and postprocessing run outside the graphs.
            Only effective with async_v3 backend. Default is True.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.
        output_schema : Enum, str, optional
            Override for the output schema, a member or member name of the
            subclass's schema enum. Ignored by models without one.
            Default is None, which auto-detects the schema from the engine.

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
            "_setup_cpu_preproc": f"image_model::_setup_cpu_preproc [{self._tag}]",
            "_setup_cuda_preproc": f"image_model::_setup_cuda_preproc [{self._tag}]",
            "_setup_trt_preproc": f"image_model::_setup_trt_preproc [{self._tag}]",
        }

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["init"])
        try:
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
            # position of the image among the engine inputs; other inputs go around it
            self._image_input_index = 0
            expected_input_size = 4
            for index, spec in enumerate(self._engine.input_spec):
                shape = tuple(spec[0])
                if len(shape) == expected_input_size and shape[1] == _COLOR_CHANNELS:
                    input_size = shape
                    input_spec = spec
                    self._image_input_index = index
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
                    raise ValueError(err_msg)
                rgb_channels = 3
                if input_size[1] != rgb_channels:
                    err_msg = (
                        f"Expected model to take {rgb_channels} channel input, found {input_size[1]}"
                    )
                    raise ValueError(err_msg)

            # Extract batch size from engine (the max profile batch for a
            # dynamic-batch engine, which bounds what the preprocessors accept)
            self._batch_size: int = input_size[0]

            self._input_size: tuple[int, int] = (input_size[3], input_size[2])
            self._dtype = input_spec[1]
            self._input_range = input_range
            self._mean = mean
            self._std = std

            self._resolve_schema(output_schema)

            # Hook for subclasses to configure model-specific state
            # after engine is loaded but before preprocessors are created.
            self._configure_model()

            # set up the preprocessor; the default is kept as a resolved method so
            # get_preprocessor() always returns the current instance
            # (update_input_range/update_mean_std rebuild them)
            self._preprocessor_method: str = self._resolve_method(preprocessor)
            self._preproc_cpu: CPUPreprocessor = self._setup_cpu_preproc()
            self._preproc_cuda: CUDAPreprocessor | None = None
            self._preproc_trt: TRTPreprocessor | None = None
            default_preprocessor = self.get_preprocessor()

            # basic profiler setup
            self._pre_profile: tuple[float, float] = (0.0, 0.0)
            self._infer_profile: tuple[float, float] = (0.0, 0.0)
            self._post_profile: tuple[float, float] = (0.0, 0.0)

            # if warmup, warmup the preprocessors
            if warmup:
                default_preprocessor.warmup()

        finally:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # init

    def _setup_cpu_preproc(self: Self) -> CPUPreprocessor:
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["_setup_cpu_preproc"])

        result = CPUPreprocessor(
            self._input_size,
            self._input_range,
            self._dtype,
            resize=self._resize_method,
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
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # setup_trt_preproc

        return result

    def _resolve_schema(self: Self, override: Enum | str | None) -> None:
        """Resolve the output schema from the engine, and set the output order to match."""
        if self._OUTPUT_SCHEMA is not None:
            self._output_schema: Any = resolve_schema(self._engine, self._OUTPUT_SCHEMA, override)
            self._set_output_order(self._output_schema.value)

    def _configure_model(self: Self) -> None:
        """Configure model-specific state after engine load, before preprocessor creation."""

    def _set_output_order(self: Self, expected: Sequence[str]) -> None:
        """Map outputs by name, as tensorrt 8.6 does not keep the onnx output order."""
        names = list(self._engine.output_names)
        self._output_order: list[int] | None = None
        if names != list(expected) and sorted(names) == sorted(expected):
            self._output_order = [names.index(name) for name in expected]

    def _order_outputs(self: Self, outputs: list[np.ndarray]) -> list[np.ndarray]:
        """Reorder raw outputs to the expected names, see :meth:`_set_output_order`."""
        if self._output_order is None:
            return outputs
        return [outputs[i] for i in self._output_order]

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
    def infer_profile(self: Self) -> tuple[float, float]:
        """The start and end times of the latest engine execution."""
        return self._infer_profile

    @property
    def is_dynamic_batch(self: Self) -> bool:
        """Check if model has dynamic batch size."""
        return self._engine.is_dynamic_batch

    def get_preprocessor(
        self: Self, method: str | None = None
    ) -> CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor:
        """
        Get the preprocessor for a method, creating it on first use.

        Parameters
        ----------
        method : str, optional
            Options are 'cpu', 'cuda', or 'trt'. 'trt' falls back to 'cuda'
            on platforms without UINT8 support, and 'cuda' to 'cpu' for
            models whose input is not float32. By default None, which
            returns the preprocessor stated in the constructor.

        Returns
        -------
        CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor
            The preprocessor.

        Raises
        ------
        ValueError
            If the method is not 'cpu', 'cuda', or 'trt'.

        """
        method = self._preprocessor_method if method is None else self._resolve_method(method)
        if method == "cuda":
            if self._preproc_cuda is None:
                with Device(self._device):
                    self._preproc_cuda = self._setup_cuda_preproc()
            return self._preproc_cuda
        if method == "trt":
            if self._preproc_trt is None:
                with Device(self._device):
                    self._preproc_trt = self._setup_trt_preproc()
            return self._preproc_trt
        return self._preproc_cpu

    def _resolve_method(self: Self, method: str) -> str:
        """
        Validate a preprocessing method and apply the platform/model fallbacks.

        Raises
        ------
        ValueError
            If the method is not 'cpu', 'cuda', or 'trt'.

        """
        if method not in _PREPROCESSOR_METHODS:
            err_msg = f"Invalid preprocessing method {method!r}, options are: {list(_PREPROCESSOR_METHODS)}"
            raise ValueError(err_msg)
        if method == "trt" and not FLAGS.TRT_HAS_UINT8:
            method = "cuda"
            LOG.warning(
                "Preprocessing method set to TensorRT, but platform doesn't support UINT8, fallback to CUDA."
            )
        if method == "cuda" and self._dtype != np.float32:
            method = "cpu"
        return method

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
        return [self._engine.get_random_input()[self._image_input_index].numpy()]

    def mock_run(
        self: Self,
        images: Sequence[np.ndarray | Buffer] | None = None,
    ) -> list[np.ndarray]:
        """
        Mock an execution of the model.

        Parameters
        ----------
        images : Sequence[np.ndarray | Buffer], optional
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
            # a single entry that already has the engine's rank is a batch
            # tensor; otherwise stack the images into one
            engine_rank = len(self._engine.input_shapes[self._image_input_index])
            if len(images) == 1 and images[0].ndim == engine_rank:
                tensor = images[0]
            else:
                tensor = np.stack(images, axis=0)
            result, _, _ = self.infer([tensor], preprocessed=True, no_copy=True)
        else:
            result = self._engine.mock_execute()

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # mock_run

        return result

    # preprocess overloads
    @overload
    def preprocess(
        self: Self,
        images: np.ndarray | Buffer,
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def preprocess(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        resize: str | None = ...,
        method: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    def preprocess(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
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
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
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

        Raises
        ------
        ValueError
            If the method is not 'cpu', 'cuda', or 'trt'.

        """
        preprocessor = self.get_preprocessor(method)
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["preprocess"])

        # Handle single-image input
        if isinstance(images, (np.ndarray, Buffer)):
            batch_images: Sequence[np.ndarray | Buffer] = [images]
        else:
            batch_images = images

        if verbose:
            LOG.debug(
                f"{self._tag}: Running preprocess, batch_size: {len(batch_images)}, with resize: {resize if resize is not None else self._resize_method}",
            )
            LOG.debug(f"{self._tag}: Using device: {method}")
        t0 = time.perf_counter()
        data = preprocessor(batch_images, resize=resize, no_copy=no_copy, verbose=verbose)
        self._pre_profile = (t0, time.perf_counter())

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # preprocess

        return data

    def prepare_inputs(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        method: str | None = None,
        *,
        preprocessed: bool = False,
        ratios: list[tuple[float, float]] | None = None,
        padding: list[tuple[float, float]] | None = None,
        verbose: bool | None = None,
    ) -> tuple[list[Buffer], list[tuple[float, float]] | None, list[tuple[float, float]] | None]:
        """
        Get the inputs of the engine call for a batch of images.

        Images are preprocessed unless ``preprocessed`` is set. GPU
        preprocessors leave the batch on the device, so it is bound in place;
        the CPU preprocessor's batch, or an already preprocessed one, is
        wrapped as a host Buffer.

        Parameters
        ----------
        images : Sequence[np.ndarray | Buffer]
            The images to preprocess, or a single already preprocessed batch tensor.
        method : str, optional
            The preprocessor to use, as in :meth:`get_preprocessor`.
            By default None, the preprocessor stated in the constructor.
        preprocessed : bool, optional
            Whether ``images`` holds an already preprocessed batch tensor.
        ratios : list[tuple[float, float]], optional
            The ratios of an already preprocessed batch.
        padding : list[tuple[float, float]], optional
            The padding of an already preprocessed batch.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        tuple[list[Buffer], list[tuple[float, float]] | None, list[tuple[float, float]] | None]
            The engine inputs, and the ratios and padding per image.

        Raises
        ------
        ValueError
            If ``preprocessed`` is set and ``images`` is not a single batch tensor.

        """
        if preprocessed:
            if len(images) != 1:
                err_msg = "Preprocessed inputs must be a list containing a single batch tensor."
                raise ValueError(err_msg)
            return self.engine_inputs(host_input(images[0]), None, ratios), ratios, padding
        preprocessor = self.get_preprocessor(method)
        t0 = time.perf_counter()
        if isinstance(preprocessor, GPUImagePreprocessor):
            image, ratios, padding = preprocessor.direct_preproc(
                images, resize=self._resize_method, no_warn=True, verbose=verbose
            )
        else:
            tensor, ratios, padding = preprocessor(
                images, no_copy=not preprocessor.dynamic, verbose=verbose
            )
            image = host_input(tensor)
        self._pre_profile = (t0, time.perf_counter())
        return self.engine_inputs(image, images, ratios), ratios, padding

    def infer(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        method: str | None = None,
        *,
        preprocessed: bool = False,
        ratios: list[tuple[float, float]] | None = None,
        padding: list[tuple[float, float]] | None = None,
        postprocess: bool = False,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[list[np.ndarray], list[tuple[float, float]] | None, list[tuple[float, float]] | None]:
        """
        Run the engine on a batch of images, returning its raw outputs.

        The inputs come from :meth:`prepare_inputs`, which takes the
        ``method``, ``preprocessed``, ``ratios``, ``padding`` and ``verbose``
        arguments.

        Parameters
        ----------
        images : Sequence[np.ndarray | Buffer]
            The images, or a single already preprocessed batch tensor.
        method : str, optional
            The preprocessor to use, as in :meth:`get_preprocessor`.
        preprocessed : bool, optional
            Whether ``images`` holds an already preprocessed batch tensor.
        ratios : list[tuple[float, float]], optional
            The ratios of an already preprocessed batch.
        padding : list[tuple[float, float]], optional
            The padding of an already preprocessed batch.
        postprocess : bool, optional
            Whether the caller postprocesses the outputs right away, in which
            case they are not copied out of the engine unless ``no_copy`` is False.
        no_copy : bool, optional
            Whether to return the engine's host output memory without a copy.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        tuple[list[np.ndarray], list[tuple[float, float]] | None, list[tuple[float, float]] | None]
            The raw outputs, and the ratios and padding per image.

        """
        engine_inputs, ratios, padding = self.prepare_inputs(
            images,
            method,
            preprocessed=preprocessed,
            ratios=ratios,
            padding=padding,
            verbose=verbose,
        )
        if no_copy is None and postprocess:
            no_copy = True
        t0 = time.perf_counter()
        outputs = self._engine(engine_inputs, no_copy=no_copy)
        self._infer_profile = (t0, time.perf_counter())
        return outputs, ratios, padding

    def engine_inputs(
        self: Self,
        image: Buffer,
        _images: Sequence[np.ndarray | Buffer] | None,
        _ratios: list[tuple[float, float]] | None,
        /,
    ) -> list[Buffer]:
        """
        Build the inputs passed to the engine call.

        Override in subclasses that need schema-specific inputs, e.g.
        Detector adds image size / scale factor arrays for DETR models.

        Parameters
        ----------
        image : Buffer
            The preprocessed batch, on the host or the device.
        _images : Sequence[np.ndarray | Buffer] | None
            The original input images, or None when the caller passed an
            already preprocessed batch.
        _ratios : list[tuple[float, float]] | None
            The scaling ratios from preprocessing.

        Returns
        -------
        list[Buffer]
            The inputs to pass to the engine call, in engine order.

        """
        return [image]

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
            return [[o for o in outputs if isinstance(o, np.ndarray)]], True
        return [o for o in outputs if isinstance(o, list)], False

    def _run_core(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
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

        Normalizes single-image input and tuple ratios/padding, runs
        :meth:`infer`, and optionally postprocesses with ``post``, which
        subclasses supply to bridge to their own postprocess signature.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image or batch of images to run inference on.
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
            Whether to return engine outputs and postprocessed results without copies.
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
        try:
            if verbose:
                LOG.debug(f"{self._tag}: run")

            # handle single-image input
            is_single = isinstance(images, (np.ndarray, Buffer))
            batch_images: Sequence[np.ndarray | Buffer] = [images] if is_single else images

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

            if verbose:
                LOG.debug(
                    f"{self._tag}: Running: preprocessed: {preprocessed}, postprocess: {postprocess}",
                )

            outputs, batch_ratios, batch_padding = self.infer(
                batch_images,
                preprocessed=preprocessed,
                ratios=batch_ratios,
                padding=batch_padding,
                postprocess=postprocess,
                no_copy=no_copy,
                verbose=verbose,
            )
            if not postprocess:
                return outputs

            if verbose:
                LOG.debug("Postprocessing outputs")
            if needs_ratios and (batch_ratios is None or batch_padding is None):
                err_msg = "Must pass ratios/padding if postprocessing and passing already preprocessed inputs."
                raise RuntimeError(err_msg)
            result = post(outputs, batch_ratios or [], batch_padding or [], no_copy)
            # unwrap for single-image input
            return result[0] if is_single else result
        finally:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # run

    def _end2end_core(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        *,
        verbose: bool | None,
        post: _PostFn,
        get: Callable[[list[list[np.ndarray]]], list[T]],
    ) -> T | list[T]:
        """
        Shared implementation of the public ``end2end`` method.

        Preprocesses on the CPU or GPU, executes the engine (which handles
        CUDA graph capture/replay), postprocesses with ``post``, then
        extracts final results with ``get``.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image or batch of images to run inference on.
        verbose : bool, optional
            Whether to log additional information.
        post : _PostFn
            Callable bridging to the subclass's postprocess signature.
        get : Callable[[list[list[np.ndarray]]], list[T]]
            Callable extracting final per-image results from postprocessed
            outputs, e.g. ``get_detections`` or ``get_classifications``.

        Returns
        -------
        T | list[T]
            Final results, unwrapped for single-image input.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["end2end"])
        try:
            if verbose:
                LOG.debug(f"{self._tag}: end2end")

            # handle single-image input
            is_single = isinstance(images, (np.ndarray, Buffer))
            batch_images: Sequence[np.ndarray | Buffer] = [images] if is_single else images

            # outputs are consumed by get() inside this call, so skip their copies
            no_copy = True
            outputs, ratios, padding = self.infer(batch_images, no_copy=no_copy, verbose=verbose)
            result = get(post(outputs, ratios or [], padding or [], no_copy))
            # unwrap for single-image input
            return result[0] if is_single else result
        finally:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # end2end
