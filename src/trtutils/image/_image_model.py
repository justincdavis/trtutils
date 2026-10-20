# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, TypeVar, overload

import numpy as np
import nvtx

from trtutils._engine import TRTEngine
from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._buffer import Buffer
from trtutils.core._device import Device

from .preprocessors import (
    CPUPreprocessor,
    CUDAPreprocessor,
    GPUImagePreprocessor,
    TRTPreprocessor,
)

if TYPE_CHECKING:
    from typing_extensions import Self

_COLOR_CHANNELS = 3

# (raw_outputs, ratios, padding, no_copy) -> postprocessed outputs per image
_PostFn = Callable[
    [List[np.ndarray], List[Tuple[float, float]], List[Tuple[float, float]], Optional[bool]],
    List[List[np.ndarray]],
]
T = TypeVar("T")


def host_input(array: np.ndarray) -> Buffer:
    """
    Wrap host data as an engine input.

    ``np.ascontiguousarray`` returns ``array`` itself when it is already
    C-contiguous, so the common case is zero-copy; only strided or
    Fortran-ordered data is gathered into a new array.
    """
    return Buffer.wrap(np.ascontiguousarray(array))


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

        # Extract batch size from engine (the max profile batch for a
        # dynamic-batch engine, which bounds what the preprocessors accept)
        self._batch_size: int = input_size[0]

        self._input_size: tuple[int, int] = (input_size[3], input_size[2])
        self._dtype = input_spec[1]
        self._input_range = input_range
        self._mean = mean
        self._std = std

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
            on platforms without UINT8 support. By default None, which
            returns the preprocessor stated in the constructor.

        Returns
        -------
        CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor
            The preprocessor.

        """
        if method is None:
            return self._preprocessor
        if method == "trt" and not FLAGS.TRT_HAS_UINT8:
            method = "cuda"
            LOG.warning(
                "Preprocessing method set to TensorRT, but platform doesn't support UINT8, fallback to CUDA."
            )
        if method == "cuda":
            if self._preproc_cuda is None:
                self._preproc_cuda = self._setup_cuda_preproc()
            return self._preproc_cuda
        if method == "trt":
            if self._preproc_trt is None:
                self._preproc_trt = self._setup_trt_preproc()
            return self._preproc_trt
        return self._preproc_cpu

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
        return [self._engine.get_random_input()[0].numpy()]

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
            # a single entry that already has the engine's rank is a batch
            # tensor; otherwise stack the images into one
            engine_rank = len(self._engine.input_shapes[0])
            if len(images) == 1 and images[0].ndim == engine_rank:
                tensor = images[0]
            else:
                tensor = np.stack(images, axis=0)
            result = self._engine.mock_execute(data=[host_input(tensor)])
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
        if isinstance(images, np.ndarray):
            batch_images: list[np.ndarray] = [images]
        else:
            batch_images = images

        resize = resize if resize is not None else self._resize_method
        if verbose:
            LOG.debug(
                f"{self._tag}: Running preprocess, batch_size: {len(batch_images)}, with method: {resize}",
            )
            LOG.debug(f"{self._tag}: Using device: {method}")
        preprocessor = self.get_preprocessor(method)
        t0 = time.perf_counter()
        data = preprocessor(batch_images, resize=resize, no_copy=no_copy, verbose=verbose)
        self._pre_profile = (t0, time.perf_counter())

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # preprocess

        return data

    def prepare_inputs(
        self: Self,
        images: list[np.ndarray],
        method: str | None = None,
        *,
        verbose: bool | None = None,
    ) -> tuple[list[Buffer], list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess images into the inputs of the engine call.

        GPU preprocessors leave the batch on the device, so it is bound in
        place; the CPU preprocessor's batch is wrapped as a host Buffer.

        Parameters
        ----------
        images : list[np.ndarray]
            The images to preprocess.
        method : str, optional
            The preprocessor to use, as in :meth:`get_preprocessor`.
            By default None, the preprocessor stated in the constructor.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        tuple[list[Buffer], list[tuple[float, float]], list[tuple[float, float]]]
            The engine inputs, and the ratios and padding per image.

        """
        preprocessor = self.get_preprocessor(method)
        t0 = time.perf_counter()
        if isinstance(preprocessor, GPUImagePreprocessor):
            image, ratios, padding = preprocessor.direct_preproc(
                images, resize=self._resize_method, no_warn=True, verbose=verbose
            )
        else:
            tensor, ratios, padding = preprocessor(
                images, resize=self._resize_method, verbose=verbose
            )
            image = host_input(tensor)
        self._pre_profile = (t0, time.perf_counter())
        return self.engine_inputs(image, images, ratios), ratios, padding

    def engine_inputs(
        self: Self,
        image: Buffer,
        _images: list[np.ndarray] | None,
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
        _images : list[np.ndarray] | None
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
        images: np.ndarray | list[np.ndarray],
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
        images : np.ndarray | list[np.ndarray]
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
        if isinstance(images, np.ndarray):
            batch_images: list[np.ndarray] = [images]
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
            # skip the output copy when the outputs are postprocessed inside this call
            no_copy_run: bool | None = True
            no_copy_post: bool | None = False
        else:
            no_copy_run = no_copy
            no_copy_post = no_copy

        if verbose:
            LOG.debug(
                f"{self._tag}: Running: preprocessed: {preprocessed}, postprocess: {postprocess}",
            )

        # handle preprocessing
        if not preprocessed:
            if verbose:
                LOG.debug("Preprocessing inputs")
            engine_inputs, batch_ratios, batch_padding = self.prepare_inputs(
                batch_images, verbose=verbose
            )
        else:
            # images is already preprocessed tensor when preprocessed=True
            if len(batch_images) != 1:
                err_msg = "Preprocessed inputs must be a list containing a single batch tensor."
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # run
                raise ValueError(err_msg)
            engine_inputs = self.engine_inputs(host_input(batch_images[0]), None, batch_ratios)

        # execute
        t0 = time.perf_counter()
        outputs: list[np.ndarray] = self._engine(engine_inputs, no_copy=no_copy_run)
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
        images: np.ndarray | list[np.ndarray],
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
        images : np.ndarray | list[np.ndarray]
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

        if verbose:
            LOG.debug(f"{self._tag}: end2end")

        # handle single-image input
        if isinstance(images, np.ndarray):
            batch_images: list[np.ndarray] = [images]
            is_single = True
        else:
            batch_images = images
            is_single = False

        engine_inputs, ratios, padding = self.prepare_inputs(batch_images, verbose=verbose)
        t0 = time.perf_counter()
        raw = self._engine.execute(engine_inputs, no_copy=True)
        self._infer_profile = (t0, time.perf_counter())
        # outputs are consumed by get() inside this call, so skip their copies
        no_copy_post = True
        pp = post(raw, ratios, padding, no_copy_post)
        result = get(pp)

        # unwrap for single-image input
        if is_single:
            if FLAGS.NVTX_ENABLED:
                nvtx.pop_range()  # end2end
            return result[0]

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # end2end

        return result
