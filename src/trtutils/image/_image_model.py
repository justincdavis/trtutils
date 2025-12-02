# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trtutils._engine import TRTEngine
from trtutils._flags import FLAGS
from trtutils._log import LOG

from .preprocessors import CPUPreprocessor, CUDAPreprocessor, TRTPreprocessor

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
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
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
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        pagelocked_mem : bool, optional
            Whether or not to use pagelocked memory for underlying CUDA operations.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
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
        if self._verbose:
            LOG.debug(f"Creating ImageModel: {self._tag}")

        self._pagelocked_mem = pagelocked_mem if pagelocked_mem is not None else True
        self._engine = TRTEngine(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
            dla_core=dla_core,
            pagelocked_mem=self._pagelocked_mem,
            unified_mem=unified_mem,
            no_warn=no_warn,
            verbose=verbose,
        )
        self._unified_mem = self._engine.unified_mem
        self._resize_method: str = resize_method
        input_spec = self._engine.input_spec[0]
        input_size: tuple[int, ...] = tuple(input_spec[0])

        expected_input_size = 4
        if len(input_size) != expected_input_size:
            err_msg = "Expected model to have input size of form: (batch, channels, height, width)"
            err_msg += f", found {input_size}"
            raise ValueError(err_msg)
        rgb_channels = 3
        if input_size[1] != rgb_channels:
            err_msg = f"Expected model to take {rgb_channels} channel input, found {input_size[1]}"
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

        # set up the preprocessor
        self._preprocessor: CPUPreprocessor | CUDAPreprocessor | TRTPreprocessor
        valid_preprocessors = ["cpu", "cuda", "trt"]
        if preprocessor not in valid_preprocessors:
            err_msg = f"Invalid preprocessor found, options are: {valid_preprocessors}"
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
        # existing logic
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

    def _setup_cpu_preproc(self: Self) -> CPUPreprocessor:
        return CPUPreprocessor(
            self._input_size,
            self._input_range,
            self._dtype,
            mean=self._mean,
            std=self._std,
            tag=self._tag,
        )

    def _setup_cuda_preproc(self: Self) -> CUDAPreprocessor:
        return CUDAPreprocessor(
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

    def _setup_trt_preproc(self: Self) -> TRTPreprocessor:
        return TRTPreprocessor(
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
        if images is not None:
            # Stack images into batch tensor if provided as list
            if len(images) == 1:
                return self._engine.mock_execute(data=[images[0]])
            batch_tensor = np.stack(images, axis=0)
            return self._engine.mock_execute(data=[batch_tensor])
        return self._engine.mock_execute()
