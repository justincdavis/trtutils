# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, overload

import numpy as np
from typing_extensions import Literal, TypeGuard

from trtutils._log import LOG

from ._image_model import ImageModel
from .interfaces import ClassifierInterface
from .postprocessors import get_classifications, postprocess_classifications
from .preprocessors import CUDAPreprocessor, TRTPreprocessor
from .preprocessors._image_preproc import _is_single_image

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


def _is_postprocessed_outputs(
    outputs: list[np.ndarray] | list[list[np.ndarray]],
) -> TypeGuard[list[list[np.ndarray]]]:
    return not outputs or isinstance(outputs[0], list)


class Classifier(ImageModel, ClassifierInterface):
    """Implementation of image classifiers."""

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
        Create a Classifier object.

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
            Only effective with async_v3 backend. Default is True.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        """
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            resize_method=resize_method,
            mean=mean,
            std=std,
            dla_core=dla_core,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            no_warn=no_warn,
            verbose=verbose,
        )

    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Postprocess the outputs.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The raw outputs from the engine to postprocess.
        no_copy : bool, optional
            If True, do not copy the data from the allocated
            memory. If the data is not copied, it WILL BE
            OVERWRITTEN INPLACE once new data is generated.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs per image.

        """
        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        t0 = time.perf_counter()
        data = postprocess_classifications(outputs, no_copy=no_copy, verbose=verbose)
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)
        return data

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def __call__(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    def __call__(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """
        Run the model on input.

        Parameters
        ----------
        images : np.ndarray | list[np.ndarray]
            A single image (HWC format) or list of images to run the model on.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If None, will preprocess inputs.
        postprocess : bool, optional
            Whether or not to postprocess the outputs.
            If None, will postprocess outputs.
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the cuda allocated host memory. Instead,
            the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE by
            future inferences.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[np.ndarray] | list[list[np.ndarray]]
            The outputs. For single image input with postprocess=True,
            returns list[np.ndarray]. For batch input, returns batch results.

        """
        return self.run(
            images,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
        )

    # run overloads - batch input (3 overloads)
    @overload
    def run(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[np.ndarray]]: ...

    @overload
    def run(
        self: Self,
        images: list[np.ndarray],
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    # run overloads - single image input (3 overloads)
    @overload
    def run(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: np.ndarray,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    def run(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        *,
        preprocessed: bool | None = None,
        postprocess: bool | None = None,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[np.ndarray] | list[list[np.ndarray]]:
        """
        Run the model on input.

        Parameters
        ----------
        images : np.ndarray | list[np.ndarray]
            A single image (HWC format) or list of images to run the model on.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If None, will preprocess inputs.
        postprocess : bool, optional
            Whether or not to postprocess the outputs.
            If None, will postprocess outputs.
        no_copy : bool, optional
            If True, the outputs will not be copied out
            from the cuda allocated host memory. Instead,
            the host memory will be returned directly.
            This memory WILL BE OVERWRITTEN INPLACE by
            future inferences.
            In special case where, preprocessing and
            postprocessing will occur during run and no_copy
            was not passed (is None), then no_copy will be used
            for preprocessing and inference stages.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[np.ndarray] | list[list[np.ndarray]]
            For single image with postprocess=True: list[np.ndarray] (single image outputs).
            For batch with postprocess=True: list[list[np.ndarray]] (per-image outputs).
            For postprocess=False: list[np.ndarray] (raw outputs).

        Raises
        ------
        ValueError
            If preprocessed inputs are not a single batch tensor.

        """
        if verbose:
            LOG.debug(f"{self._tag}: run")

        # Handle single-image input
        is_single = _is_single_image(images)
        if is_single:
            images = [images]  # type: ignore[list-item]

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

        # handle preprocessing
        if not preprocessed:
            if verbose:
                LOG.debug("Preprocessing inputs")
            tensor, _, _ = self.preprocess(images, no_copy=no_copy_pre)
        else:
            # images is already preprocessed tensor when preprocessed=True
            if len(images) != 1:
                err_msg = "Preprocessed inputs must be a list containing a single batch tensor."
                raise ValueError(err_msg)
            tensor = images[0]

        # execute
        t0 = time.perf_counter()
        outputs: list[np.ndarray] = self._engine([tensor], no_copy=no_copy_run)
        t1 = time.perf_counter()

        # handle postprocessing
        if postprocess:
            if verbose:
                LOG.debug("Postprocessing outputs")
            postprocessed_outputs = self.postprocess(outputs, no_copy=no_copy_post, verbose=verbose)
            self._infer_profile = (t0, t1)

            # Unwrap for single-image input
            if is_single:
                return postprocessed_outputs[0]
            return postprocessed_outputs

        self._infer_profile = (t0, t1)

        return outputs

    # get_classifications overloads
    @overload
    def get_classifications(
        self: Self,
        outputs: list[np.ndarray],
        top_k: int = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[tuple[int, float]]: ...

    @overload
    def get_classifications(
        self: Self,
        outputs: list[list[np.ndarray]],
        top_k: int = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[tuple[int, float]]]: ...

    def get_classifications(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[tuple[int, float]] | list[list[tuple[int, float]]]:
        """
        Get the classifications from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        top_k : int, optional
            The number of top predictions to return. Default is 5.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[tuple[int, float]] | list[list[tuple[int, float]]]
            For single image: list[tuple[int, float]] (classifications for single image).
            For batch: list[list[tuple[int, float]]] (classifications per image).

        """
        if verbose:
            LOG.debug(f"{self._tag}: get_classifications")

        # Detect if this is single-image output (list[np.ndarray]) vs batch (list[list[np.ndarray]])
        is_single = outputs and isinstance(outputs[0], np.ndarray)

        if is_single:
            # Wrap single image outputs for batch processing
            batch_outputs: list[list[np.ndarray]] = [outputs]  # type: ignore[list-item]
            result = get_classifications(batch_outputs, top_k=top_k, verbose=verbose)
            return result[0]  # Unwrap

        return get_classifications(outputs, top_k=top_k, verbose=verbose)  # type: ignore[arg-type]

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray,
        top_k: int = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[tuple[int, float]]: ...

    @overload
    def end2end(
        self: Self,
        images: list[np.ndarray],
        top_k: int = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[tuple[int, float]]]: ...

    def end2end(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[tuple[int, float]] | list[list[tuple[int, float]]]:
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_classifications in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        images : np.ndarray | list[np.ndarray]
            A single image (HWC format) or list of images to perform inference with.
        top_k : int, optional
            The number of top predictions to return. Default is 5.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[tuple[int, float]] | list[list[tuple[int, float]]]
            For single image: list[tuple[int, float]] (classifications).
            For batch: list[list[tuple[int, float]]] (classifications per image).

        Raises
        ------
        RuntimeError
            If postprocessed outputs are not available in end2end.
        RuntimeError
            If end2end_graph is enabled and image dimensions change after first call.
        RuntimeError
            If end2end_graph is enabled and CUDA graph capture fails.

        """
        if verbose:
            LOG.debug(f"{self._tag}: end2end")

        # Handle single-image input
        is_single = _is_single_image(images)
        if is_single:
            images = [images]  # type: ignore[list-item]

        # Dispatch based on graph flag
        if self._e2e_graph_enabled:
            result = self._end2end_graph(
                images,  # type: ignore[arg-type]
                top_k=top_k,
                verbose=verbose,
            )
        else:
            result = self._end2end(
                images,  # type: ignore[arg-type]
                top_k=top_k,
                verbose=verbose,
            )

        # Unwrap for single-image input
        if is_single:
            return result[0]
        return result

    def _end2end(
        self: Self,
        images: list[np.ndarray],
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[list[tuple[int, float]]]:
        """Execute the standard end2end path without graph capture."""
        outputs: list[np.ndarray] | list[list[np.ndarray]]
        # if using CPU preprocessor best you can do is remove host-to-host copies
        if not isinstance(self._preprocessor, (CUDAPreprocessor, TRTPreprocessor)):
            if verbose:
                LOG.debug(f"{self._tag}: end2end -> calling CPU preprocess")

            outputs = self.run(
                images,
                preprocessed=False,
                postprocess=True,
                no_copy=True,
                verbose=verbose,
            )
            if not _is_postprocessed_outputs(outputs):
                err_msg = "Expected postprocessed classifier outputs in end2end."
                raise RuntimeError(err_msg)
            postprocessed = outputs
        else:
            if verbose:
                LOG.debug(f"{self._tag}: end2end -> calling CUDA preprocess")

            # if using CUDA, can remove much more
            gpu_ptr, _, _ = self._preprocessor.direct_preproc(
                images,
                resize=self._resize_method,
                no_warn=True,
                verbose=verbose,
            )
            raw_outputs = self._engine.direct_exec([gpu_ptr], no_warn=True)
            postprocessed = self.postprocess(raw_outputs, no_copy=True, verbose=verbose)

        # generate the classifications
        return get_classifications(postprocessed, top_k=top_k, verbose=verbose)

    def _end2end_graph(
        self: Self,
        images: list[np.ndarray],
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[list[tuple[int, float]]]:
        """
        Execute graph-accelerated end2end path.

        This implementation captures only TRTEngine inference in the CUDA graph.
        Preprocessing runs outside the graph since H2D copies cannot be captured.
        Supports CPU, CUDA, and TRT preprocessors.
        """
        # Use shared core graph execution
        raw_outputs, _, _ = self._end2end_graph_core(images, verbose=verbose)

        # CPU postprocessing (Classifier-specific)
        postprocessed = self.postprocess(raw_outputs, no_copy=True, verbose=verbose)
        return get_classifications(postprocessed, top_k=top_k, verbose=verbose)
