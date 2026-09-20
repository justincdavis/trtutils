# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, overload

import nvtx
from typing_extensions import Literal

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._image_model import ImageModel
from .interfaces import ClassifierInterface
from .postprocessors import get_classifications, postprocess_classifications

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self


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
        device : int, optional
            The CUDA device index to use for this classifier. Default is None,
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
            device=device,
            backend=backend,
            warmup=warmup,
            pagelocked_mem=pagelocked_mem,
            unified_mem=unified_mem,
            cuda_graph=cuda_graph,
            no_warn=no_warn,
            verbose=verbose,
        )

        # prepend with 'cls_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "cls_postprocess": f"classifier::postprocess [{self._tag}]",
                "cls_get_classifications": f"classifier::get_classifications [{self._tag}]",
            }
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
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["cls_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        t0 = time.perf_counter()
        data = postprocess_classifications(outputs, no_copy=no_copy, verbose=verbose)
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

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
        return self._run_core(
            images,
            None,
            None,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
            post=lambda o, _r, _p, nc: self.postprocess(o, no_copy=nc, verbose=verbose),
            needs_ratios=False,
        )

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
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["cls_get_classifications"])

        if verbose:
            LOG.debug(f"{self._tag}: get_classifications")

        batch, single = self._as_batch(outputs)
        result = get_classifications(batch, top_k=top_k, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_classifications

        return result[0] if single else result

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
            If end2end_graph is enabled and image dimensions change after first call.
        RuntimeError
            If end2end_graph is enabled and CUDA graph capture fails.

        """
        return self._end2end_core(
            images,
            verbose=verbose,
            post=lambda o, _r, _p, nc: self.postprocess(o, no_copy=nc, verbose=verbose),
            get=lambda pp: get_classifications(pp, top_k=top_k, verbose=verbose),
            needs_ratios=False,
        )
