# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from typing_extensions import TypeGuard

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._image_model import ImageModel
from .interfaces import ClassifierInterface
from .postprocessors import get_classifications, postprocess_classifications
from .preprocessors import CUDAPreprocessor, TRTPreprocessor

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
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
            Only effective with async_v3 backend. Default is None (uses TRTEngine default).
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

    def preprocess(
        self: Self,
        images: list[np.ndarray],
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
        images : list[np.ndarray]
            The images to preprocess.
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
        return data

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

    def __call__(
        self: Self,
        images: list[np.ndarray],
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
        images : list[np.ndarray]
            The images to run the model on.
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
        list[list[np.ndarray]]
            The postprocessed outputs per image.

        """
        return self.run(
            images,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
        )

    def run(
        self: Self,
        images: list[np.ndarray],
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
        images : list[np.ndarray]
            The images to run the model on.
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
        list[list[np.ndarray]]
            The postprocessed outputs per image.

        Raises
        ------
        ValueError
            If preprocessed inputs are not a single batch tensor.

        """
        if verbose:
            LOG.debug(f"{self._tag}: run")

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
            postprocessed = self.postprocess(outputs, no_copy=no_copy_post, verbose=verbose)
            self._infer_profile = (t0, t1)
            return postprocessed

        self._infer_profile = (t0, t1)

        return outputs

    def get_classifications(
        self: Self,
        outputs: list[list[np.ndarray]],
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[list[tuple[int, float]]]:
        """
        Get the classifications from postprocessed outputs.

        Parameters
        ----------
        outputs : list[list[np.ndarray]]
            The postprocessed outputs per image.
        top_k : int, optional
            The number of top predictions to return. Default is 5.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[tuple[int, float]]]
            The classifications per image, where each entry is (class_id, confidence).

        """
        if verbose:
            LOG.debug(f"{self._tag}: get_classifications")

        return get_classifications(outputs, top_k=top_k, verbose=verbose)

    def end2end(
        self: Self,
        images: list[np.ndarray],
        top_k: int = 5,
        *,
        verbose: bool | None = None,
    ) -> list[list[tuple[int, float]]]:
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_classifications in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        images : list[np.ndarray]
            The images to perform inference with.
        top_k : int, optional
            The number of top predictions to return. Default is 5.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[tuple[int, float]]]
            The classifications per image, where each entry is (class_id, confidence).

        Raises
        ------
        RuntimeError
            If postprocessed outputs are not available in end2end.

        """
        if verbose:
            LOG.debug(f"{self._tag}: end2end")

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
        return self.get_classifications(postprocessed, top_k=top_k, verbose=verbose)
