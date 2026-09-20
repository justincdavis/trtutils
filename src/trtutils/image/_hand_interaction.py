# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, overload

import numpy as np
import nvtx
from typing_extensions import Literal

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._image_model import ImageModel
from .interfaces import HandInteractionDetectorInterface
from .postprocessors import get_interactions, postprocess_hand_interactions
from .postprocessors._cuda import CUDAPostprocessor

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self

    from .postprocessors import HandInteraction

_EXPECTED_OUTPUT_PREFIX = ["boxes", "scores", "labels", "pair_probs"]


class HandInteractionDetector(ImageModel, HandInteractionDetectorInterface):
    """
    Implementation of hand-object interaction detectors.

    Wraps engines for hand-object interaction model families (e.g. HOI-DETR,
    Hands23) that follow a unified, fixed-K output contract:
    ``boxes (B,K,4)`` xyxy in network-input pixels, ``scores (B,K)``,
    ``labels (B,K)`` (0 hand, 1 object, 2 second object), ``pair_probs
    (B,K,K,C)`` where ``[...,0]`` is the no-interaction class, and an
    optional ``side (B,K)`` (0 left, 1 right, valid where label is hand).
    All model-specific heads (pairing MLP, attribute heads, touch gating)
    are expected to run in-graph so a single postprocessor handles every
    model in the family. Letterbox padding is centered and is undone by
    the postprocessor using the ratios/padding returned from preprocessing.
    """

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.3,
        pair_thres: float = 0.5,
        second_pair_thres: float | None = None,
        nms_iou_thres: float = 0.5,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        postprocessor: str | None = None,
        cuda_graph: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a HandInteractionDetector object.

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
            The options are ['letterbox', 'linear'], default is 'letterbox'.
        conf_thres : float
            The confidence threshold used to filter candidate boxes and as
            the NMS score threshold. By default 0.3.
        pair_thres : float
            The minimum link probability to link a hand to a first object.
            By default 0.5.
        second_pair_thres : float, optional
            The minimum link probability to link a first object to a second
            object. By default None, which uses pair_thres.
        nms_iou_thres : float
            The IOU threshold to use for the class-aware NMS operation.
            By default 0.5.
        mean : tuple[float, float, float] | None, optional
            The mean values to use for normalization.
            By default, None, which means no normalization will be applied.
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for normalization.
            By default, None, which means no normalization will be applied.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        device : int, optional
            The CUDA device index to use for this detector. Default is None,
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
        postprocessor : str, optional
            The type of postprocessor to use for end2end().
            The options are ['cpu', 'cuda']. Default is None, which means
            'cpu'. With 'cuda', the raw engine outputs are postprocessed on
            the GPU and only the compacted results are copied to the host.
            Only effective with end2end(); run() always uses CPU postprocessing.
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

        Raises
        ------
        ValueError
            If the engine outputs do not follow the unified hand-object
            interaction output contract.

        """
        # stored before super().__init__ alongside the other model-specific state,
        # matching how Detector stashes its schema overrides before engine creation
        self._conf_thres: float = conf_thres
        self._pair_thres: float = pair_thres
        self._second_pair_thres: float | None = second_pair_thres
        self._nms_iou_thres: float = nms_iou_thres

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
            postprocessor=postprocessor,
            cuda_graph=cuda_graph,
            no_warn=no_warn,
            verbose=verbose,
        )

        # prepend with 'hoi_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "hoi_postprocess": f"hand_interaction_detector::postprocess [{self._tag}]",
                "hoi_get_interactions": f"hand_interaction_detector::get_interactions [{self._tag}]",
            }
        )

    def _configure_model(self: Self) -> None:
        """Validate the engine follows the unified hand-object interaction contract."""
        names = list(self._engine.output_names)
        if names[:4] != _EXPECTED_OUTPUT_PREFIX or names[4:] not in ([], ["side"]):
            err_msg = (
                "Expected hand interaction engine outputs to start with "
                f"{_EXPECTED_OUTPUT_PREFIX} and optionally be followed by 'side', "
                f"found: {names}"
            )
            raise ValueError(err_msg)
        self._has_side: bool = len(names) == 5  # noqa: PLR2004

    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None = None,
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
        ratios : list[tuple[float, float]]
            The rescale ratios used during preprocessing for each image.
        padding : list[tuple[float, float]]
            The padding values used during preprocessing for each image.
        conf_thres : float, optional
            The confidence threshold to filter candidates by.
            If not passed, will use the value from the constructor.
        no_copy : bool, optional
            Kept for interface symmetry with the other postprocessors.
            NMS-based indexing always allocates new arrays.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs per image: [bboxes, scores, labels,
            pair_probs, side].

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["hoi_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        conf_thres = conf_thres if conf_thres is not None else self._conf_thres
        t0 = time.perf_counter()
        data = postprocess_hand_interactions(
            outputs,
            ratios,
            padding,
            conf_thres,
            self._nms_iou_thres,
            no_copy=no_copy,
            verbose=verbose,
        )
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

        return data

    def _create_cuda_postprocessor(self: Self) -> CUDAPostprocessor | None:
        """Create the CUDAPostprocessor for hand interaction outputs."""
        specs = self._engine.output_spec
        try:
            return CUDAPostprocessor(
                specs,
                "hand",
                specs[0][0][1],
                stream=self._engine.stream,
                tag=self._tag,
                pagelocked_mem=self._pagelocked_mem,
                unified_mem=self._unified_mem,
                verbose=self._verbose,
            )
        except ValueError as e:
            LOG.warning(f"{self._tag}: CUDA postprocessing unavailable: {e}")
            return None

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray,
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
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
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    def __call__(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        ratios: tuple[float, float] | list[tuple[float, float]] | None = None,
        padding: tuple[float, float] | list[tuple[float, float]] | None = None,
        conf_thres: float | None = None,
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
        ratios : tuple[float, float] | list[tuple[float, float]], optional
            The ratios generated during preprocessing. For single image, pass tuple.
            For batch, pass list.
        padding : tuple[float, float] | list[tuple[float, float]], optional
            The padding values used during preprocessing. For single image, pass tuple.
            For batch, pass list.
        conf_thres : float, optional
            Optional confidence threshold to filter candidates by during
            postprocessing.
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
            The outputs. For single image with postprocess=True,
            returns list[np.ndarray]. For batch, returns batch results.

        """
        return self.run(  # ty: ignore[no-matching-overload]
            images,
            ratios,
            padding,
            conf_thres,
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
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
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
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
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
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
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
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
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
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
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
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    def run(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        ratios: tuple[float, float] | list[tuple[float, float]] | None = None,
        padding: tuple[float, float] | list[tuple[float, float]] | None = None,
        conf_thres: float | None = None,
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
        ratios : tuple[float, float] | list[tuple[float, float]], optional
            The ratios generated during preprocessing. For single image, pass tuple.
            For batch, pass list.
        padding : tuple[float, float] | list[tuple[float, float]], optional
            The padding values used during preprocessing. For single image, pass tuple.
            For batch, pass list.
        conf_thres : float, optional
            Optional confidence threshold to filter candidates by during
            postprocessing.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If None, will preprocess inputs.
        postprocess : bool, optional
            Whether or not to postprocess the outputs.
            If None, will postprocess outputs.
            If postprocessing will occur and the inputs were
            passed already preprocessed, then the ratios and
            padding must be passed for postprocessing.
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
        RuntimeError
            If postprocessing is requested for already-preprocessed inputs
            without passing ratios/padding.

        """
        return self._run_core(
            images,
            ratios,
            padding,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
            post=lambda o, r, p, nc: self.postprocess(
                o, r, p, conf_thres, no_copy=nc, verbose=verbose
            ),
        )

    # get_interactions overloads
    @overload
    def get_interactions(
        self: Self,
        outputs: list[np.ndarray],
        pair_thres: float | None = ...,
        second_pair_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[HandInteraction]: ...

    @overload
    def get_interactions(
        self: Self,
        outputs: list[list[np.ndarray]],
        pair_thres: float | None = ...,
        second_pair_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[HandInteraction]]: ...

    def get_interactions(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        pair_thres: float | None = None,
        second_pair_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[HandInteraction] | list[list[HandInteraction]]:
        """
        Get the hand-object interactions from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        pair_thres : float, optional
            The minimum link probability to link a hand to a first object.
            By default None, which uses the value from the constructor.
        second_pair_thres : float, optional
            The minimum link probability to link a first object to a second
            object. By default None, which uses the value from the constructor.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[HandInteraction] | list[list[HandInteraction]]
            For single image: list[HandInteraction] (interactions for single image).
            For batch: list[list[HandInteraction]] (interactions per image).

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["hoi_get_interactions"])

        if verbose:
            LOG.debug(f"{self._tag}: get_interactions")

        pair_thres = pair_thres if pair_thres is not None else self._pair_thres
        second_thres = (
            second_pair_thres if second_pair_thres is not None else self._second_pair_thres
        )

        batch, single = self._as_batch(outputs)
        result = get_interactions(batch, pair_thres, second_thres, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_interactions

        return result[0] if single else result

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray,
        *,
        conf_thres: float | None = ...,
        pair_thres: float | None = ...,
        second_pair_thres: float | None = ...,
        verbose: bool | None = ...,
    ) -> list[HandInteraction]: ...

    @overload
    def end2end(
        self: Self,
        images: list[np.ndarray],
        *,
        conf_thres: float | None = ...,
        pair_thres: float | None = ...,
        second_pair_thres: float | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[HandInteraction]]: ...

    def end2end(
        self: Self,
        images: np.ndarray | list[np.ndarray],
        *,
        conf_thres: float | None = None,
        pair_thres: float | None = None,
        second_pair_thres: float | None = None,
        verbose: bool | None = None,
    ) -> list[HandInteraction] | list[list[HandInteraction]]:
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_interactions in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        images : np.ndarray | list[np.ndarray]
            A single image (HWC format) or list of images to perform inference with.
        conf_thres : float, optional
            The confidence threshold to filter candidates by.
            By default None, which uses the value from the constructor.
        pair_thres : float, optional
            The minimum link probability to link a hand to a first object.
            By default None, which uses the value from the constructor.
        second_pair_thres : float, optional
            The minimum link probability to link a first object to a second
            object. By default None, which uses the value from the constructor.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[HandInteraction] | list[list[HandInteraction]]
            For single image: list[HandInteraction] (interactions).
            For batch: list[list[HandInteraction]] (interactions per image).

        Raises
        ------
        RuntimeError
            If end2end_graph is enabled and image dimensions change after first call.
        RuntimeError
            If end2end_graph is enabled and CUDA graph capture fails.

        """
        post_cuda: (
            Callable[[list[tuple[float, float]], list[tuple[float, float]]], list[list[np.ndarray]]]
            | None
        ) = None
        if self._cuda_postproc is not None:
            cp = self._cuda_postproc
            batch = 1 if isinstance(images, np.ndarray) else len(images)
            use_conf = conf_thres if conf_thres is not None else self._conf_thres

            def post_cuda(
                r: list[tuple[float, float]], p: list[tuple[float, float]]
            ) -> list[list[np.ndarray]]:
                ptrs = [o.allocation for o in self._engine._outputs]  # noqa: SLF001
                return cp.postprocess_hand_interactions(
                    batch, ptrs, r, p, use_conf, self._nms_iou_thres, verbose=verbose
                )

        return self._end2end_core(
            images,
            verbose=verbose,
            post=lambda o, r, p, nc: self.postprocess(
                o, r, p, conf_thres, no_copy=nc, verbose=verbose
            ),
            get=lambda pp: self.get_interactions(pp, pair_thres, second_pair_thres, verbose=verbose),
            post_cuda=post_cuda,
        )
