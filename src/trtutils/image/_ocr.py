# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, overload

import cv2
import numpy as np
import nvtx
from typing_extensions import Literal

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._buffer import Buffer

from ._image_model import ImageModel
from ._schema import TextDetectorOutputSchema, TextRecognizerOutputSchema, resolve_output_schema
from .interfaces import TextDetectorInterface, TextRecognizerInterface
from .postprocessors import (
    get_text_boxes,
    get_texts,
    postprocess_text_detection,
    postprocess_text_recognition,
)
from .preprocessors import (
    CPUPreprocessor,
    crop_text_region,
    preprocess_text_detection,
    preprocess_text_recognition,
    resize_text_detection,
)
from .preprocessors._ocr import _IMAGENET_MEAN, _IMAGENET_STD, _REC_MEAN, _REC_STD

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from trtutils.image.types import Text, TextBox, TextLine

    from ._image_model import _PostFn


class TextDetector(ImageModel, TextDetectorInterface):
    """Implementation of DB (Differentiable Binarization) text detectors, as used by PaddleOCR."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "cpu",
        resize_method: str = "linear",
        conf_thres: float = 0.6,
        limit_side_len: int = 960,
        limit_type: str = "max",
        thresh: float = 0.3,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        mean: tuple[float, float, float] | None = _IMAGENET_MEAN,
        std: tuple[float, float, float] | None = _IMAGENET_STD,
        output_schema: TextDetectorOutputSchema | str | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,  # noqa: ARG002
        use_dilation: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a TextDetector object.

        The engine should be built with a (min, opt, max) optimization profile over
        the input height and width, see :meth:`trtutils.models.PPOCRDetector.build`.
        The defaults are PaddleOCR's (``tools/infer/utility.py``): longer side
        limited to 960, ImageNet normalization, box threshold 0.6, unclip 1.5,
        no dilation. RapidOCR instead uses ``limit_side_len=736, limit_type="min"``,
        mean/std 0.5, ``conf_thres=0.5``, ``unclip_ratio=1.6`` and dilation.

        Parameters
        ----------
        engine_path : Path, str
            The Path or str to the compiled TensorRT engine.
        warmup_iterations : int
            The number of warmup iterations to perform.
            The default is 10.
        input_range : tuple[float, float]
            The range of input values which should be passed to
            the model, used when mean and std are None. By default [0.0, 1.0].
        preprocessor : str
            The type of preprocessor to use. Only 'cpu' is supported,
            any other value raises a ValueError. Default is 'cpu'.
        resize_method : str
            Unused, images are resized by ``limit_side_len`` and ``limit_type``.
            Kept for parity with the other image models, default is 'linear'.
        conf_thres : float
            The minimum mean probability inside a box to keep it. By default 0.6.
        limit_side_len : int
            Images are resized so their shorter (``limit_type="min"``) or longer
            (``limit_type="max"``) side matches this length, then both sides are
            rounded to a multiple of 32. Images resizing beyond the engine's max
            shape are scaled down to fit it. By default 960.
        limit_type : str
            The options are ['min', 'max'], default is 'max'.
        thresh : float
            The binarization threshold of the probability map. By default 0.3.
        max_candidates : int
            The maximum number of candidate regions. By default 1000.
        unclip_ratio : float
            How far to expand detected text kernels. By default 1.5.
        mean : tuple[float, float, float] | None, optional
            The mean values to use for the normalization, in BGR order.
            By default the ImageNet mean (0.485, 0.456, 0.406).
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for the normalization, in BGR
            order. By default the ImageNet std (0.229, 0.224, 0.225).
        output_schema : TextDetectorOutputSchema, str, optional
            Manually specify the output schema instead of auto-detection.
            Can be a TextDetectorOutputSchema enum value or a string matching
            the enum name (e.g., "PADDLE", "GENERIC").
            By default None, which means the schema will be auto-detected from
            the engine's output names, falling back to GENERIC for any other engine.
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
        cuda_graph : bool, optional
            Unused, CUDA graphs are always disabled since the input shape
            changes on almost every call.
        use_dilation : bool, optional
            Whether or not to dilate the binarized map before extracting boxes.
            By default None, which means no dilation.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        Raises
        ------
        ValueError
            If preprocessor is not 'cpu', the output schema string is invalid,
            or the engine outputs do not match the requested schema.

        """
        self._limit_side_len = limit_side_len
        self._limit_type = limit_type
        self._thresh = thresh
        self._conf_thres = conf_thres
        self._max_candidates = max_candidates
        self._unclip_ratio = unclip_ratio
        self._use_dilation = bool(use_dilation)
        # parent creates the engine and calls _configure_model(), which sets the schema
        self._output_schema_override = output_schema

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
            cuda_graph=False,
            no_warn=no_warn,
            verbose=verbose,
        )

        # prepend with 'tdet_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "tdet_postprocess": f"text_detector::postprocess [{self._tag}]",
                "tdet_get_text_boxes": f"text_detector::get_text_boxes [{self._tag}]",
            }
        )

    @property
    def output_schema(self: Self) -> TextDetectorOutputSchema:
        """Get the output schema used by this text detector."""
        return self._output_schema

    def _configure_model(self: Self) -> None:
        """Auto-detect or apply the output schema from the loaded engine."""
        self._output_schema = resolve_output_schema(
            self._engine, TextDetectorOutputSchema, self._output_schema_override
        )
        self._set_output_order(self._output_schema.value)

    def _setup_cpu_preproc(self: Self) -> CPUPreprocessor:
        width, height = self._input_size
        fn = partial(
            preprocess_text_detection,
            limit_side_len=self._limit_side_len,
            limit_type=self._limit_type,
            input_range=self._input_range,
            mean=self._mean,
            std=self._std,
            max_shape=(height, width),
        )
        return CPUPreprocessor(
            self._input_size, self._input_range, self._dtype, resize=fn, tag=self._tag
        )

    def _resolve_method(self: Self, method: str) -> str:
        if method != "cpu":
            err_msg = "TextDetector only supports cpu preprocessing"
            raise ValueError(err_msg)
        return method

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
            The minimum mean probability inside a box to keep it.
            If not passed, will use value from constructor.
        no_copy : bool, optional
            Unused, the boxes are always new arrays.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs per image, each containing
            [boxes, scores].

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["tdet_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        conf_thres = conf_thres if conf_thres is not None else self._conf_thres
        t0 = time.perf_counter()
        data = postprocess_text_detection(
            self._order_outputs(outputs),
            ratios,
            padding,
            self._thresh,
            conf_thres,
            self._max_candidates,
            self._unclip_ratio,
            use_dilation=self._use_dilation,
            no_copy=no_copy,
            verbose=verbose,
        )
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

        return data

    def _post_fn(self: Self, conf_thres: float | None, *, verbose: bool | None) -> _PostFn:
        """Bridge ``_run_core`` / ``_end2end_core`` to :meth:`postprocess`."""
        return lambda o, r, p, nc: self.postprocess(o, r, p, conf_thres, no_copy=nc, verbose=verbose)

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray | Buffer,
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
        images: Sequence[np.ndarray | Buffer],
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
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
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
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to run the model on.
        ratios : tuple[float, float] | list[tuple[float, float]], optional
            The ratios generated during preprocessing. For single image, pass tuple.
            For batch, pass list.
        padding : tuple[float, float] | list[tuple[float, float]], optional
            The padding values used during preprocessing. For single image, pass tuple.
            For batch, pass list.
        conf_thres : float, optional
            The minimum mean probability inside a box to keep it during
            postprocessing. By default None, which will use value passed
            during initialization.
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
        return self._run_core(
            images,
            ratios,
            padding,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
            post=self._post_fn(conf_thres, verbose=verbose),
        )

    # run overloads - batch input (3 overloads)
    @overload
    def run(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
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
        images: Sequence[np.ndarray | Buffer],
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
        images: Sequence[np.ndarray | Buffer],
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
        images: np.ndarray | Buffer,
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
        images: np.ndarray | Buffer,
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
        images: np.ndarray | Buffer,
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
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
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
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to run the model on.
        ratios : tuple[float, float] | list[tuple[float, float]], optional
            The ratios generated during preprocessing. For single image, pass tuple.
            For batch, pass list.
        padding : tuple[float, float] | list[tuple[float, float]], optional
            The padding values used during preprocessing. For single image, pass tuple.
            For batch, pass list.
        conf_thres : float, optional
            The minimum mean probability inside a box to keep it during
            postprocessing. By default None, which will use value passed
            during initialization.
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
        RuntimeError
            If postprocessing is running, but ratios/padding not found
        ValueError
            If preprocessed inputs are not a single batch tensor.

        """
        return self._run_core(
            images,
            ratios,
            padding,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
            post=self._post_fn(conf_thres, verbose=verbose),
        )

    # get_text_boxes overloads
    @overload
    def get_text_boxes(
        self: Self,
        outputs: list[np.ndarray],
        *,
        verbose: bool | None = ...,
    ) -> list[TextBox]: ...

    @overload
    def get_text_boxes(
        self: Self,
        outputs: list[list[np.ndarray]],
        *,
        verbose: bool | None = ...,
    ) -> list[list[TextBox]]: ...

    def get_text_boxes(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        *,
        verbose: bool | None = None,
    ) -> list[TextBox] | list[list[TextBox]]:
        """
        Get the text boxes from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[TextBox] | list[list[TextBox]]
            For single image: list[TextBox] (text boxes for single image).
            For batch: list[list[TextBox]] (text boxes per image).
            Each TextBox is (box, score) in reading order, where box is the (4, 2)
            float32 corner points (TL, TR, BR, BL) in original image coordinates.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["tdet_get_text_boxes"])

        if verbose:
            LOG.debug(f"{self._tag}: get_text_boxes")

        batch, single = self._as_batch(outputs)
        result = get_text_boxes(batch, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_text_boxes

        return result[0] if single else result

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray | Buffer,
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[TextBox]: ...

    @overload
    def end2end(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[TextBox]]: ...

    def end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[TextBox] | list[list[TextBox]]:
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_text_boxes in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to perform inference with.
        conf_thres : float, optional
            The minimum mean probability inside a box to keep it.
            By default None, which will use value passed during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[TextBox] | list[list[TextBox]]
            For single image: list[TextBox] (text boxes).
            For batch: list[list[TextBox]] (text boxes per image).
            Each TextBox is (box, score), see :meth:`get_text_boxes`.

        """
        return self._end2end_core(
            images,
            verbose=verbose,
            post=self._post_fn(conf_thres, verbose=verbose),
            get=lambda pp: self.get_text_boxes(pp, verbose=verbose),
        )


class TextRecognizer(ImageModel, TextRecognizerInterface):
    """Implementation of CTC text line recognizers, as used by PaddleOCR."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "cpu",
        resize_method: str = "linear",
        character_dict: Path | str | Sequence[str] | None = None,
        mean: tuple[float, float, float] | None = _REC_MEAN,
        std: tuple[float, float, float] | None = _REC_STD,
        output_schema: TextRecognizerOutputSchema | str | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,  # noqa: ARG002
        use_space_char: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a TextRecognizer object.

        The engine should be built with a (min, opt, max) optimization profile over
        the batch size and input width, see :meth:`trtutils.models.PPOCRRecognizer.build`.
        Text lines are resized to the engine's input height keeping their aspect
        ratio and right-padded to the widest line of the batch.

        Parameters
        ----------
        engine_path : Path, str
            The Path or str to the compiled TensorRT engine.
        warmup_iterations : int
            The number of warmup iterations to perform.
            The default is 10.
        input_range : tuple[float, float]
            The range of input values which should be passed to
            the model, used when mean and std are None. By default [0.0, 1.0].
        preprocessor : str
            The type of preprocessor to use. Only 'cpu' is supported,
            any other value raises a ValueError. Default is 'cpu'.
        resize_method : str
            Unused, lines are resized to the input height keeping their aspect ratio.
            Kept for parity with the other image models, default is 'linear'.
        character_dict : Path, str, Sequence[str]
            The model's character dictionary: a text file with one character
            per line, or the list of characters itself (excluding the CTC blank).
            Required, the default of None raises.
        mean : tuple[float, float, float] | None, optional
            The mean values to use for the normalization.
            By default (0.5, 0.5, 0.5).
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for the normalization.
            By default (0.5, 0.5, 0.5).
        output_schema : TextRecognizerOutputSchema, str, optional
            Manually specify the output schema instead of auto-detection.
            Can be a TextRecognizerOutputSchema enum value or a string matching
            the enum name (e.g., "PADDLE", "GENERIC").
            By default None, which means the schema will be auto-detected from
            the engine's output names, falling back to GENERIC for any other engine.
        dla_core : int, optional
            The DLA core to assign DLA layers of the engine to. Default is None.
            If None, any DLA layers will be assigned to DLA core 0.
        device : int, optional
            The CUDA device index to use for this recognizer. Default is None,
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
            Unused, CUDA graphs are always disabled since the input shape
            changes on almost every call.
        use_space_char : bool, optional
            Whether or not the model has an extra trailing class for the space
            character. By default None, which means True.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        Raises
        ------
        ValueError
            If no character dictionary is given, it does not match the
            engine's number of classes, preprocessor is not 'cpu', the output
            schema string is invalid, or the engine outputs do not match the
            requested schema.

        """
        if character_dict is None:
            err_msg = "TextRecognizer requires a character_dict"
            raise ValueError(err_msg)
        if isinstance(character_dict, (str, Path)):
            with Path(character_dict).open(encoding="utf-8") as f:
                chars = [line.rstrip("\r\n") for line in f]
            # a trailing newline is not a character
            if chars and not chars[-1]:
                chars.pop()
        else:
            chars = list(character_dict)
        use_space_char = use_space_char if use_space_char is not None else True
        self._characters: list[str] = ["blank", *chars] + ([" "] if use_space_char else [])
        # parent creates the engine and calls _configure_model(), which sets the schema
        self._output_schema_override = output_schema

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
            cuda_graph=False,
            no_warn=no_warn,
            verbose=verbose,
        )

        num_classes = self._engine.output_spec[0][0][-1]
        if num_classes > 0 and num_classes != len(self._characters):
            err_msg = (
                f"Character dictionary has {len(self._characters)} classes (with blank"
                f"{' and space' if use_space_char else ''}), engine outputs {num_classes}"
            )
            raise ValueError(err_msg)

        # prepend with 'trec_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "trec_postprocess": f"text_recognizer::postprocess [{self._tag}]",
                "trec_get_texts": f"text_recognizer::get_texts [{self._tag}]",
            }
        )

    @property
    def characters(self: Self) -> list[str]:
        """Get the class labels, index 0 is the CTC blank."""
        return self._characters

    @property
    def output_schema(self: Self) -> TextRecognizerOutputSchema:
        """Get the output schema used by this text recognizer."""
        return self._output_schema

    def _configure_model(self: Self) -> None:
        """Auto-detect or apply the output schema from the loaded engine."""
        self._output_schema = resolve_output_schema(
            self._engine, TextRecognizerOutputSchema, self._output_schema_override
        )
        self._set_output_order(self._output_schema.value)

    def _setup_cpu_preproc(self: Self) -> CPUPreprocessor:
        width, height = self._input_size
        fn = partial(
            preprocess_text_recognition,
            image_height=height,
            max_width=width,
            input_range=self._input_range,
            mean=self._mean,
            std=self._std,
        )
        return CPUPreprocessor(
            self._input_size, self._input_range, self._dtype, resize=fn, tag=self._tag
        )

    def _resolve_method(self: Self, method: str) -> str:
        if method != "cpu":
            err_msg = "TextRecognizer only supports cpu preprocessing"
            raise ValueError(err_msg)
        return method

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
            Unused, the results are always new arrays.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs per image, each containing
            [indices, probs] of the best class per timestep.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["trec_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        t0 = time.perf_counter()
        data = postprocess_text_recognition(
            self._order_outputs(outputs), no_copy=no_copy, verbose=verbose
        )
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

        return data

    def _post_fn(self: Self, *, verbose: bool | None) -> _PostFn:
        """Bridge ``_run_core`` / ``_end2end_core`` to :meth:`postprocess`."""
        return lambda o, _r, _p, nc: self.postprocess(o, no_copy=nc, verbose=verbose)

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray | Buffer,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def __call__(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    def __call__(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
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
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single text line image (HWC format) or list of them to run the
            model on, at most the engine's max batch size.
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
        images: Sequence[np.ndarray | Buffer],
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[np.ndarray]]: ...

    @overload
    def run(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
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
        images: np.ndarray | Buffer,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[False],
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: np.ndarray | Buffer,
        *,
        preprocessed: bool | None = ...,
        postprocess: Literal[True] | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    @overload
    def run(
        self: Self,
        images: np.ndarray | Buffer,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    def run(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
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
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single text line image (HWC format) or list of them to run the
            model on, at most the engine's max batch size.
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
            post=self._post_fn(verbose=verbose),
            needs_ratios=False,
        )

    def get_texts(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        *,
        verbose: bool | None = None,
    ) -> list[Text]:
        """
        Get the texts from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[Text]
            One Text (text, confidence) per image, a list of one for a single image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["trec_get_texts"])

        if verbose:
            LOG.debug(f"{self._tag}: get_texts")

        batch, _ = self._as_batch(outputs)
        result = get_texts(batch, self._characters, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_texts

        return result

    def end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        *,
        verbose: bool | None = None,
    ) -> list[Text]:
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_texts in that order. Any number of lines is accepted: they are
        grouped by aspect ratio into batches of the engine's max batch size,
        so similar widths share padding.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single text line image (HWC format) or list of them to perform
            inference with.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[Text]
            One Text (text, confidence) per image in input order, a list of one
            for a single image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["end2end"])

        if verbose:
            LOG.debug(f"{self._tag}: end2end")

        lines: Sequence[np.ndarray | Buffer] = (
            [images] if isinstance(images, (np.ndarray, Buffer)) else images
        )
        order = np.argsort([img.shape[1] / img.shape[0] for img in lines])
        results: list[Text] = [("", 0.0)] * len(lines)
        for start in range(0, len(order), self._batch_size):
            batch_idx = order[start : start + self._batch_size]
            # outputs are consumed by get_texts() inside this call, so skip their copies
            outputs, _, _ = self.infer([lines[i] for i in batch_idx], no_copy=True, verbose=verbose)
            batch = self.postprocess(outputs, no_copy=True, verbose=verbose)
            for i, text in zip(batch_idx, self.get_texts(batch, verbose=verbose)):
                results[i] = text

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # end2end

        return results


class OCR:
    """
    Two-stage OCR pipeline: text detection followed by text line recognition.

    Detected regions are cropped (perspective-corrected), recognized in
    batches, and low-confidence lines are dropped. Before detection, very
    large, small or thin images are resized and padded the way RapidOCR does.
    """

    def __init__(
        self: Self,
        detector: TextDetectorInterface,
        recognizer: TextRecognizerInterface,
        conf_thres: float = 0.5,
        min_height: int = 30,
        width_height_ratio: float = 8,
        max_side_len: int = 2000,
        min_side_len: int = 30,
        *,
        verbose: bool | None = None,
    ) -> None:
        """
        Create an OCR object.

        Parameters
        ----------
        detector : TextDetectorInterface
            The text detector.
        recognizer : TextRecognizerInterface
            The text recognizer.
        conf_thres : float
            The minimum recognition confidence for a line to be returned.
            By default 0.5.
        min_height : int
            Images at most this tall are padded vertically before detection,
            which helps the detector on single text lines. By default 30.
        width_height_ratio : float
            Images wider than this aspect ratio are padded vertically before
            detection. Set to -1 to disable. By default 8.
        max_side_len : int
            Images with a longer side above this are downscaled first.
            By default 2000.
        min_side_len : int
            Images with a shorter side below this are upscaled first.
            By default 30.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        """
        self._nvtx_tags = {
            "end2end": "ocr::end2end",
        }
        self._detector = detector
        self._recognizer = recognizer
        self._conf_thres = conf_thres
        self._min_height = min_height
        self._width_height_ratio = width_height_ratio
        self._max_side_len = max_side_len
        self._min_side_len = min_side_len
        if verbose:
            LOG.debug(f"Creating OCR: {detector.name} + {recognizer.name}")

    @property
    def detector(self: Self) -> TextDetectorInterface:
        """Get the text detector."""
        return self._detector

    @property
    def recognizer(self: Self) -> TextRecognizerInterface:
        """Get the text recognizer."""
        return self._recognizer

    def _ocr_image(
        self: Self,
        image: np.ndarray | Buffer,
        conf_thres: float,
        box_conf_thres: float | None,
        *,
        verbose: bool | None,
    ) -> list[TextLine]:
        img = image.numpy() if isinstance(image, Buffer) else image
        raw_h, raw_w = img.shape[:2]
        if max(raw_h, raw_w) > self._max_side_len:
            img = resize_text_detection(img, self._max_side_len, "max")
        if min(img.shape[:2]) < self._min_side_len:
            img = resize_text_detection(img, self._min_side_len, "min")

        h, w = img.shape[:2]
        scale = np.array([raw_w / w, raw_h / h], np.float32)
        pad = 0
        if h <= self._min_height or (
            self._width_height_ratio != -1 and w / h > self._width_height_ratio
        ):
            new_h = max(int(w / self._width_height_ratio), self._min_height) * 2
            pad = int(abs(new_h - h) / 2)
            img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        text_boxes = self._detector.end2end(img, box_conf_thres, verbose=verbose)
        if not text_boxes:
            return []
        boxes = np.stack([box for box, _ in text_boxes])
        lines = self._recognizer.end2end(
            [crop_text_region(img, box) for box in boxes], verbose=verbose
        )

        out_boxes = (boxes - np.array([0, pad], np.float32)) * scale
        out_boxes = np.clip(out_boxes, 0, np.array([raw_w, raw_h], np.float32))
        return [
            (box, text, score) for box, (text, score) in zip(out_boxes, lines) if score >= conf_thres
        ]

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray | Buffer,
        conf_thres: float | None = ...,
        *,
        box_conf_thres: float | None = ...,
        verbose: bool | None = ...,
    ) -> list[TextLine]: ...

    @overload
    def __call__(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        conf_thres: float | None = ...,
        *,
        box_conf_thres: float | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[TextLine]]: ...

    def __call__(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        conf_thres: float | None = None,
        *,
        box_conf_thres: float | None = None,
        verbose: bool | None = None,
    ) -> list[TextLine] | list[list[TextLine]]:
        """
        Detect and recognize all text in images, see :meth:`end2end`.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to perform inference with.
        conf_thres : float, optional
            The minimum recognition confidence for a line to be returned.
            By default None, which will use value passed during initialization.
        box_conf_thres : float, optional
            The detector's minimum text box score for this call.
            By default None, which will use the detector's own conf_thres.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[TextLine] | list[list[TextLine]]
            For single image: list[TextLine] (text lines).
            For batch: list[list[TextLine]] (text lines per image).
            Each TextLine is (box, text, score), see :meth:`end2end`.

        """
        return self.end2end(images, conf_thres, box_conf_thres=box_conf_thres, verbose=verbose)

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray | Buffer,
        conf_thres: float | None = ...,
        *,
        box_conf_thres: float | None = ...,
        verbose: bool | None = ...,
    ) -> list[TextLine]: ...

    @overload
    def end2end(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        conf_thres: float | None = ...,
        *,
        box_conf_thres: float | None = ...,
        verbose: bool | None = ...,
    ) -> list[list[TextLine]]: ...

    def end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        conf_thres: float | None = None,
        *,
        box_conf_thres: float | None = None,
        verbose: bool | None = None,
    ) -> list[TextLine] | list[list[TextLine]]:
        """
        Detect and recognize all text in images.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to perform inference with.
        conf_thres : float, optional
            The minimum recognition confidence for a line to be returned.
            By default None, which will use value passed during initialization.
        box_conf_thres : float, optional
            The detector's minimum text box score for this call.
            By default None, which will use the detector's own conf_thres.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[TextLine] | list[list[TextLine]]
            For single image: list[TextLine] (text lines).
            For batch: list[list[TextLine]] (text lines per image).
            Each TextLine is (box, text, score) in reading order, where box is the
            (4, 2) float32 corner points (TL, TR, BR, BL) in image coordinates.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["end2end"])

        if verbose:
            LOG.debug("OCR: end2end")

        conf_thres = conf_thres if conf_thres is not None else self._conf_thres
        is_single = isinstance(images, (np.ndarray, Buffer))
        batch_images: Sequence[np.ndarray | Buffer] = [images] if is_single else images
        results = [
            self._ocr_image(img, conf_thres, box_conf_thres, verbose=verbose) for img in batch_images
        ]

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # end2end

        return results[0] if is_single else results
