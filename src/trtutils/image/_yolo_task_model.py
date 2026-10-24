# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import nvtx

from trtutils._flags import FLAGS
from trtutils._log import LOG

from ._image_model import ImageModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self

    from trtutils.core._buffer import Buffer

    from ._image_model import _PostFn


class _YoloTaskModel(ImageModel):
    """
    Base for YOLO task heads (segmentation, pose, OBB).

    Subclasses set the expected engine output names and the postprocess and get
    functions. Postprocessed per-image outputs are [bboxes, scores, class_ids, extra].
    """

    _output_names: ClassVar[tuple[str, ...]]
    _postprocess_fn: ClassVar[Callable[..., list[list[np.ndarray]]]]
    _get_fn: ClassVar[Callable[..., list[list[Any]]]]
    _needs_input_size: ClassVar[bool] = False

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "trt",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
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
        cuda_graph: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a task model object.

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
        conf_thres : float, optional
            The confidence threshold above which to keep results.
            By default 0.1
        nms_iou_thres : float, optional
            The IOU threshold to use in the NMS operation. By default, 0.5
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
            The CUDA device index to use. Default is None,
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
            If the engine outputs do not match the expected output names.

        """
        self._conf_thres: float = conf_thres
        self._nms_iou: float = nms_iou_thres
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
        self._nvtx_tags["task_postprocess"] = f"{type(self).__name__}::postprocess [{self._tag}]"

    def _configure_model(self: Self) -> None:
        """Validate the engine outputs and map them by name."""
        names = list(self._engine.output_names)
        expected = list(self._output_names)
        if sorted(names) != sorted(expected):
            err_msg = f"Expected engine outputs {expected}, found: {names}"
            raise ValueError(err_msg)
        self._set_output_order(expected)

    def postprocess(
        self: Self,
        outputs: list[np.ndarray],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None = None,
        *,
        no_copy: bool | None = None,  # noqa: ARG002
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Postprocess the raw engine outputs.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The raw outputs from the engine to postprocess.
        ratios : list[tuple[float, float]]
            The rescale ratios used during preprocessing for each image.
        padding : list[tuple[float, float]]
            The padding values used during preprocessing for each image.
        conf_thres : float, optional
            The confidence threshold to filter by.
            If not passed, will use value from constructor.
        no_copy : bool, optional
            Unused, results are always new arrays.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs per image, each containing
            [bboxes, scores, class_ids, extra].

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["task_postprocess"])
        if verbose:
            LOG.debug(f"{self._tag}: postprocess")
        outputs = self._order_outputs(outputs)
        extra = {"input_size": self._input_size} if self._needs_input_size else {}
        t0 = time.perf_counter()
        data = self._postprocess_fn(
            outputs,
            ratios,
            padding,
            conf_thres or self._conf_thres,
            self._nms_iou,
            verbose=verbose,
            **extra,
        )
        self._post_profile = (t0, time.perf_counter())
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess
        return data

    def _post_fn(self: Self, conf_thres: float | None, *, verbose: bool | None) -> _PostFn:
        """Bridge ``_run_core`` / ``_end2end_core`` to :meth:`postprocess`."""
        return lambda o, r, p, nc: self.postprocess(o, r, p, conf_thres, no_copy=nc, verbose=verbose)

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
        Run the model on input, see :meth:`run`.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to run the model on.
        ratios : tuple[float, float] | list[tuple[float, float]], optional
            The ratios generated during preprocessing.
        padding : tuple[float, float] | list[tuple[float, float]], optional
            The padding values used during preprocessing.
        conf_thres : float, optional
            The confidence threshold to filter by during postprocessing.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
        postprocess : bool, optional
            Whether or not to postprocess the outputs. By default True.
        no_copy : bool, optional
            If True, the outputs are not copied out of the CUDA host memory
            and WILL BE OVERWRITTEN INPLACE by future inferences.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[np.ndarray] | list[list[np.ndarray]]
            For a single image: its postprocessed outputs. For a batch: those
            per image. With postprocess=False: the raw outputs.

        """
        return self.run(
            images,
            ratios,
            padding,
            conf_thres,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
        )

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
            The confidence threshold to filter by during postprocessing.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If postprocessing will occur on preprocessed inputs, ratios and
            padding must be passed.
        postprocess : bool, optional
            Whether or not to postprocess the outputs. By default True.
        no_copy : bool, optional
            If True, the outputs are not copied out of the CUDA host memory
            and WILL BE OVERWRITTEN INPLACE by future inferences.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[np.ndarray] | list[list[np.ndarray]]
            For a single image: its postprocessed outputs. For a batch: those
            per image. With postprocess=False: the raw outputs.

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

    def _get(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None,
        *,
        verbose: bool | None,
    ) -> list[Any]:
        batch, single = self._as_batch(outputs)
        result = self._get_fn(batch, conf_thres or self._conf_thres, verbose=verbose)
        return result[0] if single else result

    def _end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        conf_thres: float | None,
        *,
        verbose: bool | None,
    ) -> list[Any]:
        return self._end2end_core(
            images,
            verbose=verbose,
            post=self._post_fn(conf_thres, verbose=verbose),
            get=lambda pp: self._get_fn(pp, conf_thres or self._conf_thres, verbose=verbose),
        )
