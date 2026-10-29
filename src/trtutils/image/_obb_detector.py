# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast, overload

import nvtx

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._buffer import Buffer

from ._image_model import ImageModel
from ._schema import OBBDetectorOutputSchema
from .interfaces import OBBDetectorInterface
from .postprocessors import get_obb_detections, postprocess_yolo_obb
from .postprocessors._gpu import CudaObbPostprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from typing_extensions import Literal, Self

    from trtutils.image.types import OBBDetection

    from ._image_model import _PostFn


class OBBDetector(ImageModel, OBBDetectorInterface):
    """Implementation of oriented bounding box detectors (YOLO-style OBB heads)."""

    _OUTPUT_SCHEMA = OBBDetectorOutputSchema
    _cuda_postprocess = True

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "cuda",
        resize_method: str = "letterbox",
        conf_thres: float = 0.1,
        nms_iou_thres: float = 0.5,
        max_det: int = 300,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        output_schema: OBBDetectorOutputSchema | str | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        postprocessor: str | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a OBBDetector object.

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
            The options are ['cpu', 'cuda', 'trt'], default is 'cuda'.
        resize_method : str
            The type of resize algorithm to use.
            The options are ['letterbox', 'linear'], default is 'letterbox'.
        conf_thres : float, optional
            The confidence threshold above which to keep results.
            By default 0.1
        nms_iou_thres : float, optional
            The IOU threshold to use in the NMS operation. By default, 0.5
        max_det : int, optional
            The maximum number of results kept per image after NMS. By default, 300
        mean : tuple[float, float, float] | None, optional
            The mean values to use for normalization.
            By default, None, which means no normalization will be applied.
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for normalization.
            By default, None, which means no normalization will be applied.
        output_schema : OBBDetectorOutputSchema, str, optional
            Manually specify the output schema instead of auto-detection.
            Can be a OBBDetectorOutputSchema value or a string matching its name (e.g., "YOLO").
            By default None, which means the schema will be auto-detected from
            the engine's output names.
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
        postprocessor : str, optional
            Where to postprocess, 'cpu' or 'cuda'. 'cuda' decodes the engine outputs
            on the GPU and copies back only the final results. Default is None,
            which picks 'cuda' when the task has a CUDA postprocessor, else 'cpu'.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        Raises
        ------
        ValueError
            If the output schema string is invalid, or the engine outputs do
            not match the output schema.

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
            postprocessor=postprocessor,
            no_warn=no_warn,
            verbose=verbose,
            output_schema=output_schema,
        )

        # prepend with 'obb_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "obb_init": f"obbdetector::init [{self._tag}]",
                "obb_postprocess": f"obbdetector::postprocess [{self._tag}]",
                "obb_get": f"obbdetector::get_obbs [{self._tag}]",
            }
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["obb_init"])

        self._conf_thres: float = conf_thres
        self._nms_iou: float = nms_iou_thres
        self._max_det: int = max_det
        self._cuda_post = CudaObbPostprocessor() if self._postprocessor == "cuda" else None

        if self._verbose:
            LOG.debug(f"{self._tag}: Output schema: {self._output_schema}")

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # init

    @property
    def output_schema(self: Self) -> OBBDetectorOutputSchema:
        """Get the output schema used by this model."""
        return self._output_schema

    def postprocess(
        self: Self,
        outputs: list[np.ndarray] | list[Buffer],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Postprocess the raw engine outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[Buffer]
            The raw outputs from the engine to postprocess.
        ratios : list[tuple[float, float]]
            The rescale ratios used during preprocessing for each image.
        padding : list[tuple[float, float]]
            The padding values used during preprocessing for each image.
        conf_thres : float, optional
            The confidence threshold to filter by.
            If not passed, will use value from constructor.
        nms_iou_thres : float, optional
            The IOU threshold to use in the NMS operation.
            If not passed, will use value from constructor.
        no_copy : bool, optional
            Unused, results are always new arrays.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs per image, each containing
            [bboxes, scores, class_ids, rboxes].

        """
        del no_copy  # results are always new arrays
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["obb_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        conf_thres = conf_thres if conf_thres is not None else self._conf_thres
        nms_iou_thres = nms_iou_thres if nms_iou_thres is not None else self._nms_iou
        outputs = self._order_outputs(outputs)
        t0 = time.perf_counter()
        if self._cuda_post is not None and isinstance(outputs[0], Buffer):
            data = self._cuda_post(
                cast("list[Buffer]", outputs),
                ratios,
                padding,
                conf_thres,
                nms_iou_thres,
                max_det=self._max_det,
                stream=self._engine.stream,
            )
        else:
            data = postprocess_yolo_obb(
                cast("list[np.ndarray]", outputs),
                ratios,
                padding,
                conf_thres,
                nms_iou_thres,
                max_det=self._max_det,
                verbose=verbose,
            )
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

        return data

    def _post_fn(
        self: Self,
        conf_thres: float | None,
        nms_iou_thres: float | None = None,
        *,
        verbose: bool | None,
    ) -> _PostFn:
        """Bridge ``_run_core`` / ``_end2end_core`` to :meth:`postprocess`."""
        return lambda o, r, p, nc: self.postprocess(
            o,
            r,
            p,
            conf_thres,
            nms_iou_thres,
            no_copy=nc,
            verbose=verbose,
        )

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray | Buffer,
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
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
        nms_iou_thres: float | None = ...,
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
        nms_iou_thres: float | None = None,
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
        nms_iou_thres : float, optional
            The IOU threshold to use in the NMS operation during postprocessing.
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
        return self._run_core(
            images,
            ratios,
            padding,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
            post=self._post_fn(conf_thres, nms_iou_thres, verbose=verbose),
        )

    # run overloads - batch input (3 overloads)
    @overload
    def run(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
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
        nms_iou_thres: float | None = ...,
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
        nms_iou_thres: float | None = ...,
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
        nms_iou_thres: float | None = ...,
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
        nms_iou_thres: float | None = ...,
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
        nms_iou_thres: float | None = ...,
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
        nms_iou_thres: float | None = None,
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
        nms_iou_thres : float, optional
            The IOU threshold to use in the NMS operation during postprocessing.
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
            post=self._post_fn(conf_thres, nms_iou_thres, verbose=verbose),
        )

    @overload
    def get_obbs(
        self: Self,
        outputs: list[np.ndarray],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[OBBDetection]: ...
    @overload
    def get_obbs(
        self: Self,
        outputs: list[list[np.ndarray]],
        conf_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[OBBDetection]]: ...
    def get_obbs(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        conf_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[OBBDetection] | list[list[OBBDetection]]:
        """
        Get the oriented boxes from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: the postprocessed outputs of that image.
            For batch: the postprocessed outputs per image.
        conf_thres : float, optional
            The confidence threshold to filter by.
            By default None, which will use value passed during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[OBBDetection] | list[list[OBBDetection]]
            The oriented boxes of the image, or per image for a batch.
            Each is ((cx, cy, w, h, angle), score, class_id).

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["obb_get"])

        if verbose:
            LOG.debug(f"{self._tag}: get_obbs")

        conf_thres = conf_thres if conf_thres is not None else self._conf_thres

        batch, single = self._as_batch(outputs)
        result = get_obb_detections(batch, conf_thres, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_obbs

        return result[0] if single else result

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray | Buffer,
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[OBBDetection]: ...
    @overload
    def end2end(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        conf_thres: float | None = ...,
        nms_iou_thres: float | None = ...,
        *,
        verbose: bool | None = ...,
    ) -> list[list[OBBDetection]]: ...
    def end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        conf_thres: float | None = None,
        nms_iou_thres: float | None = None,
        *,
        verbose: bool | None = None,
    ) -> list[OBBDetection] | list[list[OBBDetection]]:
        """
        Perform end to end inference for an image or a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_obbs in that order.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to perform inference with.
        conf_thres : float, optional
            The confidence threshold to filter by.
            By default None, which will use value passed during initialization.
        nms_iou_thres : float, optional
            The IOU threshold to use in the NMS operation.
            By default None, which will use value passed during initialization.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[OBBDetection] | list[list[OBBDetection]]
            The oriented boxes of the image, or per image for a batch.

        """
        return self._end2end_core(
            images,
            verbose=verbose,
            post=self._post_fn(conf_thres, nms_iou_thres, verbose=verbose),
            get=lambda pp: self.get_obbs(pp, conf_thres, verbose=verbose),
        )
