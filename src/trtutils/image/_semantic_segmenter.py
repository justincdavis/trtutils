# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import time
from typing import TYPE_CHECKING, overload

import numpy as np
import nvtx
from typing_extensions import Literal

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._buffer import Buffer

from ._image_model import ImageModel
from ._schema import SemanticSegmenterOutputSchema
from .interfaces import SemanticSegmenterInterface
from .postprocessors import get_class_maps, postprocess_class_map, postprocess_semantic_logits

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from typing_extensions import Self

    from trtutils.image.types import SemanticMask

    from ._image_model import _PostFn

_POSTPROCESS: dict[SemanticSegmenterOutputSchema, Callable[..., list[list[np.ndarray]]]] = {
    SemanticSegmenterOutputSchema.CLASS_MAP: postprocess_class_map,
    SemanticSegmenterOutputSchema.LOGITS: postprocess_semantic_logits,
    SemanticSegmenterOutputSchema.GENERIC: postprocess_semantic_logits,
}


class SemanticSegmenter(ImageModel, SemanticSegmenterInterface):
    """
    Implementation of semantic segmentation models.

    Returns one class map per image: an (H, W) uint8 array of class ids at the
    original image resolution. Class ids are uint8, so models are limited to 256
    classes. For per-object masks see :class:`~trtutils.image.Segmenter`.
    """

    _OUTPUT_SCHEMA = SemanticSegmenterOutputSchema

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "cuda",
        resize_method: str = "letterbox",
        mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
        output_schema: SemanticSegmenterOutputSchema | str | None = None,
        dla_core: int | None = None,
        device: int | None = None,
        backend: str = "auto",
        *,
        warmup: bool | None = None,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        cuda_graph: bool | None = None,
        postprocessor: str | None = None,
        cpu_resize: bool | None = None,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Create a SemanticSegmenter object.

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
            The options are ['cpu', 'cuda', 'trt', 'dla0', 'dla1'], default is 'cuda'.
        resize_method : str
            The type of resize algorithm to use.
            The options are ['letterbox', 'linear', 'center_crop'], default is 'letterbox',
            which keeps the aspect ratio. The class maps are mapped back to the full
            original image; with 'center_crop' the cropped-away border takes the
            nearest predicted class.
        mean : tuple[float, float, float] | None, optional
            The mean values to use for the imagenet normalization.
            By default, (0.485, 0.456, 0.406) (ImageNet mean).
            Pass None to disable normalization.
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for the imagenet normalization.
            By default, (0.229, 0.224, 0.225) (ImageNet std).
            Pass None to disable normalization.
        output_schema : SemanticSegmenterOutputSchema, str, optional
            Manually specify the output schema instead of auto-detection.
            Can be a SemanticSegmenterOutputSchema enum value or a string matching
            the enum name (e.g., "CLASS_MAP", "LOGITS", "GENERIC").
            By default None, which means the schema will be auto-detected from
            the engine's output names, falling back to GENERIC (logits) for any
            other engine.
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
        postprocessor : str, optional
            Where to postprocess. Only 'cpu' is available. Default is None,
            which is 'cpu'.
        cpu_resize : bool, optional
            Whether to resize images on the CPU and leave the rest of the
            preprocessing to the preprocessor. Default is None, which resizes on
            the CPU only what the preprocessor cannot resize, see :class:`ImageModel`.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        Raises
        ------
        ValueError
            If the output schema string is invalid, or the engine outputs
            do not match the requested schema.

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
            cpu_resize=cpu_resize,
            no_warn=no_warn,
            verbose=verbose,
            output_schema=output_schema,
        )

        # prepend with 'sem_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "sem_postprocess": f"semantic_segmenter::postprocess [{self._tag}]",
                "sem_get_class_maps": f"semantic_segmenter::get_class_maps [{self._tag}]",
            }
        )

    @property
    def output_schema(self: Self) -> SemanticSegmenterOutputSchema:
        """Get the output schema used by this model."""
        return self._output_schema

    def postprocess(
        self: Self,
        outputs: list[np.ndarray] | list[Buffer],
        ratios: list[tuple[float, float]],
        padding: list[tuple[float, float]],
        *,
        image_shapes: list[tuple[int, int]] | None = None,
        no_copy: bool | None = None,  # noqa: ARG002
        verbose: bool | None = None,
    ) -> list[list[np.ndarray]]:
        """
        Postprocess the raw engine outputs into class maps at the original resolution.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[Buffer]
            The raw outputs from the engine to postprocess.
        ratios : list[tuple[float, float]]
            The rescale ratios used during preprocessing for each image.
        padding : list[tuple[float, float]]
            The padding values used during preprocessing for each image.
        image_shapes : list[tuple[int, int]], optional
            The (height, width) of each original image. By default None, which
            recovers the sizes from the ratios and padding; letterbox rounding can
            make those a pixel or two off when the image was downscaled.
            :meth:`run` and :meth:`end2end` pass the image shapes.
        no_copy : bool, optional
            Kept for interface symmetry with the other task models. The class
            maps are always new arrays, so this has no effect.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            Per image, a list holding its (H, W) uint8 class map.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["sem_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        t0 = time.perf_counter()
        data = _POSTPROCESS[self._output_schema](
            self._host_outputs(outputs),
            ratios,
            padding,
            self._input_size,
            image_shapes=image_shapes,
            verbose=verbose,
        )
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

        return data

    def _post_fn(
        self: Self,
        image_shapes: list[tuple[int, int]] | None,
        *,
        verbose: bool | None,
    ) -> _PostFn:
        """Bridge ``_run_core`` / ``_end2end_core`` to :meth:`postprocess`."""
        return lambda o, r, p, nc: self.postprocess(
            o, r, p, image_shapes=image_shapes, no_copy=nc, verbose=verbose
        )

    # run overloads - batch input (3 overloads)
    @overload
    def run(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        ratios: list[tuple[float, float]] | None = ...,
        padding: list[tuple[float, float]] | None = ...,
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
            For batch, pass list of tuples.
        padding : tuple[float, float] | list[tuple[float, float]], optional
            The padding values used during preprocessing. For single image, pass tuple.
            For batch, pass list of tuples.
        preprocessed : bool, optional
            Whether or not the inputs have been preprocessed.
            If None, will preprocess inputs. Preprocessed inputs carry no original
            image size, so their class maps are sized from the ratios and padding.
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
            For single image with postprocess=True: list[np.ndarray] (single image outputs).
            For batch with postprocess=True: list[list[np.ndarray]] (per-image outputs).
            For postprocess=False: list[np.ndarray] (raw outputs).

        Raises
        ------
        ValueError
            If preprocessed inputs are not a single batch tensor.
        RuntimeError
            If postprocessing preprocessed inputs without ratios and padding.

        """
        return self._run_core(
            images,
            ratios,
            padding,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
            post=self._post_fn(_image_shapes(images, preprocessed=preprocessed), verbose=verbose),
        )

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray | Buffer,
        ratios: tuple[float, float] | None = ...,
        padding: tuple[float, float] | None = ...,
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
        return self._run_core(
            images,
            ratios,
            padding,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
            post=self._post_fn(_image_shapes(images, preprocessed=preprocessed), verbose=verbose),
        )

    # get_class_maps overloads
    @overload
    def get_class_maps(
        self: Self,
        outputs: list[np.ndarray],
        *,
        verbose: bool | None = ...,
    ) -> SemanticMask: ...

    @overload
    def get_class_maps(
        self: Self,
        outputs: list[list[np.ndarray]],
        *,
        verbose: bool | None = ...,
    ) -> list[SemanticMask]: ...

    def get_class_maps(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        *,
        verbose: bool | None = None,
    ) -> SemanticMask | list[SemanticMask]:
        """
        Get the class maps from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        SemanticMask | list[SemanticMask]
            For single image: the (H, W) uint8 class map.
            For batch: the class map per image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["sem_get_class_maps"])

        if verbose:
            LOG.debug(f"{self._tag}: get_class_maps")

        batch, single = self._as_batch(outputs)
        result = get_class_maps(batch, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_class_maps

        return result[0] if single else result

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray | Buffer,
        *,
        verbose: bool | None = ...,
    ) -> SemanticMask: ...

    @overload
    def end2end(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        *,
        verbose: bool | None = ...,
    ) -> list[SemanticMask]: ...

    def end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        *,
        verbose: bool | None = None,
    ) -> SemanticMask | list[SemanticMask]:
        """
        Perform end to end inference for an image or a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_class_maps in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to perform inference with.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        SemanticMask | list[SemanticMask]
            For single image: the (H, W) uint8 class map at the image's resolution.
            For batch: the class map per image.

        """
        return self._end2end_core(
            images,
            verbose=verbose,
            post=self._post_fn(_image_shapes(images, preprocessed=False), verbose=verbose),
            get=lambda pp: get_class_maps(pp, verbose=verbose),
        )


def _image_shapes(
    images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
    *,
    preprocessed: bool | None,
) -> list[tuple[int, int]] | None:
    """Return the (height, width) of each image, None for a preprocessed batch tensor."""
    if preprocessed:
        return None
    batch = [images] if isinstance(images, (np.ndarray, Buffer)) else images
    return [(image.shape[0], image.shape[1]) for image in batch]
