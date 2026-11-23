# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import time
from functools import partial
from typing import TYPE_CHECKING, overload

import cv2
import numpy as np
import nvtx
from typing_extensions import Literal

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._buffer import Buffer

from ._image_model import ImageModel
from ._schema import ImageRestorerOutputSchema
from .interfaces import ImageRestorerInterface
from .postprocessors import get_restored_images, postprocess_restoration
from .postprocessors._cuda_restoration import CUDARestorePostprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from typing_extensions import Self

    from ._image_model import _PostFn
    from .types import RestoredImage

_IMAGE_RANK = 4
_COLOR_CHANNELS = 3


def pad_to_input(image: np.ndarray, input_shape: tuple[int, int]) -> np.ndarray:
    """
    Pad an image at the bottom and right to the engine input size, without scaling it.

    The border mirrors the image (OpenCV ``BORDER_REFLECT_101``, PyTorch's ``reflect``), as
    Real-ESRGAN pads its inputs, so the restored image keeps no seam at its edges.

    Parameters
    ----------
    image : np.ndarray
        The HWC image, at most ``input_shape`` in each dimension.
    input_shape : tuple[int, int]
        The (width, height) of the engine input.

    Returns
    -------
    np.ndarray
        The padded image, or ``image`` itself when it already has the input size.

    Raises
    ------
    ValueError
        If the image is larger than the input in either dimension.

    """
    width, height = input_shape
    img_height, img_width = image.shape[:2]
    if img_height > height or img_width > width:
        err_msg = (
            f"Image of (height, width) {(img_height, img_width)} is larger than the engine "
            f"input {(height, width)}; image restoration does not rescale images, so build "
            f"the engine at the image size, e.g. Model.build(..., imgsz={(img_height, img_width)})"
        )
        raise ValueError(err_msg)
    if (img_height, img_width) == (height, width):
        return image
    return cv2.copyMakeBorder(
        image, 0, height - img_height, 0, width - img_width, cv2.BORDER_REFLECT_101
    )


class ImageRestorer(ImageModel, ImageRestorerInterface):
    """
    Implementation of image restoration / super-resolution models.

    The engine has a static input size and images are never rescaled: an image smaller
    than the input is padded at the bottom and right, and its restored image is cropped
    back to ``scale`` times its size. An image larger than the input raises a ``ValueError``.
    """

    _OUTPUT_SCHEMA = ImageRestorerOutputSchema
    _CUDA_POSTPROCESSOR = CUDARestorePostprocessor

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        input_range: tuple[float, float] = (0.0, 1.0),
        preprocessor: str = "cuda",
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        output_schema: ImageRestorerOutputSchema | str | None = None,
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
        Create an ImageRestorer object.

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
        mean : tuple[float, float, float] | None, optional
            The mean values to use for normalization.
            By default None, which means no normalization.
        std : tuple[float, float, float] | None, optional
            The standard deviation values to use for normalization.
            By default None, which means no normalization.
        output_schema : ImageRestorerOutputSchema, str, optional
            Manually specify the output schema instead of auto-detection.
            Can be an ImageRestorerOutputSchema enum value or a string matching
            the enum name (e.g., "REAL_ESRGAN", "GENERIC").
            By default None, which means the schema will be auto-detected from
            the engine's output names, falling back to GENERIC for any other engine.
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
            Where to postprocess, 'cpu' or 'cuda'. 'cuda' converts the restored image to
            uint8 on the GPU and copies back only that, a quarter of the float32 output.
            Default is None, which is 'cuda' when the engine outputs float32, else 'cpu'.
        cpu_resize : bool, optional
            Whether to resize images on the CPU and leave the rest of the
            preprocessing to the preprocessor. Images keep their size, so this only
            moves the copy. Default is None, see :class:`ImageModel`.
        no_warn : bool, optional
            If True, suppresses warnings from TensorRT during engine deserialization.
            Default is None, which means warnings will be shown.
        verbose : bool, optional
            Whether or not to log additional information.
            Only covers the initialization phase.

        Raises
        ------
        ValueError
            If the output schema string is invalid, the engine outputs do not match
            the requested schema, or the restored image is not an RGB image an integer
            multiple of the input size.

        """
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            input_range=input_range,
            preprocessor=preprocessor,
            # images are padded to the input size first, so this never scales
            resize_method="pad",
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

        # prepend with 'restore_' to avoid conflicts with ImageModel._nvtx_tags
        self._nvtx_tags.update(
            {
                "restore_postprocess": f"image_restorer::postprocess [{self._tag}]",
                "restore_get_restored_images": (
                    f"image_restorer::get_restored_images [{self._tag}]"
                ),
            }
        )

    def _configure_model(self: Self) -> None:
        """Read the upscale factor from the engine: output size over input size."""
        shape = self._order_outputs(list(self._engine.output_shapes))[0]
        width, height = self._input_size
        out_height, out_width = (shape[2], shape[3]) if len(shape) == _IMAGE_RANK else (0, 0)
        scale = out_height // height
        if (
            len(shape) != _IMAGE_RANK
            or shape[1] != _COLOR_CHANNELS
            or scale < 1
            or (out_height, out_width) != (scale * height, scale * width)
        ):
            err_msg = (
                f"{self._tag}: expected a restored image of shape (batch, 3, scale * {height}, "
                f"scale * {width}), found {tuple(shape)}"
            )
            raise ValueError(err_msg)
        self._scale: int = scale

    @property
    def output_schema(self: Self) -> ImageRestorerOutputSchema:
        """Get the output schema used by this image restorer."""
        return self._output_schema

    @property
    def scale(self: Self) -> int:
        """Get the upscale factor of the model, read from the engine."""
        return self._scale

    def _fit(self: Self, images: Sequence[np.ndarray | Buffer]) -> list[np.ndarray | Buffer]:
        """Pad smaller images to the input size, a Buffer through the host."""
        fitted: list[np.ndarray | Buffer] = []
        for image in images:
            if tuple(image.shape[1::-1]) == self._input_size:
                fitted.append(image)
                continue
            array = image.numpy() if isinstance(image, Buffer) else image
            fitted.append(pad_to_input(array, self._input_size))
        return fitted

    @staticmethod
    def _sizes(
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
    ) -> list[tuple[int, int]]:
        """Return the (height, width) of each image."""
        batch = [images] if isinstance(images, (np.ndarray, Buffer)) else images
        return [(int(image.shape[0]), int(image.shape[1])) for image in batch]

    def preprocess(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        resize: str | None = None,
        method: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess the input images.

        Images smaller than the engine input are padded at the bottom and right,
        never scaled; keep their sizes to crop the restored images, see :meth:`run`.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to preprocess.
        resize : str, optional
            Only 'pad' (the default, None) is supported.
        method : str, optional
            The underlying preprocessor to use, as in :meth:`get_preprocessor`.
            By default None, which will use the preprocessor stated in the constructor.
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

        Raises
        ------
        ValueError
            If ``resize`` is not 'pad', or an image is larger than the engine input.

        """
        if resize is not None and resize != "pad":
            err_msg = f"{self._tag}: image restoration does not rescale, got resize={resize!r}"
            raise ValueError(err_msg)
        batch = [images] if isinstance(images, (np.ndarray, Buffer)) else images
        return super().preprocess(self._fit(batch), resize, method, no_copy=no_copy, verbose=verbose)

    def prepare_inputs(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        method: str | None = None,
        *,
        preprocessed: bool = False,
        ratios: list[tuple[float, float]] | None = None,
        padding: list[tuple[float, float]] | None = None,
        verbose: bool | None = None,
    ) -> tuple[list[Buffer], list[tuple[float, float]] | None, list[tuple[float, float]] | None]:
        """
        Get the inputs of the engine call, padding smaller images, see :meth:`ImageModel.prepare_inputs`.

        Parameters
        ----------
        images : Sequence[np.ndarray | Buffer]
            The images to preprocess, or a single already preprocessed batch tensor.
        method : str, optional
            The preprocessor to use, as in :meth:`get_preprocessor`.
        preprocessed : bool, optional
            Whether ``images`` holds an already preprocessed batch tensor.
        ratios : list[tuple[float, float]], optional
            The ratios of an already preprocessed batch.
        padding : list[tuple[float, float]], optional
            The padding of an already preprocessed batch.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        tuple[list[Buffer], list[tuple[float, float]] | None, list[tuple[float, float]] | None]
            The engine inputs, and the ratios and padding per image.

        """
        return super().prepare_inputs(
            images if preprocessed else self._fit(images),
            method,
            preprocessed=preprocessed,
            ratios=ratios,
            padding=padding,
            verbose=verbose,
        )

    def postprocess(
        self: Self,
        outputs: list[np.ndarray] | list[Buffer],
        sizes: list[tuple[int, int]] | None = None,
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
        sizes : list[tuple[int, int]], optional
            The (height, width) of each input image, to crop its restored image to
            ``scale`` times that size. By default None, which keeps the whole output.
        no_copy : bool, optional
            Kept for interface symmetry, the restored images are always new arrays.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        list[list[np.ndarray]]
            The postprocessed outputs per image, each a single uint8 BGR image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["restore_postprocess"])

        if verbose:
            LOG.debug(f"{self._tag}: postprocess")

        t0 = time.perf_counter()
        data = self._postprocess_with(
            partial(postprocess_restoration, no_copy=no_copy),
            outputs,
            sizes,
            self._scale,
            verbose=verbose,
        )
        t1 = time.perf_counter()
        self._post_profile = (t0, t1)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # postprocess

        return data

    def _post_fn(
        self: Self, sizes: list[tuple[int, int]] | None, *, verbose: bool | None
    ) -> _PostFn:
        """Bridge ``_run_core`` / ``_end2end_core`` to :meth:`postprocess`."""
        return lambda o, _r, _p, nc: self.postprocess(o, sizes, no_copy=nc, verbose=verbose)

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray | Buffer,
        sizes: tuple[int, int] | list[tuple[int, int]] | None = ...,
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
        sizes: tuple[int, int] | list[tuple[int, int]] | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray] | list[list[np.ndarray]]: ...

    def __call__(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        sizes: tuple[int, int] | list[tuple[int, int]] | None = None,
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
        sizes : tuple[int, int] | list[tuple[int, int]], optional
            The (height, width) of each image of an already preprocessed batch,
            see :meth:`run`.
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
            sizes,
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
        sizes: tuple[int, int] | list[tuple[int, int]] | None = ...,
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
        sizes: tuple[int, int] | list[tuple[int, int]] | None = ...,
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
        sizes: tuple[int, int] | list[tuple[int, int]] | None = ...,
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
        sizes: tuple[int, int] | list[tuple[int, int]] | None = ...,
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
        sizes: tuple[int, int] | list[tuple[int, int]] | None = ...,
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
        sizes: tuple[int, int] | list[tuple[int, int]] | None = ...,
        *,
        preprocessed: bool | None = ...,
        postprocess: bool | None = ...,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> list[np.ndarray]: ...

    def run(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        sizes: tuple[int, int] | list[tuple[int, int]] | None = None,
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
        sizes : tuple[int, int] | list[tuple[int, int]], optional
            The (height, width) of each image, to crop the restored images to.
            By default None, which takes them from the images, or keeps the whole
            output for an already preprocessed batch.
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
            If preprocessed inputs are not a single batch tensor, or an image is
            larger than the engine input.

        """
        if isinstance(sizes, tuple):
            sizes = [sizes]
        if sizes is None and not preprocessed:
            sizes = self._sizes(images)
        return self._run_core(
            images,
            None,
            None,
            preprocessed=preprocessed,
            postprocess=postprocess,
            no_copy=no_copy,
            verbose=verbose,
            post=self._post_fn(sizes, verbose=verbose),
            needs_ratios=False,
        )

    # get_restored_images overloads
    @overload
    def get_restored_images(
        self: Self,
        outputs: list[np.ndarray],
        *,
        verbose: bool | None = ...,
    ) -> RestoredImage: ...

    @overload
    def get_restored_images(
        self: Self,
        outputs: list[list[np.ndarray]],
        *,
        verbose: bool | None = ...,
    ) -> list[RestoredImage]: ...

    def get_restored_images(
        self: Self,
        outputs: list[np.ndarray] | list[list[np.ndarray]],
        *,
        verbose: bool | None = None,
    ) -> RestoredImage | list[RestoredImage]:
        """
        Get the restored images from postprocessed outputs.

        Parameters
        ----------
        outputs : list[np.ndarray] | list[list[np.ndarray]]
            For single image: list[np.ndarray] (single image's postprocessed outputs).
            For batch: list[list[np.ndarray]] (postprocessed outputs per image).
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        RestoredImage | list[RestoredImage]
            For single image: the (scale * height, scale * width, 3) uint8 BGR image.
            For batch: the restored image per image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["restore_get_restored_images"])

        if verbose:
            LOG.debug(f"{self._tag}: get_restored_images")

        batch, single = self._as_batch(outputs)
        result = get_restored_images(batch, verbose=verbose)

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # get_restored_images

        return result[0] if single else result

    # end2end overloads
    @overload
    def end2end(
        self: Self,
        images: np.ndarray | Buffer,
        *,
        verbose: bool | None = ...,
    ) -> RestoredImage: ...

    @overload
    def end2end(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        *,
        verbose: bool | None = ...,
    ) -> list[RestoredImage]: ...

    def end2end(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        *,
        verbose: bool | None = None,
    ) -> RestoredImage | list[RestoredImage]:
        """
        Perform end to end inference for a batch of images.

        Equivalent to running preprocess, run, postprocess, and
        get_restored_images in that order. Makes some memory transfer
        optimizations under the hood to improve performance.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to perform inference with.
        verbose : bool, optional
            Whether or not to log additional information.

        Returns
        -------
        RestoredImage | list[RestoredImage]
            For single image: the (scale * height, scale * width, 3) uint8 BGR image.
            For batch: the restored image per image.

        Raises
        ------
        ValueError
            If an image is larger than the engine input.

        """
        return self._end2end_core(
            images,
            verbose=verbose,
            post=self._post_fn(self._sizes(images), verbose=verbose),
            get=lambda pp: get_restored_images(pp, verbose=verbose),
        )
