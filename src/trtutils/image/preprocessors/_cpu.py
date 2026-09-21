# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import nvtx

from trtutils._flags import FLAGS
from trtutils.core._buffer import Buffer, MemoryLocation

from ._image_preproc import ImagePreprocessor, normalize_image_input
from ._process import preprocess

if TYPE_CHECKING:
    from typing_extensions import Self

    from trtutils.image.interfaces import ImageInput

_COLOR_CHANNELS = 3
# glibc mmaps any allocation above 32 MB and unmaps it on free, so the whole
# buffer first-touches fresh zero pages on every call. Reusing the batch
# tensor avoids that; below the threshold the allocator recycles the block and
# reuse would only add bookkeeping. Set below 32 MB so the transition is
# covered rather than straddled.
_REUSE_MIN_BYTES = 16 * 1024 * 1024


class CPUPreprocessor(ImagePreprocessor):
    """CPU-based preprocessor for image processing models."""

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype,
        resize: str = "letterbox",
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        tag: str | None = None,
    ) -> None:
        """
        Create a CPUPreprocessor for image processing models.

        Parameters
        ----------
        output_shape : tuple[int, int]
            The shape of the image the model expects.
            In the form [width, height]
        output_range : tuple[float, float]
            The range of the image values the model expects.
            Examples: (0.0, 1.0), (0.0, 255.0)
        dtype : np.dtype
            The datatype of the image.
            Examples: np.float32, np.float16, np.uint8
        resize : str, optional
            The method to resize the image with.
            Options are [letterbox, linear], will use method
            provided in constructor by default.
        mean : tuple[float, float, float], optional
            The mean to subtract from the image.
            By default, None, which will not subtract any mean.
        std : tuple[float, float, float], optional
            The standard deviation to divide the image by.
            By default, None, which will not divide by any standard deviation.
        tag : str
            The tag to prefix to all logging statements made.
            By default, 'CPUPreprocessor'
            If used within a model class, will be the model tag.

        """
        tag = "CPUPreprocessor" if tag is None else f"{tag}.CPUPreprocessor"
        super().__init__(
            output_shape=output_shape,
            output_range=output_range,
            dtype=dtype,
            resize=resize,
            mean=mean,
            std=std,
            tag=tag,
        )

        self._nvtx_tags.update(
            {
                "cpu_warmup": f"preproc::cpu_warmup [{self._tag}]",
                "cpu_preprocess": f"preproc::cpu_preprocess [{self._tag}]",
            }
        )

        # Grow-only batch tensor, mirroring the GPU preprocessors' staging
        # buffers. Allocating this fresh per call is not merely an allocation:
        # past glibc's 32 MB mmap ceiling the buffer is mmap'd and munmap'd
        # every call, so every write first-touches a new zero page. For a
        # 640x640 float32 batch that ceiling falls at batch 7, where the cost
        # jumps roughly fivefold.
        self._batch_buffer: Buffer | None = None

    def warmup(self: Self) -> None:
        """Compatibility function for CPU/CUDA parity."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["cpu_warmup"])

        rand_data: np.ndarray = np.random.default_rng().integers(
            0,
            255,
            (*self._o_shape, 3),
            dtype=np.uint8,
        )
        self.preprocess([rand_data])

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # cpu_warmup

    def _to_host_image(self: Self, image: ImageInput) -> np.ndarray:
        """
        Get a host ndarray for an ImageInput.

        A device Buffer is copied to host once (one D2H copy via
        ``numpy()``); a host Buffer is unwrapped zero-copy; a plain
        ``np.ndarray`` passes through unchanged.
        """
        if not isinstance(image, Buffer):
            return image
        normalize_image_input(image, self._tag)  # validates shape/dtype
        return image.numpy()

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: ImageInput,
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def __call__(
        self: Self,
        images: list[ImageInput],
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    def __call__(
        self: Self,
        images: ImageInput | list[ImageInput],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess images for the model.

        Parameters
        ----------
        images : ImageInput | list[ImageInput]
            A single image or list of images, each an HWC uint8
            ``np.ndarray`` or a ``Buffer`` (host or device) holding one.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]
        no_copy : bool, optional
            If True, return a view of the preprocessor's reusable batch
            buffer rather than a copy. The view is valid until the next
            preprocess call. By default False, which returns a private copy.
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.

        Returns
        -------
        tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
            The preprocessed batch tensor, list of ratios, and list of padding per image.

        """
        return self.preprocess(images, resize=resize, no_copy=no_copy, verbose=verbose)

    # preprocess overloads
    @overload
    def preprocess(
        self: Self,
        images: ImageInput,
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def preprocess(
        self: Self,
        images: list[ImageInput],
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    def preprocess(
        self: Self,
        images: ImageInput | list[ImageInput],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess images for the model.

        Parameters
        ----------
        images : ImageInput | list[ImageInput]
            A single image or list of images, each an HWC uint8
            ``np.ndarray`` or a ``Buffer`` (host or device) holding one. A
            device Buffer is copied to host once (one D2H copy) before
            preprocessing.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]
        no_copy : bool, optional
            If True, return a view of the preprocessor's reusable batch
            buffer rather than a copy. The view is valid until the next
            preprocess call. By default False, which returns a private copy.
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.

        Returns
        -------
        tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
            The preprocessed batch tensor, list of ratios, and list of padding per image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["cpu_preprocess"])

        # Handle single-image input
        if isinstance(images, (np.ndarray, Buffer)):
            batch_images: list[ImageInput] = [images]
        else:
            batch_images = images

        # a device Buffer needs a host array to run the CPU kernels against
        # (one D2H copy); a host Buffer is unwrapped zero-copy, a plain
        # ndarray passes through
        host_images: list[np.ndarray] = [self._to_host_image(img) for img in batch_images]

        resize = resize if resize is not None else self._resize
        mean = self._mean
        std = self._std
        mean_tuple: tuple[float, float, float] | None = None
        std_tuple: tuple[float, float, float] | None = None
        if mean is not None:
            mean_tuple = tuple(
                mean.reshape(-1)
                if mean.size == _COLOR_CHANNELS
                else mean.flatten()[:_COLOR_CHANNELS]
            )
        if std is not None:
            std_tuple = tuple(
                std.reshape(-1) if std.size == _COLOR_CHANNELS else std.flatten()[:_COLOR_CHANNELS]
            )
        width, height = self._o_shape
        buffer = self._resolve_batch_buffer(len(host_images), height, width)

        tensor, ratios, padding = preprocess(
            host_images,
            self._o_shape,
            self._o_dtype,
            self._o_range,
            resize,
            mean_tuple,
            std_tuple,
            buffer,
            verbose=verbose,
        )

        # the buffer is reused by the next call, so a caller keeping the
        # result needs its own copy unless it opted out
        if not no_copy and tensor is buffer:
            tensor = tensor.copy()

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # cpu_preprocess

        return tensor, ratios, padding

    def _resolve_batch_buffer(
        self: Self,
        batch_size: int,
        height: int,
        width: int,
    ) -> np.ndarray | None:
        """
        Get a reusable batch tensor, growing the allocation only when needed.

        Parameters
        ----------
        batch_size : int
            The number of images in the batch.
        height : int
            The height of the model input.
        width : int
            The width of the model input.

        Returns
        -------
        np.ndarray, optional
            A prefix view of the reusable batch buffer shaped
            (batch_size, 3, height, width), or None when the requested
            geometry cannot be served, in which case the preprocessor
            falls back to allocating per call.

        """
        shape = (batch_size, _COLOR_CHANNELS, height, width)
        needed = batch_size * _COLOR_CHANNELS * height * width
        if needed * self._o_dtype.itemsize < _REUSE_MIN_BYTES:
            # Small batches come from the heap, where the allocator recycles
            # the block and the pages stay mapped. Reusing a buffer there only
            # adds bookkeeping, and handing back a private array is the safer
            # default. Reuse earns its keep once the allocation is large
            # enough to be mmap'd.
            return None
        current = self._batch_buffer
        if current is None or current.size < needed or current.dtype != self._o_dtype:
            try:
                self._batch_buffer = Buffer.empty((needed,), self._o_dtype, MemoryLocation.HOST)
            except (MemoryError, ValueError, RuntimeError):
                self._batch_buffer = None
                return None
            current = self._batch_buffer
        # a prefix view keeps smaller batches on the same allocation
        return current.array[:needed].reshape(shape)
