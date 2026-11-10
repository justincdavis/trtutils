# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, List, Tuple, overload

import cv2
import numpy as np
import nvtx
from cv2ext.image import letterbox, resize_linear

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._buffer import Buffer

from ._base import ImagePreprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

_COLOR_CHANNELS = 3

# (images, *, verbose) -> (batch tensor, ratios, padding)
PreprocessFn = Callable[..., Tuple[np.ndarray, List[Tuple[float, float]], List[Tuple[float, float]]]]


def preprocess(
    images: list[np.ndarray],
    input_shape: tuple[int, int],
    dtype: np.dtype,
    input_range: tuple[float, float] = (0.0, 1.0),
    method: str = "letterbox",
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
    out: np.ndarray | None = None,
    *,
    verbose: bool | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Preprocess inputs for a YOLO network.

    Parameters
    ----------
    images : list[np.ndarray]
        The images to be preprocessed.
    input_shape : tuple[int, int]
        The shape to resize the inputs.
    dtype : np.dtype
        The datatype of the inputs to the network.
    input_range : tuple[float, float]
        The range of the model expects for inputs.
        By default, [0.0, 1.0] (divide input by 255.0)
    method : str
        The method by which to resize the image.
        By default letterbox will be used.
        Options are [letterbox, linear]
    mean : tuple[float, float, float], optional
        The mean to subtract from the image.
        By default, None, which will not subtract any mean.
    std : tuple[float, float, float], optional
        The standard deviation to divide the image by.
        By default, None, which will not divide by any standard deviation.
    out : np.ndarray, optional
        A batch tensor to write into, used when its shape and dtype match.
        By default None, which allocates a new tensor.
    verbose : bool, optional
        Whether or not to log additional information.

    Returns
    -------
    tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
        The preprocessed batch tensor, list of ratios, and list of padding per image.

    Raises
    ------
    ValueError
        If the method for resizing is not 'letterbox' or 'linear'
    ValueError
        If only one of mean or std is provided

    """
    # asses if mean and std are provided and valid
    if mean is not None and std is None:
        err_msg = "Mean provided but std is not"
        raise ValueError(err_msg)
    if std is not None and mean is None:
        err_msg = "Std provided but mean is not"
        raise ValueError(err_msg)

    if method not in ("letterbox", "linear"):
        err_msg = "Unknown method for image resizing. Options are ['letterbox', 'linear']"
        raise ValueError(err_msg)

    # fold the normalization into one float32 multiply-add per pixel
    if mean is not None and std is not None:
        std_arr = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        scale = 1.0 / (255.0 * std_arr)
        offset = -np.array(mean, dtype=np.float32).reshape(3, 1, 1) / std_arr
    else:
        low, high = input_range
        scale = np.float32((high - low) / 255.0)
        offset = np.float32(low)

    width, height = input_shape
    shape = (len(images), 3, height, width)
    reuse = out is not None and out.shape == shape and out.dtype == dtype and out.flags.c_contiguous
    batch_tensor = out if reuse else np.empty(shape, dtype=dtype)
    # other dtypes go through float32 so the affine never rounds twice
    scratch = None if dtype == np.float32 else np.empty(shape[1:], dtype=np.float32)

    ratios_list: list[tuple[float, float]] = []
    padding_list: list[tuple[float, float]] = []
    for i, image in enumerate(images):
        if method == "letterbox":
            tensor, ratios, padding = letterbox(image, new_shape=input_shape)
        else:
            tensor, ratios = resize_linear(image, new_shape=input_shape)
            padding = (0.0, 0.0)
        tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)

        # writing the transposed view straight into the slot does HWC->CHW,
        # the cast and the affine in one pass
        dst = batch_tensor[i] if scratch is None else scratch
        np.multiply(tensor.transpose(2, 0, 1), scale, out=dst, dtype=np.float32)
        dst += offset
        if scratch is not None:
            batch_tensor[i] = scratch

        if verbose:
            LOG.debug(
                f"Preprocess {image.shape} -> {input_shape}, ratios: {ratios}, padding: {padding}"
            )
        ratios_list.append(ratios)
        padding_list.append(padding)

    return batch_tensor, ratios_list, padding_list


class CPUPreprocessor(ImagePreprocessor):
    """CPU-based preprocessor for image processing models."""

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype,
        resize: str | PreprocessFn = "letterbox",
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
        resize : str | Callable, optional
            The method to resize the image with.
            Options are [letterbox, linear], will use method
            provided in constructor by default.
            Can also be a function doing the whole preprocessing,
            (images, *, verbose) -> (batch tensor, ratios, padding), for models whose
            input shape changes per batch. The function normalizes itself,
            so output_range, mean and std only apply to letterbox and linear,
            and per-call resize and no_copy are not supported.
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
        self._preprocess_fn: PreprocessFn | None = None if isinstance(resize, str) else resize
        super().__init__(
            output_shape=output_shape,
            output_range=output_range,
            dtype=dtype,
            # unused when a function is set
            resize=resize if isinstance(resize, str) else "letterbox",
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

        # grow-only batch tensor for no_copy calls, saves faulting in fresh pages each call
        self._batch: np.ndarray = np.empty(0, dtype=self._o_dtype)

    @property
    def dynamic(self: Self) -> bool:
        """Whether a preprocessing function sizes each batch, see ``resize``."""
        return self._preprocess_fn is not None

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

    # __call__ overloads
    @overload
    def __call__(
        self: Self,
        images: np.ndarray | Buffer,
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def __call__(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    def __call__(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess images for the model.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to preprocess.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]
            Not supported when the preprocessor was built with a function.
        no_copy : bool, optional
            If True, write into a reused buffer and return a view of it,
            valid until the next call. By default False.
            Not supported when the preprocessor was built with a function.
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
        images: np.ndarray | Buffer,
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    @overload
    def preprocess(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        resize: str | None = ...,
        *,
        no_copy: bool | None = ...,
        verbose: bool | None = ...,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]: ...

    def preprocess(
        self: Self,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        resize: str | None = None,
        *,
        no_copy: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess images for the model.

        Parameters
        ----------
        images : np.ndarray | Buffer | Sequence[np.ndarray | Buffer]
            A single image (HWC format) or list of images to preprocess.
        resize : str
            The method to resize the image with.
            By default letterbox, options are [letterbox, linear]
            Not supported when the preprocessor was built with a function.
        no_copy : bool, optional
            If True, write into a reused buffer and return a view of it,
            valid until the next call. By default False.
            Not supported when the preprocessor was built with a function.
        verbose : bool, optional
            Whether or not to output additional information
            to stdout. If not provided, will default to overall
            engines verbose setting.

        Returns
        -------
        tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
            The preprocessed batch tensor, list of ratios, and list of padding per image.

        Raises
        ------
        ValueError
            If resize or no_copy is given while a preprocessing function is set.

        """
        if self._preprocess_fn is not None:
            return self._preprocess_with_fn(
                self._preprocess_fn, images, resize, no_copy=no_copy, verbose=verbose
            )

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["cpu_preprocess"])

        # Handle single-image input
        if isinstance(images, (np.ndarray, Buffer)):
            images = [images]
        batch_images = [img.numpy() if isinstance(img, Buffer) else img for img in images]

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
        out = None
        if no_copy:
            width, height = self._o_shape
            shape = (len(batch_images), _COLOR_CHANNELS, height, width)
            size = math.prod(shape)
            if self._batch.size < size:
                self._batch = np.empty(size, dtype=self._o_dtype)
            out = self._batch[:size].reshape(shape)
        result = preprocess(
            batch_images,
            self._o_shape,
            self._o_dtype,
            self._o_range,
            resize,
            mean_tuple,
            std_tuple,
            out,
            verbose=verbose,
        )

        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # cpu_preprocess

        return result

    def _preprocess_with_fn(
        self: Self,
        fn: PreprocessFn,
        images: np.ndarray | Buffer | Sequence[np.ndarray | Buffer],
        resize: str | None,
        *,
        no_copy: bool | None,
        verbose: bool | None,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
        if resize is not None or no_copy:
            err_msg = (
                f"{self._tag}: resize and no_copy are not supported with a preprocessing function"
            )
            raise ValueError(err_msg)
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["cpu_preprocess"])
        if isinstance(images, (np.ndarray, Buffer)):
            images = [images]
        batch_images = [img.numpy() if isinstance(img, Buffer) else img for img in images]
        tensor, ratios, padding = fn(batch_images, verbose=verbose)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # cpu_preprocess
        return tensor.astype(self._o_dtype, copy=False), ratios, padding
