# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import math
from functools import cached_property
from typing import TYPE_CHECKING, Any, NamedTuple, overload

import cv2
import numpy as np
import nvtx

from trtutils._flags import FLAGS
from trtutils._log import LOG
from trtutils.core._bindings import create_binding
from trtutils.core._buffer import DEVICE_ALIGNMENT, Buffer, MemoryLocation
from trtutils.core._engine import create_engine
from trtutils.core._kernels import Kernel, cached_args
from trtutils.core._memory import memcpy_host_to_device_async
from trtutils.core._stream import create_stream, destroy_stream, stream_synchronize
from trtutils.image.kernels import BGR_TO_BGRX_INT8, HALF_TO_FLOAT

from ._base import DEFAULT_CROP_PCT, ImagePreprocessor
from ._cpu import resize_image
from ._dla_network import (
    INPUT_NAME,
    OUTPUT_NAMES,
    PHASES,
    ResizePlan,
    build_dla_preproc,
    normalization,
    plan_resize,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from trtutils.compat._libs import cudart, trt
    from trtutils.core._bindings import Binding

_COLOR_CHANNELS = 3
_IMAGE_DIMENSIONS = 3
_THREADS = 256
# pixels or values per thread of the conversion kernels
_VECTOR = 4
_HALF_BYTES = 2
_LETTERBOX_FILL = 114.0
# staging slots of whole 64-row blocks of BGRx start on tensorrt's address alignment
_SLOT_ROWS = DEVICE_ALIGNMENT // 4
_XOR_LUT = np.arange(256, dtype=np.uint8) ^ np.uint8(0x80)


class _Runner(NamedTuple):
    plan: ResizePlan
    engine: trt.ICudaEngine
    context: trt.IExecutionContext
    # the int8 BGRx images, one slot per batch entry, mapped for the CPU to write
    staging: Buffer
    flipped: np.ndarray


class _Job(NamedTuple):
    runner: _Runner
    image: np.ndarray | Buffer
    ratios: tuple[float, float]
    padding: tuple[float, float]


class DLAPreprocessor(ImagePreprocessor):
    """
    Preprocessor that resizes, swaps channels and normalizes on a DLA.

    Each input size gets its own DLA engine, built on first use and cached.
    Letterbox and linear resize by whole factors: letterbox when the image spans
    the output width (e.g. 1920x1080 or 1280x720 to 640x640), linear when both
    sides divide. Some inputs wider than 4K UHD (3840 pixels) exceed the DLA's
    limits. Their pixels match the bilinear GPU preprocessors to within a level
    or so. A center crop takes inputs up to 3840 wide whose width is a multiple
    of 16, and resizes like torchvision's antialiased ``Resize`` (box then
    triangle filters), not bilinearly like the CPU and CUDA preprocessors' center
    crop, which differ from it by a few levels on average. The DLA does not pad
    resize. Host images of the sizes or methods the DLA cannot resize can be
    resized on the CPU instead (``cpu_resize``), with the DLA doing the rest at
    the output size. Images go
    to the DLA as int8 BGRx, converted on the CPU for numpy arrays and by a small
    kernel for device Buffers; the GPU otherwise only casts the float16 output
    for float32 models.
    """

    def __init__(
        self: Self,
        output_shape: tuple[int, int],
        output_range: tuple[float, float],
        dtype: np.dtype[Any],
        batch_size: int = 1,
        resize: str = "letterbox",
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        stream: cudart.cudaStream_t | None = None,
        tag: str | None = None,
        *,
        dla_core: int = 0,
        pagelocked_mem: bool | None = None,
        unified_mem: bool | None = None,
        crop_pct: float = DEFAULT_CROP_PCT,
        cpu_resize: bool | None = None,
    ) -> None:
        """
        Create a DLAPreprocessor for image processing models.

        Parameters
        ----------
        output_shape : tuple[int, int]
            The shape of the image the model expects, in the form [width, height].
            The width must be a multiple of 16.
        output_range : tuple[float, float]
            The range of the image values the model expects.
            Examples: (0.0, 1.0), (0.0, 255.0)
        dtype : np.dtype
            The datatype of the image, float16 or float32. The DLA writes float16;
            float32 adds a cast on the GPU.
        batch_size : int
            The most images per call. Default is 1.
        resize : str, optional
            The default resize method, 'letterbox', 'linear', 'center_crop' or 'pad'.
            By default, letterbox resizing will be used. 'pad' always resizes host
            images on the CPU and raises for device Buffers or with ``cpu_resize=False``.
        mean : tuple[float, float, float], optional
            The mean to subtract from the image.
            By default, None, which will not subtract any mean.
        std : tuple[float, float, float], optional
            The standard deviation to divide the image by.
            By default, None, which will not divide by any standard deviation.
        stream : cudart.cudaStream_t, optional
            The CUDA stream to use for preprocessing execution.
            If not provided, the preprocessor will use its own stream.
        tag : str
            The tag to prefix to all logging statements made.
            By default, 'DLAPreprocessor'
            If used within a model class, will be the model tag.
        dla_core : int
            The DLA core to run on. Default is 0.
        pagelocked_mem : bool, optional
            Whether or not to allocate output memory as pagelocked.
            By default, pagelocked memory will be used.
        unified_mem : bool, optional
            Whether or not the system has unified memory.
            If True, use cudaHostAllocMapped to take advantage of unified memory.
            By default None, which means the default host allocation will be used.
        crop_pct : float
            The fraction of the resized image kept by center_crop resizing.
            By default 0.875 (resize 256, crop 224). Ignored by other methods.
        cpu_resize : bool, optional
            Whether to resize host images on the CPU, as :class:`CPUPreprocessor`
            does, and only normalize on the DLA. True always does (its center crop is
            bilinear, not the DLA's antialiased one), False never does. By default
            None, which does for the sizes the DLA cannot resize, see :meth:`resizes_on_cpu`.

        Raises
        ------
        RuntimeError
            If the system has no DLA, or TensorRT lacks per-tensor INT8 dynamic
            ranges (TensorRT 11) or ``execute_async_v3``.
        ValueError
            If the DLA core does not exist, the resize method, dtype or output
            width is not supported, or crop_pct is not in (0, 1].

        """
        tag = "DLAPreprocessor" if tag is None else f"{tag}.DLAPreprocessor"
        if not FLAGS.HAS_DLA:
            err_msg = f"{tag}: this system has no DLA"
            raise RuntimeError(err_msg)
        if not FLAGS.DYNAMIC_RANGE or not FLAGS.EXEC_ASYNC_V3:
            err_msg = (
                f"{tag}: the DLA preprocessor needs per-tensor INT8 dynamic ranges and "
                f"execute_async_v3, which TensorRT {FLAGS.TRT_VERSION} does not have"
            )
            raise RuntimeError(err_msg)
        if not 0 <= dla_core < FLAGS.NUM_DLA_CORES:
            err_msg = (
                f"{tag}: DLA core {dla_core} does not exist, this system has {FLAGS.NUM_DLA_CORES}"
            )
            raise ValueError(err_msg)
        if np.dtype(dtype) not in (np.float16, np.float32):
            err_msg = f"{tag}: the DLA preprocessor outputs float16 or float32, not {dtype}"
            raise ValueError(err_msg)
        if output_shape[0] % PHASES:
            err_msg = f"{tag}: the DLA preprocessor needs an output width that is a multiple of 16"
            raise ValueError(err_msg)
        self._batch_size = batch_size
        self._runners: dict[tuple[int, int, str], _Runner] = {}
        self._cpu_resize = cpu_resize
        self._on_cpu: dict[tuple[int, int, str], bool] = {}
        super().__init__(
            output_shape, output_range, dtype, resize, mean, std, tag, crop_pct=crop_pct
        )

        self._nvtx_tags.update(
            {
                "dla_preprocess": f"preproc::dla_preprocess [{self._tag}]",
                "dla_direct_preproc": f"preproc::dla_direct_preproc [{self._tag}]",
            }
        )
        self._dla_core = dla_core
        self._own_stream = stream is None
        self._stream: cudart.cudaStream_t = create_stream() if stream is None else stream

        width, height = self._o_shape
        shape = (batch_size, _COLOR_CHANNELS, height, width)
        self._output = create_binding(
            np.zeros(shape, dtype=self._o_dtype),
            pagelocked_mem=pagelocked_mem if pagelocked_mem is not None else True,
            unified_mem=unified_mem,
        )
        # the DLA writes float16; float32 models get a cast of it
        self._half = self._output.device
        self._cast: Kernel | None = None
        if self._o_dtype != np.float16:
            self._half = Buffer.empty(shape, np.float16, MemoryLocation.DEVICE)
            self._cast = Kernel(*HALF_TO_FLOAT)

        self._norm = normalization(self._o_range, self._mean, self._std)
        fill = (_LETTERBOX_FILL * self._norm[0] + self._norm[1]).astype(np.float16)
        self._border = np.ascontiguousarray(
            np.broadcast_to(fill.reshape(-1, 1, 1), (_COLOR_CHANNELS, height, width))
        )
        # the image rows each batch entry was last written with
        self._slot_rows: list[int | None] = [None] * batch_size

    def __del__(self: Self) -> None:
        # a borrowed stream may be destroyed already
        with contextlib.suppress(AttributeError, RuntimeError):
            if self._own_stream and self._runners:
                stream_synchronize(self._stream)
        with contextlib.suppress(AttributeError):
            del self._runners
        with contextlib.suppress(AttributeError, RuntimeError):
            if self._own_stream:
                self._own_stream = False
                destroy_stream(self._stream)

    @property
    def output_binding(self: Self) -> Binding:
        """The binding holding the preprocessed batch."""
        return self._output

    @cached_property
    def _convert(self: Self) -> Kernel:
        return Kernel(*BGR_TO_BGRX_INT8)

    def warmup(self: Self) -> None:
        """Run each DLA engine loaded so far once, with cpu_resize the output size's first."""
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["warmup"])
        if self._cpu_resize:
            width, height = self._o_shape
            self._runner(height, width, "linear", verbose=None)
        for height, width, method in self._runners:
            image = Buffer.empty((height, width, _COLOR_CHANNELS), np.uint8, MemoryLocation.DEVICE)
            self.direct_preproc([image], method, no_warn=True)
        stream_synchronize(self._stream)
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # warmup

    def _runner(
        self: Self, height: int, width: int, method: str, *, verbose: bool | None
    ) -> _Runner:
        """Get the DLA engine for an input size, building it on first use."""
        key = (height, width, method)
        runner = self._runners.get(key)
        if runner is None:
            if verbose:
                LOG.debug(f"{self._tag}: loading the DLA engine for {width}x{height} {method}")
            plan = plan_resize(height, width, self._o_shape, method, self._crop_pct)
            path = build_dla_preproc(plan, *self._norm, self._dla_core)
            engine, context, _, _ = create_engine(
                path, stream=self._stream, dla_core=self._dla_core, no_warn=True
            )
            rows = len(plan.input_rows)
            slot_rows = math.ceil(rows / _SLOT_ROWS) * _SLOT_ROWS
            staging = Buffer.empty(
                (self._batch_size, slot_rows, width, 4), np.uint8, pinned=True, mapped=True
            )
            flipped = np.empty((rows, width, _COLOR_CHANNELS), dtype=np.uint8)
            runner = _Runner(plan, engine, context, staging, flipped)
            self._runners[key] = runner
        return runner

    def resizes_on_cpu(self: Self, height: int, width: int, resize: str | None = None) -> bool:
        """
        Whether host images of a size are resized on the CPU before the DLA.

        With ``cpu_resize=None`` the first call for a size and method decides: the
        CPU resizes when the DLA plan or engine build fails. Device Buffers are
        always resized on the DLA.

        Parameters
        ----------
        height : int
            The image height.
        width : int
            The image width.
        resize : str, optional
            The resize method. By default None, the method given in the constructor.

        Returns
        -------
        bool
            Whether the CPU resizes such images to the output shape.

        """
        if self._cpu_resize is not None:
            return self._cpu_resize
        method = resize if resize is not None else self._resize
        key = (height, width, method)
        if key not in self._on_cpu:
            try:
                plan = plan_resize(height, width, self._o_shape, method, self._crop_pct)
                build_dla_preproc(plan, *self._norm, self._dla_core)
            except (ValueError, RuntimeError) as err:
                LOG.info(f"{self._tag}: resizing {width}x{height} images on the CPU: {err}")
                self._on_cpu[key] = True
            else:
                self._on_cpu[key] = False
        return self._on_cpu[key]

    def _prepare(
        self: Self, image: np.ndarray | Buffer, method: str, *, verbose: bool | None
    ) -> _Job:
        """Get the engine, the image it reads, and the ratios and padding of one image."""
        if (
            image.ndim != _IMAGE_DIMENSIONS
            or image.shape[2] != _COLOR_CHANNELS
            or image.dtype != np.uint8
        ):
            err_msg = f"{self._tag}: Image must be uint8 (height, width, channels=3)"
            raise ValueError(err_msg)
        height, width = image.shape[:2]
        on_device = isinstance(image, Buffer) and image.is_device
        if not on_device and self.resizes_on_cpu(height, width, method):
            host = image.array if isinstance(image, Buffer) else image
            resized, ratios, padding = resize_image(host, self._o_shape, method, self._crop_pct)
            out_width, out_height = self._o_shape
            runner = self._runner(out_height, out_width, "linear", verbose=verbose)
            return _Job(runner, resized, ratios, padding)
        runner = self._runner(height, width, method, verbose=verbose)
        # the DLA reads device and mapped buffers through their device address
        if isinstance(image, Buffer) and not image.device_visible:
            image = image.array
        return _Job(runner, image, runner.plan.ratios, runner.plan.padding)

    def _launch(self: Self, kernel: Kernel, count: int, *args: int) -> None:
        """Launch a conversion kernel over ``count`` pixels or values."""
        blocks = (math.ceil(count / _VECTOR / _THREADS), 1, 1)
        kernel.call(blocks, (_THREADS, 1, 1), self._stream, cached_args(kernel, *args, count))

    def _stage(self: Self, runner: _Runner, slot: int, image: np.ndarray | Buffer) -> int:
        """Write the rows the engine reads as int8 BGRx into a staging slot, giving its address."""
        rows = runner.plan.input_rows
        slot_bgrx = runner.staging.array[slot]
        bgrx = slot_bgrx[: len(rows)]
        address = runner.staging.device_ptr + slot * slot_bgrx.nbytes
        if isinstance(image, Buffer):
            source = image.device_ptr + rows.start * bgrx.shape[1] * _COLOR_CHANNELS
            self._launch(self._convert, bgrx.shape[0] * bgrx.shape[1], source, address)
        else:
            cv2.LUT(image[rows.start : rows.stop], _XOR_LUT, dst=runner.flipped)
            cv2.cvtColor(runner.flipped, cv2.COLOR_BGR2BGRA, dst=bgrx)
        return address

    def _enqueue(self: Self, runner: _Runner, slot: int, image: np.ndarray | Buffer) -> bool:
        """Enqueue the DLA engine on one image, writing batch entry ``slot`` of the output."""
        plan = runner.plan
        width, height = self._o_shape
        plane = height * width * _HALF_BYTES
        slot_ptr = self._half.device_ptr + slot * _COLOR_CHANNELS * plane
        # the engine writes only the image rows, the letterbox border stays from the fill
        if plan.rows != height and self._slot_rows[slot] != plan.rows:
            memcpy_host_to_device_async(slot_ptr, self._border, self._stream)
        self._slot_rows[slot] = plan.rows
        context = runner.context
        context.set_tensor_address(INPUT_NAME, self._stage(runner, slot, image))
        rows_ptr = slot_ptr + plan.top * width * _HALF_BYTES
        for color, name in enumerate(OUTPUT_NAMES):
            context.set_tensor_address(name, rows_ptr + color * plane)
        return bool(context.execute_async_v3(self._stream))

    def _cast_output(self: Self, batch_size: int) -> None:
        """Cast the float16 DLA output of a float32 model's batch."""
        if self._cast is None or batch_size == 0:
            return
        width, height = self._o_shape
        count = batch_size * _COLOR_CHANNELS * height * width
        self._launch(self._cast, count, self._half.device_ptr, self._output.allocation)

    def direct_preproc(
        self: Self,
        images: Sequence[np.ndarray | Buffer],
        resize: str | None = None,
        *,
        no_warn: bool | None = None,
        verbose: bool | None = None,
    ) -> tuple[Buffer, list[tuple[float, float]], list[tuple[float, float]]]:
        """
        Preprocess images on the DLA, leaving the batch on the device.

        Parameters
        ----------
        images : Sequence[np.ndarray | Buffer]
            The images to preprocess (HWC format, uint8, BGR).
        resize : str, optional
            The method to resize the images with, 'letterbox', 'linear', 'center_crop' or 'pad'.
            By default None, which uses the method given in the constructor.
        no_warn : bool, optional
            If True, do not warn about usage.
        verbose : bool, optional
            Whether or not to output additional information.

        Returns
        -------
        tuple[Buffer, list[tuple[float, float]], list[tuple[float, float]]]
            Device Buffer of the preprocessed batch, shaped (batch, 3, height, width)
            for exactly the given images, list of ratios, and list of padding per image.
            The Buffer views the preprocessor's output memory and is overwritten by the
            next call.

        Raises
        ------
        ValueError
            If there are more images than the batch size, the resize method is
            unknown, an image is not a uint8 color image, or the DLA cannot resize
            an image the CPU does not (see the class docstring).
        RuntimeError
            If TensorRT cannot build the DLA engine for an image the CPU does not
            resize (some inputs wider than 4K UHD), or fails to enqueue it.

        """
        if verbose:
            LOG.debug(f"{self._tag}: direct_preproc")
        if not no_warn:
            LOG.warning(
                "Calling direct_preproc is potentially dangerous. Outputs can be overwritten inplace!",
            )
        if len(images) > self._batch_size:
            err_msg = f"{self._tag}: Batch size {len(images)} exceeds configured batch size {self._batch_size}"
            raise ValueError(err_msg)
        method = resize if resize is not None else self._resize
        jobs = [self._prepare(image, method, verbose=verbose) for image in images]

        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["dla_direct_preproc"])
        if any(isinstance(job.image, np.ndarray) for job in jobs):
            # the DLA may still read the staging the CPU is about to overwrite
            stream_synchronize(self._stream)
        for slot, job in enumerate(jobs):
            if not self._enqueue(job.runner, slot, job.image):
                err_msg = f"{self._tag}: TensorRT failed to enqueue the DLA engine"
                if FLAGS.NVTX_ENABLED:
                    nvtx.pop_range()  # dla_direct_preproc
                raise RuntimeError(err_msg)
        self._cast_output(len(jobs))
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # dla_direct_preproc

        return (
            self._output.device[: len(jobs)],
            [job.ratios for job in jobs],
            [job.padding for job in jobs],
        )

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
        resize : str, optional
            The method to resize the images with, 'letterbox', 'linear', 'center_crop' or 'pad'.
            By default None, which uses the method given in the constructor.
        no_copy : bool, optional
            If True, return the host output memory without a copy. It WILL BE
            OVERWRITTEN INPLACE by future preprocessing calls.
        verbose : bool, optional
            Whether or not to output additional information.

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
        resize : str, optional
            The method to resize the images with, 'letterbox', 'linear', 'center_crop' or 'pad'.
            By default None, which uses the method given in the constructor.
        no_copy : bool, optional
            If True, return the host output memory without a copy. It WILL BE
            OVERWRITTEN INPLACE by future preprocessing calls.
        verbose : bool, optional
            Whether or not to output additional information.

        Returns
        -------
        tuple[np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]
            The preprocessed batch tensor, list of ratios, and list of padding per image.

        """
        if FLAGS.NVTX_ENABLED:
            nvtx.push_range(self._nvtx_tags["dla_preprocess"])
        batch = [images] if isinstance(images, (np.ndarray, Buffer)) else images
        output, ratios, padding = self.direct_preproc(
            batch, resize=resize, no_warn=True, verbose=verbose
        )
        host = self._output.fetch(output.shape, self._stream)
        stream_synchronize(self._stream)
        result = host.array if no_copy else host.array.copy()
        if FLAGS.NVTX_ENABLED:
            nvtx.pop_range()  # dla_preprocess
        return result, ratios, padding
