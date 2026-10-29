# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np

from trtutils._flags import FLAGS
from trtutils.compat._libs import cudart
from trtutils.core import Buffer, Kernel, MemoryLocation, cuda_call

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

with contextlib.suppress(ImportError):
    import PyNvVideoCodec

    FLAGS.FOUND_PYNVVIDEOCODEC = True

_RGB_TO_BGR = Path(__file__).parent / "_kernels" / "rgb_to_bgr.cu"
_THREADS = (32, 32, 1)


class VideoReader:
    """
    Iterate a video file as device BGR frames, decoded on the GPU with PyNvVideoCodec.

    Each frame is uint8 HWC BGR. All frames are views of one Buffer owned by the
    reader, so a yielded frame is only valid until the next one is requested; copy
    it (``Buffer.copy_to``) to keep it. Every frame is complete on the device when
    yielded (the null stream is synchronized), so engine streams can read it directly.

    Colour conversion is the decoder's YUV->RGB followed by a small RGB->BGR kernel,
    so pixels differ slightly from OpenCV's conversion (different colour matrix).

    Parameters
    ----------
    path : str | Path
        The video file to read.

    Raises
    ------
    ImportError
        If PyNvVideoCodec is not installed.

    """

    def __init__(self: Self, path: str | Path) -> None:
        if not FLAGS.FOUND_PYNVVIDEOCODEC:
            err_msg = "PyNvVideoCodec is not installed. Install it with `pip install trtutils[nvvideocodec]`."
            raise ImportError(err_msg)
        # the decoder needs an open binary file object, not a path
        self._file: BinaryIO = Path(path).open("rb")  # noqa: SIM115
        try:
            self._decoder = PyNvVideoCodec.CreateSimpleDecoder(
                self._file,
                gpuid=0,
                useDeviceMemory=True,
                outputColorType=PyNvVideoCodec.OutputColorType.RGB,
            )
            meta = self._decoder.get_stream_metadata()
        except BaseException:
            self._file.close()
            raise
        self._num_frames: int = meta.num_frames
        self._shape = (int(meta.height), int(meta.width), 3)
        self._out = Buffer.empty(self._shape, np.uint8, MemoryLocation.DEVICE)
        self._kernel = Kernel(_RGB_TO_BGR, "rgbToBgr")
        height, width, _ = self._shape
        self._blocks = (
            (width + _THREADS[0] - 1) // _THREADS[0],
            (height + _THREADS[1] - 1) // _THREADS[1],
            1,
        )

    def __len__(self: Self) -> int:
        """Return the number of frames in the video."""
        return self._num_frames

    @property
    def shape(self: Self) -> tuple[int, int, int]:
        """Frame shape (height, width, 3)."""
        return self._shape

    def __iter__(self: Self) -> Iterator[Buffer]:
        """Yield each frame as a device BGR Buffer, valid until the next frame."""
        height, width, _ = self._shape
        for i in range(self._num_frames):
            frame = self._decoder[i]
            plane = frame.cuda()[0].__cuda_array_interface__
            ptr = int(plane["data"][0])
            pitch = int(plane["strides"][0])
            args = self._kernel.create_args(ptr, self._out.ptr, width, height, pitch)
            self._kernel.call(self._blocks, _THREADS, cudart.cudaStream_t(0), args)
            # also guarantees the decoder frame is not reused under the kernel
            cuda_call(cudart.cudaStreamSynchronize(0))
            yield self._out

    def close(self: Self) -> None:
        """Close the video file."""
        self._file.close()

    def __enter__(self: Self) -> Self:
        return self

    def __exit__(self: Self, *_: object) -> None:
        self.close()
