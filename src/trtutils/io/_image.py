# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import functools
from pathlib import Path

from trtutils._flags import FLAGS
from trtutils.compat._libs import cudart
from trtutils.core import Buffer, cuda_call

with contextlib.suppress(ImportError):
    from nvidia import nvimgcodec

    FLAGS.FOUND_NVIMGCODEC = True


def _require_nvimgcodec() -> None:
    if not FLAGS.FOUND_NVIMGCODEC:
        err_msg = (
            "nvImageCodec is not installed. Install it with "
            "`pip install trtutils[nvimgcodec-cu12]` (or -cu13)."
        )
        raise ImportError(err_msg)


@functools.lru_cache(maxsize=1)
def _decoder() -> nvimgcodec.Decoder:
    return nvimgcodec.Decoder(device_id=0)


@functools.lru_cache(maxsize=1)
def _bgr_params() -> nvimgcodec.DecodeParams:
    # I_BGR with the default backend is bit-identical to cv2.imread
    return nvimgcodec.DecodeParams(sample_format=nvimgcodec.SampleFormat.I_BGR)


def _read(source: str | Path | bytes) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    return bytes(source)


def _wrap(image: object | None, source: object) -> Buffer:
    if image is None:
        err_msg = f"nvImageCodec could not decode {source!r}"
        raise ValueError(err_msg)
    return Buffer.wrap(image)


def decode_image(source: str | Path | bytes) -> Buffer:
    """
    Decode an encoded image (JPEG, PNG, ...) straight to device memory.

    The result is uint8 HWC in BGR order, bit-identical to ``cv2.imread`` for JPEG
    and ready for any trtutils image model. The call returns after the decode has
    finished on the device (the null stream is synchronized), so engine streams
    may read the Buffer immediately. The Buffer keeps the decoder allocation alive.

    Parameters
    ----------
    source : str | Path | bytes
        An image file path or the encoded bytes.

    Returns
    -------
    Buffer
        Device Buffer of shape (height, width, 3), dtype uint8.

    Raises
    ------
    ImportError
        If nvImageCodec is not installed.
    ValueError
        If the image cannot be decoded.

    """
    _require_nvimgcodec()
    data = _read(source)
    image = _decoder().decode(data, params=_bgr_params())
    cuda_call(cudart.cudaStreamSynchronize(0))
    return _wrap(image, source if isinstance(source, (str, Path)) else "bytes")
