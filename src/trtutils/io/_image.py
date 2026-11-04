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


@functools.lru_cache(maxsize=1)
def _decoder() -> nvimgcodec.Decoder:
    return nvimgcodec.Decoder(device_id=0)


@functools.lru_cache(maxsize=1)
def _bgr_params() -> nvimgcodec.DecodeParams:
    # I_BGR with the default backend is bit-identical to cv2.imread
    return nvimgcodec.DecodeParams(sample_format=nvimgcodec.SampleFormat.I_BGR)


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
    if not FLAGS.FOUND_NVIMGCODEC:
        err_msg = (
            "nvImageCodec is not installed. Install it with "
            "`pip install trtutils[nvimgcodec-cu11]` "
            "(nvimgcodec-cu12 and nvimgcodec-cu13 require Python 3.9+)."
        )
        raise ImportError(err_msg)
    is_path = isinstance(source, (str, Path))
    data = Path(source).read_bytes() if is_path else source
    image = _decoder().decode(data, params=_bgr_params())
    if image is None:
        err_msg = f"nvImageCodec could not decode {source if is_path else 'bytes'!r}"
        raise ValueError(err_msg)
    cuda_call(cudart.cudaStreamSynchronize(0))
    return Buffer.wrap(image)
