# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
GPU decoding of images and video straight into device Buffers.

Optional: needs ``nvidia-nvimgcodec`` (images) and/or ``PyNvVideoCodec`` (video).

Functions
---------
:func:`decode_image`
    Decode one encoded image to a device BGR Buffer.

Classes
-------
:class:`VideoReader`
    Iterate a video file as device BGR Buffers.

"""

from __future__ import annotations

from ._image import decode_image
from ._video import VideoReader

__all__ = ["VideoReader", "decode_image"]
