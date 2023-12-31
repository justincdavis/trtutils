"""A package for enabling high-level usage of TensorRT in Python."""

__author__ = "Justin Davis"
__version__ = "0.0.2"

import contextlib

with contextlib.suppress(ImportError):
    import pycuda.autoinit  # noqa: F401

from ._engine import TRTEngine
from ._model import TRTModel

__all__ = [
    "TRTEngine",
    "TRTModel",
]
