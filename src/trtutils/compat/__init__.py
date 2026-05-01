# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Compatibility wrappers for other libraries.

Submodules
----------
:mod:`sahi`
    Compatibility wrappers for the SAHI library.

"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import sahi as sahi

_LAZY_SUBMODULES = {"sahi"}

__all__ = ["sahi"]


def __getattr__(name: str) -> object:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    err_msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(err_msg)


def __dir__() -> list[str]:
    return list(__all__)
