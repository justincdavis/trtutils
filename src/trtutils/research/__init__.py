# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementations for research papers.

Submodules
----------
:mod:`axonn`
    Implementation of the AxoNN paper for energy-aware multi-accelerator
    neural network inference optimization.

"""

from __future__ import annotations

import contextlib
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    from . import axonn as axonn

_SUBMODULES = ["axonn"]
_LAZY_SUBMODULES = {"axonn"}

__all__ = [
    "axonn",
    "discover_submodules",
    "register_cli",
]


def __getattr__(name: str) -> object:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    err_msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(err_msg)


def __dir__() -> list[str]:
    return list(__all__)


def discover_submodules() -> list[str]:
    """Return a list of importable research submodule names."""
    available: list[str] = []
    for name in _SUBMODULES:
        with contextlib.suppress(ImportError):
            __import__(f"trtutils.research.{name}")
            available.append(name)
    return available


def register_cli(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
    """Register CLI subcommands for all available research submodules."""
    for name in discover_submodules():
        with contextlib.suppress(ImportError):
            mod = __import__(f"trtutils.research.{name}._cli", fromlist=["register_cli"])
            mod.register_cli(subparsers, parents)
