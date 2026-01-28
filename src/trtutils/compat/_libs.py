# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING


class _EmptyModule(ModuleType):
    """Empty module used as fallback when library is not installed."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__name__ = name

    def __getattr__(self, name: str) -> object:
        err_msg = f"{self.__name__} is not installed. Cannot access {name}"
        raise AttributeError(err_msg)


def _import_or_attr(module_name: str, fallback_name: str, attr_name: str | None = None) -> object:
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        if attr_name is not None:
            try:
                base = importlib.import_module("cuda")
                return getattr(base, attr_name)
            except (ImportError, ModuleNotFoundError, AttributeError):
                return _EmptyModule(fallback_name)
        return _EmptyModule(fallback_name)


# define the types for the modules
if TYPE_CHECKING:
    trt: ModuleType
    cuda: ModuleType
    cudart: ModuleType
    nvrtc: ModuleType
    cudart_bindings: ModuleType

# if we are not type checking, actually import the modules
# we handle the behavior of cuda.bindings vs. cuda in the _import_or_attr function
else:
    trt = _import_or_attr("tensorrt", "tensorrt")
    cuda = _import_or_attr("cuda.bindings.driver", "cuda", "cuda")
    cudart = _import_or_attr("cuda.bindings.runtime", "cudart", "cudart")
    nvrtc = _import_or_attr("cuda.bindings.nvrtc", "nvrtc", "nvrtc")
    cudart_bindings = _import_or_attr("cuda.bindings.cudart", "cudart_bindings", "cudart")


__all__ = ["cuda", "cudart", "cudart_bindings", "nvrtc", "trt"]
