# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import tensorrt as trt

from ._log import LOG

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass
class Config:
    """Class for storing config information on trtutils."""

    _PLUGINS_LOADED: bool = field(default=False, init=False)

    def load_plugins(self: Self) -> None:
        """Load the libnvinfer plugins for TensorRT."""
        if not Config._PLUGINS_LOADED:
            trt.init_libnvinfer_plugins(LOG, "")
            Config._PLUGINS_LOADED = True


CONFIG = Config()
