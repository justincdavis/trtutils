# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from ._flags import FLAGS

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self


def enable_nvtx() -> None:
    """Enable trtutils NVTX profiling."""
    FLAGS.NVTX_ENABLED = True


def disable_nvtx() -> None:
    """Disable trtutils NVTX profiling."""
    FLAGS.NVTX_ENABLED = False


class NVTX:
    """Context manager for trtutils NVTX profiling."""

    def __init__(self) -> None:
        """Initialize trtutils NVTX context manager."""
        self._pre_enabled = False

    def __enter__(self: Self) -> None:
        """Enter the NVTX context manager."""
        if FLAGS.NVTX_ENABLED:
            self._pre_enabled = True
        else:
            enable_nvtx()

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the NVTX context manager."""
        if not self._pre_enabled:
            disable_nvtx()
