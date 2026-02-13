# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils.compat._libs import cudart

from ._cuda import cuda_call

if TYPE_CHECKING:
    from typing_extensions import Self


def get_device() -> int:
    """
    Get the current CUDA device.

    Returns
    -------
    int
        The current CUDA device index.

    """
    return cuda_call(cudart.cudaGetDevice())


def set_device(device: int) -> None:
    """
    Set the current CUDA device.

    Parameters
    ----------
    device : int
        The CUDA device index to set.

    """
    cuda_call(cudart.cudaSetDevice(device))


def get_device_count() -> int:
    """
    Get the number of CUDA devices available.

    Returns
    -------
    int
        The number of CUDA devices.

    """
    return cuda_call(cudart.cudaGetDeviceCount())


class DeviceGuard:
    """
    Context manager that saves and restores the current CUDA device.

    When ``device`` is ``None`` the guard is a no-op: ``__enter__`` and
    ``__exit__`` only check a single attribute, adding negligible overhead
    on the hot path.

    Instances are **reusable** — engines store one as ``self._device_guard``
    and enter/exit it on every ``execute()`` call.
    """

    __slots__ = ("_device", "_previous")

    def __init__(self, device: int | None) -> None:
        self._device = device
        self._previous: int = 0

    def __enter__(self: Self) -> Self:
        if self._device is not None:
            self._previous = get_device()
            set_device(self._device)
        return self

    def __exit__(self, *args: object) -> None:
        if self._device is not None:
            set_device(self._previous)
