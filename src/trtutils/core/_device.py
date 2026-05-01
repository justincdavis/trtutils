# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from trtutils._log import LOG
from trtutils.compat._libs import cudart, trt

from ._cuda import cuda_call

if TYPE_CHECKING:
    from typing_extensions import Self


_SM_ARCH_MAP: dict[int, str | dict[int, str]] = {
    3: "kepler",
    5: "maxwell",
    6: "pascal",
    7: {0: "volta", 1: "volta", 2: "volta", 5: "turing"},
    8: {0: "ampere", 6: "ampere", 7: "ampere", 9: "ada"},
    9: "hopper",
    10: "blackwell",
    12: "blackwell",
}


def get_sm_arch(major: int, minor: int) -> str:
    """
    Get the GPU architecture name from a compute capability version.

    Parameters
    ----------
    major : int
        The major compute capability version.
    minor : int
        The minor compute capability version.

    Returns
    -------
    str
        The architecture name (e.g. "turing", "blackwell").
        Returns "unknown" if the compute capability is not recognized.

    """
    if major == 0:
        return "unknown"
    entry = _SM_ARCH_MAP.get(major)
    if entry is None:
        return "unknown"
    if isinstance(entry, str):
        return entry
    return entry.get(minor, "unknown")


def get_device_name(device: int = 0) -> str:
    """
    Get the name of a CUDA device.

    Parameters
    ----------
    device : int, optional
        The CUDA device index. Default is 0.

    Returns
    -------
    str
        The device name (e.g. "NVIDIA GeForce RTX 5080").

    """
    props = cuda_call(cudart.cudaGetDeviceProperties(device))
    name = props.name
    return name.decode() if isinstance(name, bytes) else name


def get_compute_capability(device: int = 0) -> tuple[int, int]:
    """
    Get the compute capability (SM version) of a CUDA device.

    Parameters
    ----------
    device : int, optional
        The CUDA device index. Default is 0.

    Returns
    -------
    tuple[int, int]
        A tuple of (major, minor) compute capability version.

    """
    props = cuda_call(cudart.cudaGetDeviceProperties(device))
    return (props.major, props.minor)


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


def get_num_dla_cores() -> int:
    """
    Get the number of DLA cores available via TensorRT Runtime.

    Returns
    -------
    int
        Number of DLA cores available on the system.

    """
    runtime = trt.Runtime(LOG)
    return runtime.num_DLA_cores


def get_device_count() -> int:
    """
    Get the number of CUDA devices available.

    Returns
    -------
    int
        The number of CUDA devices.

    """
    return cuda_call(cudart.cudaGetDeviceCount())


class Device:
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
