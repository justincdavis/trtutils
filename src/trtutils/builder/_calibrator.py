# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

with contextlib.suppress(ImportError):
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found]

from trtutils.core import memcpy_host_to_device, cuda_malloc

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._batcher import AbstractBatcher


_log = logging.getLogger(__name__)


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """Implements the trt.IInt8EntropyCalibrator2."""

    def __init__(
        self: Self,
        calibration_cache: Path | str | None = None,
    ) -> None:
        """
        Create an EngineCalibrator.

        Parameters
        ----------
        calibration_cache : Path, str, optional
            The path to the calibration cache.

        """
        self._cache_path: Path = Path(calibration_cache).resolve() if calibration_cache is not None else Path("calibration.cache").resolve()
        self._batcher: AbstractBatcher | None = None

    def set_batcher(self: Self, batcher: AbstractBatcher) -> None:
        """Set the batcher."""
        self._batcher = batcher

    def get_batch_size(self: Self) -> int:
        """
        Get the batch size.

        Overrides from trt.IInt8EntropyCalibrator2.

        Returns
        -------
        int
            The batch size

        """
        if self._batcher:
            return self._batcher.batch_size
        return 1

    def get_batch(self: Self) -> list[int] | None:
        """
        Get the next batch of data.

        Overrides from trt.IInt8EntropyCalibrator2.
    
        Returns
        -------
        list[int]
            GPU-Memory pointers of the next batch

        """
        # if we dont have an image batcher, dont handle calibration
        if self._batcher is None:
            return None

        # if we do load the image
        batch = self._batcher.get_next_batch()
        if batch is None:
            return None

        # # TODO: need to fill out remainder
        # # STEP 1: make GPU memory allocation for batch
        # # STEP 2: copy the host side to device
        # # STEP 3: return the GPU pointers

        ptr = cuda_malloc(batch.nbytes)
        memcpy_host_to_device(batch.ctypes.data, ptr)
        return [ptr]

    def read_calibration_cache(self: Self) -> bytes | None:
        """
        Read the calibration cache file if it exists.

        Overrides from trt.IInt8EntropyCalibrator2.

        Returns
        -------
        bytes | None
            The calibration cache contents if it exists

        """
        if self._calib_cache is None:
            return None
        if not self._calib_cache.exists():
            return None

        with self._calib_cache.open("rb") as f:
            _log.debug(f"Reading calibration cache file: {self._calib_cache}")
            return f.read()

    def write_calibration_cache(self: Self, cache: bytes) -> None:
        """
        Write the calibration date to the calibration cache file.

        Overrides from trt.IInt8EntropyCalibrator2.

        Parameters
        ----------
        cache : bytes
            The calibration data generated.

        """
        if self._calib_cache is None:
            return

        with self._calib_cache.open("wb") as f:
            _log.debug(f"Writing calibration cache file: {self._calib_cache}")
            f.write(cache)
