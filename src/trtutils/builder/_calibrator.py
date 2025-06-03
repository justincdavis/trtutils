# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

with contextlib.suppress(ImportError):
    import tensorrt as trt

from trtutils._log import LOG
from trtutils.core import cuda_malloc, memcpy_host_to_device

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._batcher import AbstractBatcher


class EngineCalibrator(trt.IInt8EntropyCalibrator2):  # type: ignore[misc]
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
        super().__init__()

        self._cache_path: Path = (
            Path(calibration_cache).resolve()
            if calibration_cache is not None
            else Path("calibration.cache").resolve()
        )
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

    def get_batch(self: Self, names: list[str]) -> list[int] | None:  # noqa: ARG002
        """
        Get the next batch of data.

        Overrides from trt.IInt8EntropyCalibrator2.

        Parameters
        ----------
        names : list[str]
            The list of inputs, if useful to define the batch.

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

        # allocate GPU memory for the batch
        # return the GPU pointer
        ptr = cuda_malloc(batch.nbytes)
        memcpy_host_to_device(ptr, batch)
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
        if self._cache_path is None:
            return None
        if not self._cache_path.exists():
            return None

        with self._cache_path.open("rb") as f:
            LOG.debug(f"Reading calibration cache file: {self._cache_path}")
            data: bytes = f.read()
            return data

    def write_calibration_cache(self: Self, cache: bytes) -> None:
        """
        Write the calibration date to the calibration cache file.

        Overrides from trt.IInt8EntropyCalibrator2.

        Parameters
        ----------
        cache : bytes
            The calibration data generated.

        """
        if self._cache_path is None:
            return

        with self._cache_path.open("wb") as f:
            LOG.debug(f"Writing calibration cache file: {self._cache_path}")
            f.write(cache)
