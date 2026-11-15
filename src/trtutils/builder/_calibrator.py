# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG
from trtutils.compat._libs import trt
from trtutils.core._memory import cuda_free, cuda_malloc, memcpy_host_to_device

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

    from ._batcher import AbstractBatcher


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
        super().__init__()

        self._temp_cache = calibration_cache is None
        self._ptr: int | None = None
        self._ptr_size = 0
        if calibration_cache is not None:
            self._cache_path: Path = Path(calibration_cache).resolve()
        else:
            with tempfile.NamedTemporaryFile(suffix=".cache", delete=False) as tmp:
                self._cache_path = Path(tmp.name)
            LOG.warning(
                "No calibration cache path provided, using a temporary file."
                " Calibration data will not persist across runs."
            )
        self._batcher: AbstractBatcher | None = None
        self._batch_dtypes: dict[str, np.dtype] = {}
        self._warned_dtype = False

    def set_batcher(self: Self, batcher: AbstractBatcher) -> None:
        """Set the batcher."""
        self._batcher = batcher

    def set_input_dtypes(self: Self, dtypes: Mapping[str, np.dtype | type[np.generic]]) -> None:
        """
        Set the dtype of each network input, by input name.

        TensorRT reads a calibration batch as raw bytes of the dtype it binds the
        input as, so ``get_batch`` casts batches of any other dtype to it. That is
        the input's own dtype, except for int8: calibration runs the network
        unquantized, so TensorRT binds an int8 input as float32, and batches for it
        are cast to float32 (integer values are kept). On TensorRT 8.6, float16,
        int32, uint8 and bool inputs are read in their own dtype. ``build_engine``
        sets this from the network after hooks and tensor formats are applied.

        Parameters
        ----------
        dtypes : Mapping[str, np.dtype | type[np.generic]]
            The numpy dtype of each network input, keyed by input name.

        """
        self._batch_dtypes = {}
        for name, dtype in dtypes.items():
            np_dtype = np.dtype(dtype)
            # int8 inputs are bound as float32 while calibrating; int8 bytes there
            # are read 4x past the batch (illegal memory access or garbage scales)
            self._batch_dtypes[name] = np.dtype(np.float32) if np_dtype == np.int8 else np_dtype

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

    def get_batch(self: Self, names: list[str]) -> list[int] | None:
        """
        Get the next batch of data.

        Overrides from trt.IInt8EntropyCalibrator2. A batch is copied in the dtype
        TensorRT reads the input as (see ``set_input_dtypes``) and cast to it first
        if needed: float32 for an int8 input, the input's own dtype otherwise.

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

        # tensorrt reads the buffer as raw bytes of the bound dtype, so a float32 batch
        # for a float16 input would calibrate on garbage (inf scales, broken engine)
        expected = self._batch_dtypes.get(names[0]) if names else None
        if expected is not None and batch.dtype != expected:
            if not self._warned_dtype:
                LOG.warning(
                    f"TensorRT reads calibration batches for network input '{names[0]}'"
                    f" as {expected}, casting {batch.dtype} batches to {expected}."
                )
                self._warned_dtype = True
            batch = batch.astype(expected)

        # reuse one device allocation, growing only if a batch is larger
        if self._ptr is None or batch.nbytes > self._ptr_size:
            self._free()
            self._ptr = cuda_malloc(batch.nbytes)
            self._ptr_size = batch.nbytes
        memcpy_host_to_device(self._ptr, batch)
        return [self._ptr]

    def _free(self: Self) -> None:
        if self._ptr is not None:
            cuda_free(self._ptr)
            self._ptr = None
            self._ptr_size = 0

    def close(self: Self) -> None:
        """Free the device buffer and delete a temporary cache file."""
        self._free()
        if self._temp_cache:
            with contextlib.suppress(OSError):
                self._cache_path.unlink()

    def __del__(self: Self) -> None:
        with contextlib.suppress(Exception):
            self.close()

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
