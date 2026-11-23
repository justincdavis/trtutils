# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="import-untyped"
from __future__ import annotations

import contextlib
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from trtutils._log import LOG
from trtutils.compat._libs import trt
from trtutils.core._memory import cuda_free, cuda_malloc, memcpy_host_to_device

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from ._batcher import AbstractBatcher


class _Calibrator:
    """Feeds calibration batches to TensorRT and keeps the calibration cache."""

    def __init__(
        self: Self,
        calibration_cache: Path | str | None = None,
    ) -> None:
        """
        Create a calibrator.

        Parameters
        ----------
        calibration_cache : Path, str, optional
            The path to the calibration cache.

        """
        super().__init__()

        self._temp_cache = calibration_cache is None
        # one reused device allocation (pointer, size) per network input
        self._buffers: dict[str, tuple[int, int]] = {}
        self._error: Exception | None = None
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
        self._input_shapes: dict[str, tuple[int, ...]] = {}
        self._warned_dtype: set[str] = set()

    @property
    def error(self: Self) -> Exception | None:
        """
        The error raised by ``get_batch``, if any.

        TensorRT only prints an error raised while it calibrates and treats it as
        the end of the data, so ``build_engine`` raises it after the build.

        Returns
        -------
        Exception | None
            The error, or None if every batch was read.

        """
        return self._error

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

    def set_input_shapes(self: Self, shapes: Mapping[str, Sequence[int]]) -> None:
        """
        Set the shape TensorRT calibrates each network input at, by input name.

        TensorRT reads a whole input of this shape from each calibration buffer,
        so ``get_batch`` rejects a batch with fewer values, which TensorRT would
        read past. ``build_engine`` sets this to each input's static shape, or to
        the opt shape given in ``shapes`` for an input with dynamic dimensions
        (TensorRT calibrates at the first optimization profile).

        Parameters
        ----------
        shapes : Mapping[str, Sequence[int]]
            The calibration shape of each network input, keyed by input name.

        """
        self._input_shapes = {name: tuple(int(d) for d in shape) for name, shape in shapes.items()}

    def get_batch_size(self: Self) -> int:
        """
        Get the batch size.

        Overrides from the TensorRT calibrator.

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

        Overrides from the TensorRT calibrator. The batcher may return one
        array for a single-input network, a list or tuple of arrays in network
        input order, or a dict of arrays keyed by input name. Each array is
        copied to its own reused device buffer in the dtype TensorRT reads that
        input as (see ``set_input_dtypes``), cast to it first if needed: float32
        for an int8 input, the input's own dtype otherwise.

        Parameters
        ----------
        names : list[str]
            The network inputs TensorRT wants a batch for, in network input order.

        Returns
        -------
        list[int] | None
            GPU-Memory pointers of the next batch, one per name in ``names``,
            or None when there is no more data.

        Raises
        ------
        ValueError
            If the batch does not have one array per network input, a dict batch
            is not keyed by the network input names, or an array has fewer values
            than TensorRT reads for its input (see ``set_input_shapes``).
        TypeError
            If the batch is not an array, a list or tuple, or a dict.

        """
        try:
            return self._get_batch(names)
        except Exception as err:
            # tensorrt swallows errors raised here, build_engine raises it after the build
            self._error = err
            raise

    def _get_batch(self: Self, names: list[str]) -> list[int] | None:
        # if we dont have a batcher, dont handle calibration
        if self._batcher is None:
            return None
        batch = self._batcher.get_next_batch()
        if batch is None:
            return None

        ptrs: list[int] = []
        for name, given in zip(names, _match_inputs(batch, names)):
            # tensorrt reads a whole input from the buffer, past the end of a smaller batch
            shape = self._input_shapes.get(name)
            if shape is not None and given.size < math.prod(shape):
                err_msg = (
                    f"The calibration batch for network input '{name}' has shape {given.shape}"
                    f" ({given.size} values), but TensorRT reads {math.prod(shape)} values for"
                    f" its calibration shape {shape}."
                )
                raise ValueError(err_msg)

            # tensorrt reads the buffer as raw bytes of the bound dtype, so a float32 batch
            # for a float16 input would calibrate on garbage (inf scales, broken engine)
            array = given
            expected = self._batch_dtypes.get(name)
            if expected is not None and given.dtype != expected:
                if name not in self._warned_dtype:
                    LOG.warning(
                        f"TensorRT reads calibration batches for network input '{name}'"
                        f" as {expected}, casting {given.dtype} batches to {expected}."
                    )
                    self._warned_dtype.add(name)
                array = given.astype(expected)

            # reuse one device allocation per input, growing only if a batch is larger
            ptr, size = self._buffers.get(name, (0, 0))
            if array.nbytes > size:
                if ptr:
                    cuda_free(ptr)
                ptr, size = cuda_malloc(array.nbytes), array.nbytes
                self._buffers[name] = (ptr, size)
            memcpy_host_to_device(ptr, array)
            ptrs.append(ptr)
        return ptrs

    def _free(self: Self) -> None:
        for ptr, _ in self._buffers.values():
            cuda_free(ptr)
        self._buffers = {}

    def close(self: Self) -> None:
        """Free the device buffers and delete a temporary cache file."""
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

        Overrides from the TensorRT calibrator.

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

        Overrides from the TensorRT calibrator.

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


class EngineCalibrator(_Calibrator, trt.IInt8EntropyCalibrator2):
    """
    Entropy calibration (``trt.IInt8EntropyCalibrator2``), TensorRT's default.

    Each tensor's range minimizes the information lost to quantization, so rare large
    values may be clipped. The cache starts with ``EntropyCalibration2``.
    """


class MinMaxCalibrator(_Calibrator, trt.IInt8MinMaxCalibrator):
    """
    Min-max calibration (``trt.IInt8MinMaxCalibrator``).

    Each tensor's range covers every value seen while calibrating, so nothing is
    clipped. Suits tensors whose large values matter, such as a detection head that
    concatenates box regression with scores in [0, 1], or a 0-255 image input.
    The cache starts with ``MinMaxCalibration``.
    """


def _match_inputs(
    batch: np.ndarray | Sequence[np.ndarray] | Mapping[str, np.ndarray],
    names: list[str],
) -> list[np.ndarray]:
    """Order the arrays of a batch like ``names``, checking there is one per input."""
    if isinstance(batch, np.ndarray):
        if len(names) != 1:
            err_msg = (
                f"The calibration batcher returned one array, but the network has {len(names)}"
                f" inputs {names}. Return a list of arrays in input order, or a dict of arrays"
                " keyed by input name."
            )
            raise ValueError(err_msg)
        return [batch]
    if isinstance(batch, Mapping):
        missing = [name for name in names if name not in batch]
        unknown = [str(key) for key in batch if key not in names]
        if missing or unknown:
            err_msg = (
                f"Calibration batch keys {list(batch)} do not match the network inputs {names}:"
                f" missing {missing}, unknown {unknown}."
            )
            raise ValueError(err_msg)
        return [batch[name] for name in names]
    if isinstance(batch, (list, tuple)):
        if len(batch) != len(names):
            err_msg = (
                f"The calibration batch holds {len(batch)} array(s), but the network has"
                f" {len(names)} inputs {names}."
            )
            raise ValueError(err_msg)
        return list(batch)
    err_msg = (
        "A calibration batch must be a numpy array, a list or tuple of arrays, or a dict of"
        f" arrays keyed by input name, got {type(batch).__name__}."
    )
    raise TypeError(err_msg)
