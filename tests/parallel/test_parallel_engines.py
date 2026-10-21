# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for QueuedTRTEngine / ParallelTRTEngines -- worker error handling."""

from __future__ import annotations

import numpy as np
import pytest

from trtutils._engine import TRTEngine
from trtutils.parallel import ParallelTRTEngines, QueuedTRTEngine

pytestmark = pytest.mark.cpu


class _FakeEngine(TRTEngine):
    """A TRTEngine stand-in: inputs holding a negative value fail."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, data, **_kwargs: object) -> list[np.ndarray]:
        if data[0][0] < 0:
            err_msg = "bad input"
            raise ValueError(err_msg)
        return [np.asarray(data) * 2]

    def __del__(self) -> None:
        pass


def test_queued_engine_error_is_raised_by_retrieve_and_the_worker_survives() -> None:
    """An input that fails raises in the caller's thread; the next input still runs."""
    queued = QueuedTRTEngine(_FakeEngine("fake"))
    try:
        queued.submit([[-1]])
        queued.submit([[2]])
        with pytest.raises(ValueError, match="bad input"):
            queued.retrieve(timeout=5.0)
        outputs = queued.retrieve(timeout=5.0)
        assert outputs is not None
        np.testing.assert_array_equal(outputs[0], [[4]])
    finally:
        queued.stop()


def test_parallel_engines_take_every_result_before_raising() -> None:
    """One engine failing never leaves another engine's result queued for the next call."""
    engines = ParallelTRTEngines([_FakeEngine("a"), _FakeEngine("b")])
    try:
        engines.submit([[[-1]], [[1]]])
        with pytest.raises(ValueError, match="bad input"):
            engines.retrieve(timeout=5.0)
        engines.submit([[[1]], [[3]]])
        outputs = engines.retrieve(timeout=5.0)
        assert [out[0].tolist() if out is not None else None for out in outputs] == [
            [[2]],
            [[6]],
        ]
    finally:
        engines.stop()
