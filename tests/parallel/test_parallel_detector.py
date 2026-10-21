# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/parallel/image/_detector.py -- worker error handling."""

from __future__ import annotations

from queue import Queue
from threading import Event, Thread

import numpy as np
import pytest

from trtutils.parallel.image._detector import EngineInfo, ParallelDetector, _InputPacket

pytestmark = pytest.mark.cpu


class _FakeDetector:
    """Stands in for a Detector: batches marked ``preprocess_method="cpu"`` fail."""

    infer_profile = (0.0, 0.0)

    def __init__(self, **_kwargs: object) -> None:
        pass

    def infer(self, images, method, **_kwargs: object):
        if method == "cpu":
            err_msg = "bad batch"
            raise ValueError(err_msg)
        return [np.full(1, len(images))], None, None


def _parallel_detector(models: int) -> ParallelDetector:
    """Assemble a ParallelDetector whose workers run _FakeDetector models."""
    det = ParallelDetector.__new__(ParallelDetector)
    det._engine_info = [
        EngineInfo(engine_path="fake", detector_class=_FakeDetector) for _ in range(models)
    ]
    for name in (
        "input_range", "preprocessor", "resize_method", "conf_thres", "nms_iou_thres",
        "mean", "std", "backend", "warmup", "pagelocked_mem", "unified_mem", "cuda_graph",
        "extra_nms", "agnostic_nms",
    ):  # fmt: skip
        setattr(det, f"_{name}", None)
    det._tag = "test"
    det._nvtx_tags = {"retrieve": "", "retrieve_model": ""}
    det._warmup_iterations = 0
    det._no_warn = True
    det._verbose = False
    det._stopflag = Event()
    det._iqueues = [Queue() for _ in range(models)]
    det._oqueues = [Queue() for _ in range(models)]
    det._profilers = [(0.0, 0.0)] * models
    det._flags = [Event() for _ in range(models)]
    det._models = [None] * models
    det._threads = [Thread(target=det._run, args=(i,), daemon=True) for i in range(models)]
    for thread in det._threads:
        thread.start()
    return det


def _submit(det: ParallelDetector, modelid: int, method: str | None) -> None:
    packet = _InputPacket(data=[np.zeros(1)], preprocess_method=method, postprocess=False)
    det._iqueues[modelid].put(packet)


@pytest.fixture
def detector():
    """Two fake models with running workers, stopped after the test."""
    det = _parallel_detector(models=2)
    yield det
    det._stopflag.set()
    for thread in det._threads:
        thread.join(timeout=1.0)


def test_worker_error_is_raised_by_retrieve_and_the_worker_survives(detector) -> None:
    """A batch that fails raises in the caller's thread; the next batch still runs."""
    _submit(detector, 0, "cpu")
    _submit(detector, 0, None)
    with pytest.raises(ValueError, match="bad batch"):
        detector.retrieve_model(0)
    outputs, _, _ = detector.retrieve_model(0)
    np.testing.assert_array_equal(outputs[0], [1])


def test_retrieve_takes_every_result_before_raising(detector) -> None:
    """One model failing never leaves another model's result queued for the next call."""
    _submit(detector, 0, "cpu")
    _submit(detector, 1, None)
    with pytest.raises(ValueError, match="bad batch"):
        detector.retrieve()
    for modelid in range(2):
        _submit(detector, modelid, None)
    outputs, _, _ = detector.retrieve()
    assert len(outputs) == 2
    assert all(queue.empty() for queue in detector._oqueues)
