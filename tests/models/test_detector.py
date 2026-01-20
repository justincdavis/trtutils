# Copyright (c) 2024-2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: FBT001
from __future__ import annotations

import pytest

from .common import (
    DETECTOR_CONFIG,
    DLA_ENGINES,
    GPU_ENGINES,
    detector_results,
    detector_run,
    detector_run_in_thread,
    detector_run_multiple,
    detector_run_multiple_threads,
    detector_swapping_preproc_results,
)

# Get all model IDs from config (all are strings now)
DETECTOR_MODELS = list(DETECTOR_CONFIG.keys())
PREPROCESSORS = ["cpu", "cuda", "trt"]


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
@pytest.mark.parametrize("use_dla", [False, True])
def test_detector_run(model_id: str, preprocessor: str, use_dla: bool) -> None:
    """Test detector engine runs with different preprocessors."""
    detector_run(model_id, preprocessor=preprocessor, use_dla=use_dla)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
@pytest.mark.parametrize("use_dla", [False, True])
def test_detector_run_in_thread(model_id: str, preprocessor: str, use_dla: bool) -> None:
    """Test detector engine runs in a thread with different preprocessors."""
    detector_run_in_thread(model_id, preprocessor=preprocessor, use_dla=use_dla)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
@pytest.mark.parametrize("use_dla", [False, True])
def test_detector_run_multiple(model_id: str, preprocessor: str, use_dla: bool) -> None:
    """Test multiple detector engines run with different preprocessors."""
    count = DLA_ENGINES if use_dla else GPU_ENGINES
    detector_run_multiple(model_id, preprocessor=preprocessor, count=count, use_dla=use_dla)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
@pytest.mark.parametrize("use_dla", [False, True])
def test_detector_run_multiple_threads(model_id: str, preprocessor: str, use_dla: bool) -> None:
    """Test multiple detector engines run across multiple threads with different preprocessors."""
    count = DLA_ENGINES if use_dla else GPU_ENGINES
    detector_run_multiple_threads(model_id, preprocessor=preprocessor, count=count, use_dla=use_dla)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
@pytest.mark.parametrize("use_dla", [False, True])
def test_detector_results(model_id: str, preprocessor: str, use_dla: bool) -> None:
    """Test detector engine produces valid results with different preprocessors."""
    detector_results(model_id, preprocessor=preprocessor, use_dla=use_dla)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("use_dla", [False, True])
def test_detector_swapping_preproc_results(model_id: str, use_dla: bool) -> None:
    """Test swapping the preprocessing method at runtime and check results."""
    detector_swapping_preproc_results(model_id, use_dla=use_dla)
