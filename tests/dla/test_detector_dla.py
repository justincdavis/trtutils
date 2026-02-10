# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
"""DLA-specific detector tests (Jetson only)."""

from __future__ import annotations

import pytest

from tests.models.common import (
    DETECTOR_CONFIG,
    DLA_ENGINES,
    detector_pagelocked_perf,
    detector_results,
    detector_run,
    detector_run_in_thread,
    detector_run_multiple,
    detector_run_multiple_threads,
    detector_swapping_preproc_results,
)

pytestmark = [pytest.mark.gpu, pytest.mark.dla]

DETECTOR_MODELS = list(DETECTOR_CONFIG.keys())
PREPROCESSORS = ["cpu", "cuda", "trt"]


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_detector_run_dla(model_id: str, preprocessor: str) -> None:
    """Test detector engine runs with DLA."""
    detector_run(model_id, preprocessor=preprocessor, use_dla=True)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_detector_run_in_thread_dla(model_id: str, preprocessor: str) -> None:
    """Test detector engine runs in a thread with DLA."""
    detector_run_in_thread(model_id, preprocessor=preprocessor, use_dla=True)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_detector_run_multiple_dla(model_id: str, preprocessor: str) -> None:
    """Test multiple detector engines run with DLA."""
    detector_run_multiple(model_id, preprocessor=preprocessor, count=DLA_ENGINES, use_dla=True)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_detector_run_multiple_threads_dla(model_id: str, preprocessor: str) -> None:
    """Test multiple detector engines run across multiple threads with DLA."""
    detector_run_multiple_threads(
        model_id, preprocessor=preprocessor, count=DLA_ENGINES, use_dla=True
    )


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("preprocessor", PREPROCESSORS)
def test_detector_results_dla(model_id: str, preprocessor: str) -> None:
    """Test detector engine produces valid results with DLA."""
    detector_results(model_id, preprocessor=preprocessor, use_dla=True)


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
def test_detector_swapping_preproc_results_dla(model_id: str) -> None:
    """Test swapping the preprocessing method at runtime with DLA."""
    detector_swapping_preproc_results(model_id, use_dla=True)


@pytest.mark.performance
@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
def test_detector_pagelocked_perf_dla(model_id: str) -> None:
    """Test the performance of detector models with pagelocked memory on DLA."""
    detector_pagelocked_perf(model_id, use_dla=True)
