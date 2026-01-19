# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import pytest

from .common import DETECTOR_CONFIG, detector_pagelocked_perf

# Get all model IDs from config (all are strings now)
DETECTOR_MODELS = list(DETECTOR_CONFIG.keys())


@pytest.mark.parametrize("model_id", DETECTOR_MODELS)
@pytest.mark.parametrize("use_dla", [False, True])
def test_detector_pagelocked_perf(model_id: str, use_dla: bool) -> None:
    """Test the performance of detector models with pagelocked memory."""
    detector_pagelocked_perf(model_id, use_dla=use_dla)
