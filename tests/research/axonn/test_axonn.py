# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for axonn types and public API."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# add src to path to import without going through main trtutils package
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from trtutils.research.axonn._types import (  # noqa: E402
    LayerCost,
    Processor,
    Schedule,
)

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


class TestLayerCost:
    """Tests for LayerCost dataclass."""

    def test_create_gpu_only(self) -> None:
        cost = LayerCost(gpu_time_ms=1.0, gpu_energy_mj=0.5)
        assert cost.gpu_time_ms == 1.0
        assert cost.gpu_energy_mj == 0.5
        assert cost.dla_time_ms is None
        assert cost.dla_energy_mj is None

    def test_create_with_dla(self) -> None:
        cost = LayerCost(
            gpu_time_ms=1.0,
            gpu_energy_mj=0.5,
            dla_time_ms=1.5,
            dla_energy_mj=0.3,
        )
        assert cost.dla_time_ms == 1.5
        assert cost.dla_energy_mj == 0.3


class TestSchedule:
    """Tests for Schedule dataclass."""

    def test_empty_schedule(self) -> None:
        schedule = Schedule()
        assert len(schedule.assignments) == 0
        assert schedule.num_transitions == 0

    def test_num_transitions(self) -> None:
        schedule = Schedule(assignments={0: Processor.GPU, 1: Processor.DLA, 2: Processor.GPU})
        assert schedule.num_transitions == 2

    def test_no_transitions(self) -> None:
        schedule = Schedule(assignments={0: Processor.GPU, 1: Processor.GPU, 2: Processor.GPU})
        assert schedule.num_transitions == 0

    def test_single_layer(self) -> None:
        schedule = Schedule(assignments={0: Processor.GPU})
        assert schedule.num_transitions == 0


class TestPublicImport:
    """Test that the public API is importable."""

    def test_build_engine_importable(self) -> None:
        from trtutils.research.axonn import build_engine

        assert callable(build_engine)


@pytest.mark.jetson
class TestBuildEngine:
    """Integration tests for build_engine (requires Jetson hardware)."""

    def test_build_engine_yolov10n(self) -> None:
        from trtutils.builder import SyntheticBatcher
        from trtutils.research.axonn import build_engine

        onnx_path = DATA_DIR / "yolov10" / "yolov10n_640.onnx"
        if not onnx_path.exists():
            pytest.skip(f"ONNX model not found: {onnx_path}")

        batcher = SyntheticBatcher(
            shape=(640, 640, 3),
            dtype=np.dtype(np.float32),
            batch_size=8,
            num_batches=10,
            data_range=(0.0, 1.0),
            order="NCHW",
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "axonn_test.engine"
            time_ms, energy_mj, transitions, gpu_layers, dla_layers = build_engine(
                onnx=onnx_path,
                output=output,
                calibration_batcher=batcher,
                energy_ratio=0.8,
                max_transitions=1,
                profile_iterations=100,
                warmup_iterations=10,
                verbose=True,
            )

            assert output.exists()
            assert time_ms > 0
            assert energy_mj > 0
            assert transitions >= 0
            assert gpu_layers + dla_layers > 0
