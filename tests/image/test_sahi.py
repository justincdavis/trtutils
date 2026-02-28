# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc"
"""Tests for SAHI (Slicing Aided Hyper Inference) integration."""

from __future__ import annotations

from pathlib import Path

import pytest

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"


def _sahi_available() -> bool:
    """Check if the trtutils SAHI module is importable."""
    try:
        from trtutils.image.sahi import SAHI  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def yolov10_engine(build_test_engine) -> Path:
    """Build and cache a YOLOv10n engine for the test module."""
    if not YOLOV10_ONNX.exists():
        pytest.skip("yolov10n_640.onnx not available")
    return build_test_engine(YOLOV10_ONNX)


class TestSAHIIntegration:
    """Test SAHI slicing and merging integration."""

    def test_sahi_available(self) -> None:
        """SAHI module can be imported from trtutils.image.sahi."""
        if not _sahi_available():
            pytest.skip("SAHI module not available")
        from trtutils.image.sahi import SAHI

        assert SAHI is not None

    def test_sahi_init_with_detector(self, yolov10_engine: Path) -> None:
        """SAHI can be initialized with a Detector."""
        if not _sahi_available():
            pytest.skip("SAHI module not available")
        from trtutils.image.sahi import SAHI
        from trtutils.models import YOLOv10

        det = YOLOv10(yolov10_engine, warmup=False)
        sahi = SAHI(det)
        assert sahi is not None

    def test_sahi_end2end(self, yolov10_engine: Path, images) -> None:
        """SAHI end2end runs without error and returns detections."""
        horse_image = images["horse"].array
        if not _sahi_available():
            pytest.skip("SAHI module not available")
        from trtutils.image.sahi import SAHI
        from trtutils.models import YOLOv10

        det = YOLOv10(yolov10_engine, warmup=False)
        sahi = SAHI(det)
        detections = sahi.end2end(horse_image)
        assert isinstance(detections, list)
        for d in detections:
            assert len(d) == 3  # (bbox, score, class_id)

    def test_sahi_with_slice_size(self, yolov10_engine: Path, images) -> None:
        """SAHI accepts custom slice_size parameter."""
        horse_image = images["horse"].array
        if not _sahi_available():
            pytest.skip("SAHI module not available")
        from trtutils.image.sahi import SAHI
        from trtutils.models import YOLOv10

        det = YOLOv10(yolov10_engine, warmup=False)
        sahi = SAHI(det, slice_size=(320, 320))
        detections = sahi.end2end(horse_image)
        assert isinstance(detections, list)
