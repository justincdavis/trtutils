# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""An engine built with embed_sst matches a normal YOLOv10 engine on the same image."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import pytest

from tests.conftest import DATA_DIR
from trtutils.models import YOLOv10

if TYPE_CHECKING:
    from pathlib import Path

ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"
HORSE = DATA_DIR / "horse.jpg"


@pytest.mark.slow
@pytest.mark.skipif(not ONNX.exists(), reason="yolov10n onnx is not in data/")
def test_embedded_sst_matches_the_preprocessor(tmp_path: Path) -> None:
    """Letterbox stays outside; scale, swap, and transpose inside the engine give the same boxes."""
    plain = tmp_path / "plain.engine"
    embedded = tmp_path / "embedded.engine"
    YOLOv10.build(ONNX, plain, imgsz=640, opt_level=1)
    YOLOv10.build(ONNX, embedded, imgsz=640, opt_level=1, embed_sst=True)

    image = cv2.resize(cv2.imread(str(HORSE)), (1920, 1080))
    reference = YOLOv10(plain, preprocessor="cuda", warmup_iterations=1).end2end(image)
    got = YOLOv10(embedded, preprocessor="cuda", warmup_iterations=1).end2end(image)
    assert len(reference) == len(got)
    assert len(reference) > 0
    for det, alt in zip(reference, got):
        assert det[2] == alt[2]
        assert abs(det[1] - alt[1]) < 1e-3
        assert all(abs(a - b) <= 2 for a, b in zip(det[0], alt[0]))
