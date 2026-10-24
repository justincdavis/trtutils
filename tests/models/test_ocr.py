# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the PP-OCR wrappers in src/trtutils/models/ocr/."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tests.conftest import DATA_DIR
from trtutils.models import PPOCR, PPOCRDetector, PPOCRRecognizer

DET_ONNX = DATA_DIR / "ppocr" / "ppocrv4_det_736.onnx"
REC_ONNX = DATA_DIR / "ppocr" / "ppocrv4_rec_48.onnx"
LINES = ["Hello World", "TensorRT inference 2026", "quick brown fox jumps"]


def _words(text: str) -> str:
    # Hershey fonts space words tightly, so a space may or may not be decoded
    return text.replace(" ", "")


def _render(lines: list[str], width: int = 900) -> np.ndarray:
    """White image with one line of black text per entry."""
    img = np.full((70 * len(lines) + 40, width, 3), 255, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (30, 70 + 70 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return img


def _line(text: str) -> np.ndarray:
    """Tightly cropped single text line, as the detector would hand to the recognizer."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    img = np.full((h + base + 16, w + 16, 3), 255, np.uint8)
    cv2.putText(img, text, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return img


@pytest.fixture(scope="module")
def det_engine(build_model_engine):
    """Dynamic-shape PP-OCR detection engine."""
    return build_model_engine(PPOCRDetector, DET_ONNX)


@pytest.fixture(scope="module")
def rec_engine(build_model_engine):
    """Dynamic-shape PP-OCR recognition engine."""
    return build_model_engine(PPOCRRecognizer, REC_ONNX)


def test_detector_finds_lines(det_engine) -> None:
    """One box per rendered line, in reading order, inside the image."""
    img = _render(LINES)
    boxes, scores = PPOCRDetector(det_engine)(img)
    assert boxes.shape == (len(LINES), 4, 2)
    assert scores.shape == (len(LINES),)
    assert np.all(np.diff(boxes[:, 0, 1]) > 0)
    assert boxes[..., 0].max() <= img.shape[1]
    assert boxes[..., 1].max() <= img.shape[0]


def test_detector_varying_shapes(det_engine) -> None:
    """The dynamic engine handles differently sized inputs back to back."""
    det = PPOCRDetector(det_engine)
    for lines, width in [(LINES[:1], 500), (LINES, 900), (LINES[:2], 1400)]:
        boxes, _ = det(_render(lines, width))
        assert len(boxes) == len(lines)


def test_recognizer_lines(rec_engine) -> None:
    """Single lines and batches larger than the engine batch size decode in input order."""
    rec = PPOCRRecognizer(rec_engine)
    crops = [_line(t) for t in LINES * 3]
    assert [_words(t) for t, _ in rec(crops)] == [_words(t) for t in LINES * 3]
    text, score = rec(crops[0])[0]
    assert _words(text) == _words(LINES[0])
    assert 0.9 < score <= 1.0


def test_recognizer_dict_mismatch(rec_engine) -> None:
    """A character dictionary that does not match the engine's classes raises."""
    with pytest.raises(ValueError, match="classes"):
        PPOCRRecognizer(rec_engine, character_dict=list("abc"))


def test_ppocr_pipeline(det_engine, rec_engine) -> None:
    """The full pipeline recognizes every rendered line with its box."""
    results = PPOCR(det_engine, rec_engine)(_render(LINES))
    assert [_words(text) for _, text, _ in results] == [_words(t) for t in LINES]
    for box, _, score in results:
        assert box.shape == (4, 2)
        assert score >= 0.5


def test_ppocr_pipeline_no_text(det_engine, rec_engine) -> None:
    """An image without text returns no results."""
    assert PPOCR(det_engine, rec_engine)(np.full((200, 300, 3), 255, np.uint8)) == []
