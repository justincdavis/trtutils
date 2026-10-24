# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the PP-OCR wrappers in src/trtutils/models/ocr/."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import tensorrt as trt

from tests.conftest import DATA_DIR, ENGINES_DIR
from trtutils.core import Buffer, MemoryLocation
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


def _rows(boxes: np.ndarray) -> int:
    """Number of distinct rendered rows the boxes fall on (a line may be split into words)."""
    return len(np.unique(boxes[:, :, 1].mean(axis=1) // 70))


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
def small_det_engine(det_engine):  # det_engine skips when the ONNX is missing
    """Detection engine whose profile tops out at 320x320."""
    path = ENGINES_DIR / f"ppocrv4_det_max320_PPOCRDetector_{trt.__version__}.engine"
    if not path.exists():
        PPOCRDetector.build(DET_ONNX, path, imgsz=320, max_imgsz=320, opt_level=1)
    return path


@pytest.fixture(scope="module")
def rec_engine(build_model_engine):
    """Dynamic-shape PP-OCR recognition engine."""
    return build_model_engine(PPOCRRecognizer, REC_ONNX)


def test_detector_finds_lines(det_engine) -> None:
    """Boxes cover every rendered line, in reading order, inside the image."""
    img = _render(LINES)
    boxes, scores = PPOCRDetector(det_engine)(img)
    assert boxes.shape == (len(scores), 4, 2)
    assert _rows(boxes) == len(LINES)
    assert np.all(np.diff(boxes[:, 0, 1]) > -10)
    assert boxes[..., 0].max() <= img.shape[1]
    assert boxes[..., 1].max() <= img.shape[0]


def test_detector_varying_shapes(det_engine) -> None:
    """The dynamic engine handles differently sized inputs back to back."""
    det = PPOCRDetector(det_engine)
    for lines, width in [(LINES[:1], 500), (LINES, 900), (LINES[:2], 1400)]:
        boxes, _ = det(_render(lines, width))
        assert _rows(boxes) == len(lines)


@pytest.mark.parametrize("shape", [(15, 15), (30, 4000)], ids=["tiny", "thin-strip"])
def test_detector_degenerate_shapes(det_engine, shape) -> None:
    """Tiny and very thin images resize to at least 32 px a side and run."""
    boxes, scores = PPOCRDetector(det_engine)(np.full((*shape, 3), 255, np.uint8))
    assert boxes.shape == (len(scores), 4, 2)


def test_detector_fits_limited_profile(small_det_engine) -> None:
    """Images larger than the engine profile are scaled to fit it instead of failing."""
    det = PPOCRDetector(small_det_engine)
    img = _render(LINES, 1400)
    assert max(det.preprocess(img).shape[2:]) <= 320
    boxes, _ = det(img)
    assert _rows(boxes) == len(LINES)
    assert boxes[..., 0].max() <= img.shape[1]


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
    """The full pipeline recognizes every rendered line, in reading order."""
    results = PPOCR(det_engine, rec_engine)(_render(LINES))
    assert _words("".join(text for _, text, _ in results)) == _words("".join(LINES))
    for box, _, score in results:
        assert box.shape == (4, 2)
        assert score >= 0.5


def test_ppocr_buffer_input(det_engine, rec_engine) -> None:
    """Host and device Buffer images give the same results as numpy arrays."""
    img = _render(LINES)
    ocr = PPOCR(det_engine, rec_engine)
    expected = ocr(img)
    expected_boxes, _ = ocr.detector(img)
    for buf in (Buffer.wrap(img), Buffer.from_array(img, MemoryLocation.DEVICE)):
        assert [t for _, t, _ in ocr(buf)] == [t for _, t, _ in expected]
        np.testing.assert_array_equal(ocr.detector(buf)[0], expected_boxes)
        assert ocr.recognizer([buf, img]) == ocr.recognizer([img, img])


def test_ppocr_pipeline_no_text(det_engine, rec_engine) -> None:
    """An image without text returns no results."""
    assert PPOCR(det_engine, rec_engine)(np.full((200, 300, 3), 255, np.uint8)) == []
