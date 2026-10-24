# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/models/ocr/_ppocr.py -- PP-OCR detection, recognition and pipeline."""

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
    # hershey fonts space words tightly, so a space may or may not be decoded
    return text.replace(" ", "")


def _render(lines: list[str], width: int = 900) -> np.ndarray:
    """White image with one line of black text per entry."""
    img = np.full((70 * len(lines) + 40, width, 3), 255, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (30, 70 + 70 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return img


def _rows(boxes: list[tuple[np.ndarray, float]]) -> int:
    """Number of distinct rendered rows the boxes fall on (a line may be split into words)."""
    return len({int(box[:, 1].mean() // 70) for box, _score in boxes})


def _line(text: str) -> np.ndarray:
    """Tightly cropped single text line, as the detector would hand to the recognizer."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    img = np.full((h + base + 16, w + 16, 3), 255, np.uint8)
    cv2.putText(img, text, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return img


def _blank(height: int, width: int) -> np.ndarray:
    return np.full((height, width, 3), 255, np.uint8)


@pytest.fixture(scope="module")
def det_engine(build_model_engine):
    """Dynamic-shape PP-OCR detection engine."""
    return build_model_engine(PPOCRDetector, DET_ONNX)


@pytest.fixture(scope="module")
def small_det_engine(det_engine):  # det_engine skips when the onnx is missing
    """Detection engine whose profile tops out at batch 2 and 320x320."""
    path = ENGINES_DIR / f"ppocrv4_det_max320_b2_PPOCRDetector_{trt.__version__}.engine"
    if not path.exists():
        PPOCRDetector.build(DET_ONNX, path, imgsz=320, batch_size=2, max_imgsz=320, opt_level=1)
    return path


@pytest.fixture(scope="module")
def rec_engine(build_model_engine):
    """Dynamic-shape PP-OCR recognition engine."""
    return build_model_engine(PPOCRRecognizer, REC_ONNX)


@pytest.mark.parametrize(
    ("engine", "preprocessor", "image", "rows"),
    [
        pytest.param("det_engine", "cpu", _render(LINES), 3, id="lines"),
        pytest.param("det_engine", "trt", _render(LINES), 3, id="lines-trt-falls-back-to-cpu"),
        pytest.param("det_engine", "cpu", _render(LINES[:1], 500), 1, id="one-line-narrow"),
        pytest.param("det_engine", "cpu", _render(LINES[:2], 1400), 2, id="two-lines-wide"),
        pytest.param("det_engine", "cpu", _blank(15, 15), 0, id="tiny"),
        pytest.param("det_engine", "cpu", _blank(30, 4000), 0, id="thin-strip"),
        pytest.param("small_det_engine", "cpu", _render(LINES, 1400), 3, id="limited-profile"),
    ],
)
def test_text_detector_end2end(request, engine, preprocessor, image, rows) -> None:
    """run(), get_text_boxes() and end2end() agree and box every line in reading order inside the image."""
    model = PPOCRDetector(request.getfixturevalue(engine), preprocessor=preprocessor)
    height, width = image.shape[:2]
    max_w, max_h = model.input_shape

    tensor, ratios, padding = model.preprocess(image)
    assert tensor.shape[2] <= max_h
    assert tensor.shape[3] <= max_w
    raw = model.run(image, postprocess=False)
    assert raw[0].shape[2:] == tensor.shape[2:]

    postprocessed = model.run(image)
    boxes, scores = postprocessed
    assert boxes.shape == (len(scores), 4, 2)
    via_run = model.get_text_boxes(postprocessed)
    via_e2e = model.end2end(image)
    via_preprocessed = model.get_text_boxes(
        model.run(tensor, ratios[0], padding[0], preprocessed=True)
    )
    for other in (via_e2e, via_preprocessed):
        assert len(other) == len(via_run)
        for (box, score), (ref_box, ref_score) in zip(other, via_run):
            np.testing.assert_array_equal(box, ref_box)
            assert score == ref_score

    assert _rows(via_run) == rows
    assert all(b[0][0, 1] - a[0][0, 1] > -10 for a, b in zip(via_run, via_run[1:]))
    for box, score in via_run:
        assert box.shape == (4, 2)
        assert np.all(box >= 0)
        assert np.all(box[:, 0] <= width)
        assert np.all(box[:, 1] <= height)
        assert 0.6 <= score <= 1.0
    assert model.end2end(image, box_thresh=1.0) == []


def test_text_detector_batch(small_det_engine) -> None:
    """A batch of differently sized images is padded together and matches per-image runs."""
    model = PPOCRDetector(small_det_engine)
    batch = [_render(LINES[:1], 500), _render(LINES, 900)]
    outputs = model.end2end(batch)
    assert len(outputs) == len(batch)
    for image, dets in zip(batch, outputs):
        ref = model.end2end(image)
        assert len(dets) == len(ref)
        # padding changes what the network sees near the edges, a couple of map pixels here
        for (box, _score), (ref_box, _ref_score) in zip(dets, ref):
            np.testing.assert_allclose(box, ref_box, atol=4)


def test_text_recognizer_end2end(rec_engine) -> None:
    """run(), get_texts() and end2end() agree, and more lines than the batch size decode in input order."""
    model = PPOCRRecognizer(rec_engine)
    crops = [_line(t) for t in LINES]

    raw = model.run(crops, postprocess=False)
    assert raw[0].shape[0] == len(crops)
    assert raw[0].shape[2] == len(model.characters)
    via_run = model.get_texts(model.run(crops))
    assert via_run == model.end2end(crops)
    assert [_words(t) for t, _ in via_run] == [_words(t) for t in LINES]

    many = crops * 3
    assert len(many) > model.batch_size
    assert [_words(t) for t, _ in model.end2end(many)] == [_words(t) for t in LINES * 3]

    text, score = model.end2end(crops[0])
    assert (text, score) == model.get_texts(model.run(crops[0]))
    assert _words(text) == _words(LINES[0])
    assert 0.9 < score <= 1.0


@pytest.mark.parametrize(
    ("character_dict", "match"),
    [
        pytest.param(list("abc"), "classes", id="mismatch"),
        pytest.param(None, "character_dict", id="missing"),
    ],
)
def test_text_recognizer_bad_dict(rec_engine, character_dict, match) -> None:
    """A missing character dictionary or one that does not match the engine's classes raises."""
    with pytest.raises(ValueError, match=match):
        PPOCRRecognizer(rec_engine, character_dict=character_dict)


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        pytest.param(_render(LINES), "".join(LINES), id="lines"),
        pytest.param(_blank(200, 300), "", id="no-text"),
    ],
)
def test_ppocr_pipeline(det_engine, rec_engine, image, expected) -> None:
    """The pipeline reads every line in order, and single, batch and Buffer inputs agree."""
    ocr = PPOCR(det_engine, rec_engine)
    results = ocr(image)
    assert _words("".join(text for _, text, _ in results)) == _words(expected)
    for box, _, score in results:
        assert box.shape == (4, 2)
        assert score >= 0.5
    assert ocr.end2end(image, text_score=1.1) == []

    texts = [t for _, t, _ in results]
    batch = ocr.end2end([image, image])
    assert [[t for _, t, _ in res] for res in batch] == [texts, texts]
    for buf in (Buffer.wrap(image), Buffer.from_array(image, MemoryLocation.DEVICE)):
        assert [t for _, t, _ in ocr(buf)] == texts
        for (box, _), (ref_box, _) in zip(ocr.detector.end2end(buf), ocr.detector.end2end(image)):
            np.testing.assert_array_equal(box, ref_box)


def test_ppocr_recognizer_buffer_input(det_engine, rec_engine) -> None:
    """Host and device Buffer text lines give the same texts as numpy arrays."""
    ocr = PPOCR(det_engine, rec_engine)
    line = _line(LINES[1])
    expected = ocr.recognizer.end2end([line, line])
    for buf in (Buffer.wrap(line), Buffer.from_array(line, MemoryLocation.DEVICE)):
        assert ocr.recognizer.end2end([buf, line]) == expected
