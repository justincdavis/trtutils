# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/models/ocr/_ppocr.py -- PP-OCR detection, recognition and pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tests._gpu_fixtures import engine_cache_path
from tests.conftest import DATA_DIR
from trtutils.core import Buffer, MemoryLocation
from trtutils.image._schema import TextDetectorOutputSchema, TextRecognizerOutputSchema
from trtutils.image.preprocessors import crop_text_region
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
    path = engine_cache_path(f"{DET_ONNX.stem}_PPOCRDetector_img320_max320", 2)
    if not path.exists():
        PPOCRDetector.build(DET_ONNX, path, imgsz=320, batch_size=2, max_imgsz=320, opt_level=1)
    return path


@pytest.fixture(scope="module")
def rec_engine(build_model_engine):
    """Dynamic-shape PP-OCR recognition engine, batch 3 so the three LINES crops fit one run."""
    return build_model_engine(PPOCRRecognizer, REC_ONNX, batch_size=len(LINES))


@pytest.mark.parametrize(
    ("engine", "image", "rows"),
    [
        pytest.param("det_engine", _render(LINES), 3, id="lines"),
        pytest.param("det_engine", _render(LINES[:1], 500), 1, id="one-line-narrow"),
        pytest.param("det_engine", _render(LINES[:2], 1400), 2, id="two-lines-wide"),
        pytest.param("det_engine", _blank(15, 15), 0, id="tiny"),
        pytest.param("det_engine", _blank(30, 4000), 0, id="thin-strip"),
        pytest.param("small_det_engine", _render(LINES, 1400), 3, id="limited-profile"),
    ],
)
def test_text_detector_end2end(request, engine, image, rows) -> None:
    """run(), get_text_boxes() and end2end() agree and box every line in reading order inside the image."""
    model = PPOCRDetector(request.getfixturevalue(engine))
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
    assert model.end2end(image, 1.0) == []


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
    """run(), get_texts() and end2end() agree on arrays and Buffers; extra lines keep input order."""
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

    [(text, score)] = model.end2end(crops[0])
    assert [(text, score)] == model.get_texts(model.run(crops[0]))
    assert _words(text) == _words(LINES[0])
    assert 0.9 < score <= 1.0

    for buf in (Buffer.wrap(crops[1]), Buffer.from_array(crops[1], MemoryLocation.DEVICE)):
        assert model.end2end([buf, crops[1]]) == model.end2end([crops[1], crops[1]])


@pytest.mark.parametrize(
    ("model_cls", "engine", "name"),
    [
        pytest.param(PPOCRDetector, "det_engine", "TextDetector", id="detector"),
        pytest.param(PPOCRRecognizer, "rec_engine", "TextRecognizer", id="recognizer"),
    ],
)
@pytest.mark.parametrize("preprocessor", ["cuda", "trt"])
def test_ocr_rejects_gpu_preprocessor(request, model_cls, engine, name, preprocessor) -> None:
    """Requesting a cuda or trt preprocessor raises, at construction and per call."""
    path = request.getfixturevalue(engine)
    match = f"{name} only supports cpu preprocessing"
    with pytest.raises(ValueError, match=match):
        model_cls(path, preprocessor=preprocessor, warmup=False)
    model = model_cls(path, warmup=False)
    with pytest.raises(ValueError, match=match):
        model.preprocess(_line(LINES[0]), method=preprocessor)


@pytest.mark.parametrize(
    ("model_cls", "engine", "override", "expected"),
    [
        pytest.param(
            PPOCRDetector, "det_engine", None, TextDetectorOutputSchema.PADDLE, id="det-auto"
        ),
        pytest.param(
            PPOCRDetector, "det_engine", "GENERIC", TextDetectorOutputSchema.GENERIC, id="det-str"
        ),
        pytest.param(
            PPOCRRecognizer, "rec_engine", None, TextRecognizerOutputSchema.PADDLE, id="rec-auto"
        ),
        pytest.param(
            PPOCRRecognizer,
            "rec_engine",
            TextRecognizerOutputSchema.GENERIC,
            TextRecognizerOutputSchema.GENERIC,
            id="rec-enum",
        ),
    ],
)
def test_ocr_output_schema(request, model_cls, engine, override, expected) -> None:
    """PP-OCRv4 engines resolve to PADDLE, and the wrappers pass enum and string overrides on."""
    model = model_cls(request.getfixturevalue(engine), output_schema=override, warmup=False)
    assert model.output_schema is expected


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
    assert ocr.end2end(image, 1.1) == []
    assert ocr.end2end(image, box_conf_thres=1.0) == []

    texts = [t for _, t, _ in results]
    batch = ocr.end2end([image, image])
    assert [[t for _, t, _ in res] for res in batch] == [texts, texts]
    for buf in (Buffer.wrap(image), Buffer.from_array(image, MemoryLocation.DEVICE)):
        assert [t for _, t, _ in ocr(buf)] == texts
        for (box, _), (ref_box, _) in zip(ocr.detector.end2end(buf), ocr.detector.end2end(image)):
            np.testing.assert_array_equal(box, ref_box)


def _page(*, rotate: bool = False) -> np.ndarray:
    """A 1080p page of 14 text lines (more crops than a recognizer batch), optionally turned upright."""
    image = np.full((1080, 1920, 3), 255, np.uint8)
    for i in range(14):
        line = f"Line {i + 1}: the quick brown fox jumps over the lazy dog {1000 + 37 * i}"
        cv2.putText(image, line, (60, 70 + 72 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) if rotate else image


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(_render(LINES), id="lines"),
        pytest.param(_render(LINES[:1], 1400), id="one-wide-line-padded"),
        pytest.param(_page(), id="page"),
        pytest.param(_page(rotate=True), id="page-rotated"),
    ],
)
def test_ocr_cuda_postprocessor_matches_cpu(det_engine, rec_engine, image) -> None:
    """postprocessor="cuda" (GPU preprocessing, crops and decode) gives the CPU result exactly."""
    cpu = PPOCR(det_engine, rec_engine, postprocessor="cpu")
    cuda = PPOCR(det_engine, rec_engine, postprocessor="cuda")
    assert cpu._gpu is None
    assert cuda._gpu is not None

    ref = cpu(image)
    assert len(ref) > 0
    for results in (cuda(image), cuda.end2end(image), cuda([image, image])[1]):
        assert len(results) == len(ref)
        for (box, text, score), (ref_box, ref_text, ref_score) in zip(results, ref):
            np.testing.assert_array_equal(box, ref_box)
            assert (text, score) == (ref_text, ref_score)

    # the detector's boxes through the GPU crops equal the CPU crops, texts and scores exactly
    boxes = np.stack([box for box, _ in cpu.detector.end2end(image)])
    frame = Buffer.from_array(image, MemoryLocation.DEVICE)
    want = cpu.recognizer.end2end([crop_text_region(image, box) for box in boxes])
    assert cuda.recognizer.end2end_regions(frame, boxes) == want


def test_ocr_cuda_recognizer_and_empty(det_engine, rec_engine) -> None:
    """The CUDA recognizer decodes crops as the CPU one, and an image without text gives nothing."""
    cpu = PPOCR(det_engine, rec_engine, postprocessor="cpu")
    cuda = PPOCR(det_engine, rec_engine, postprocessor="cuda")
    crops = [_line(t) for t in LINES]
    for got, want in zip(cuda.recognizer.run(crops), cpu.recognizer.run(crops)):
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g, w)
    assert cuda.detector.end2end(_blank(30, 4000)) == []
    assert cuda(_blank(200, 300)) == []


def test_ocr_regions_need_cuda(det_engine, rec_engine) -> None:
    """The CPU recognizer has no GPU crops."""
    recognizer = PPOCR(det_engine, rec_engine, postprocessor="cpu").recognizer
    frame = Buffer.from_array(_blank(40, 40), MemoryLocation.DEVICE)
    with pytest.raises(RuntimeError, match="postprocessor"):
        recognizer.end2end_regions(frame, np.zeros((1, 4, 2)))
