# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the feature extractor wrappers in src/trtutils/models/feature_extractors/."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import pytest

from tests.conftest import DATA_DIR, PREPROCESSORS
from trtutils import Buffer, MemoryLocation
from trtutils._flags import FLAGS
from trtutils.builder import build_engine
from trtutils.image._schema import FeatureExtractorOutputSchema
from trtutils.image.preprocessors import crop_boxes
from trtutils.models import DINOv3
from trtutils.models._hooks import vit_attention_fp32_build_hook

_DINOV3 = DATA_DIR / "dinov3"
_VITS = _DINOV3 / "dinov3_vits16_256.onnx"
_VITS_WIDTH = 384

# (x1, y1, x2, y2) on the 447x288 horse: mixed sizes, float coordinates, clipped edges
_BOXES = np.array(
    [
        [0, 0, 447, 288],
        [50, 40, 250, 260],
        [200.5, 10.2, 440.7, 200.9],
        [-20, 100, 120, 320],
        [300, 150, 460, 300],
        [10, 10, 74, 202],
        [100, 200, 400, 280],
        [150, 50, 182, 82],
        [60, 60, 300, 240],
        [5.5, 120.5, 90.5, 180.5],
    ]
)

# (wrapper, onnx, embedding width, transformer depth)
FEATURE_EXTRACTORS = [
    pytest.param(DINOv3, _DINOV3 / "dinov3_vits16_256.onnx", 384, 12, id="dinov3-vits16"),
    pytest.param(DINOv3, _DINOV3 / "dinov3_vits16plus_256.onnx", 384, 12, id="dinov3-vits16plus"),
    pytest.param(DINOv3, _DINOV3 / "dinov3_vitb16_256.onnx", 768, 12, id="dinov3-vitb16"),
]


def _cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize(("model_cls", "onnx_path", "width", "depth"), FEATURE_EXTRACTORS)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_feature_extractor_end2end(
    build_model_engine, images, model_cls, onnx_path, width, depth, preprocessor, cpu_resize
) -> None:
    """run() + get_*() match end2end(), shapes follow the patch grid, and normalize gives unit norm."""
    del depth
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False)
    assert model.output_schema is FeatureExtractorOutputSchema.EMBEDDING_MAP
    image = images["horse"].array
    grid = model.input_shape[0] // 16

    postprocessed = model.run(image)
    embedding = model.get_embeddings(postprocessed)
    feature_map = model.get_feature_maps(postprocessed)
    np.testing.assert_array_equal(embedding, model.end2end(image))

    assert embedding.shape == (width,)
    assert embedding.dtype == np.float32
    assert feature_map.shape == (width, grid, grid)
    assert feature_map.dtype == np.float32
    assert np.all(np.isfinite(embedding))
    assert np.all(np.isfinite(feature_map))

    normed = model.end2end(image, normalize=True)
    np.testing.assert_allclose(np.linalg.norm(normed), 1.0, rtol=1e-5)
    np.testing.assert_allclose(normed, embedding / np.linalg.norm(embedding), rtol=1e-5, atol=1e-7)
    # the constructor default applies when the call does not override it
    normed_model = model_cls(
        engine, preprocessor=preprocessor, cpu_resize=cpu_resize, normalize=True, warmup=False
    )
    np.testing.assert_allclose(normed_model.end2end(image), normed, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(normed_model.end2end(image, normalize=False), embedding)


@pytest.mark.correctness
@pytest.mark.parametrize(("model_cls", "onnx_path", "width", "depth"), FEATURE_EXTRACTORS)
def test_feature_extractor_similarity(
    build_model_engine, images, model_cls, onnx_path, width, depth
) -> None:
    """The same image embeds to the same vector, different images to clearly different ones."""
    del width, depth
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, warmup=False)
    horse, people = images["horse"].array, images["people"].array

    # each result must survive the next inference, end2end skips the output copies
    first = model.end2end(horse)
    second = model.end2end(horse.copy())
    other = model.end2end(people)
    np.testing.assert_array_equal(first, model.end2end(horse))
    assert _cosine(first, second) > 0.9999
    assert _cosine(first, other) < 0.9


@pytest.mark.correctness
@pytest.mark.parametrize(("model_cls", "onnx_path", "width", "depth"), FEATURE_EXTRACTORS)
def test_feature_extractor_matches_onnxruntime(
    build_model_engine, images, model_cls, onnx_path, width, depth
) -> None:
    """The default build (FP16 where the platform has it) embeds like onnxruntime on the same input."""
    del width, depth
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor="cpu", warmup=False)
    image = images["horse"].array

    tensor, _ratios, _padding = model.preprocess(image, method="cpu")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    reference = session.run(["embedding"], {"input": tensor})[0][0]

    assert _cosine(model.end2end(image), reference) > 0.999


@pytest.mark.skipif(FLAGS.TRT_VERSION < (10, 0), reason="the hook pins nothing before TensorRT 10")
@pytest.mark.parametrize(("model_cls", "onnx_path", "width", "depth"), FEATURE_EXTRACTORS)
def test_attention_hook_matches(model_cls, onnx_path, width, depth) -> None:
    """The fp16 build hook pins the QK matmul and softmax of every block, so it is not a no-op."""
    del width
    if not onnx_path.exists():
        pytest.skip(f"missing {onnx_path}")
    assert vit_attention_fp32_build_hook in model_cls._build_hooks
    overrides = vit_attention_fp32_build_hook(onnx=onnx_path)
    assert len(overrides["layer_precision"]) == 2 * depth


@pytest.fixture(scope="module")
def dynamic_vits_engine(tmp_path_factory):
    """A DINOv3 ViT-S engine with a (1, 4, 8) dynamic batch profile."""
    if not _VITS.exists():
        pytest.skip(f"missing {_VITS}")
    engine_path = tmp_path_factory.mktemp("dinov3") / "dinov3_vits16_256_dynamic.engine"
    shape = (3, 256, 256)
    build_engine(
        _VITS,
        engine_path,
        shapes=[("input", ((1, *shape), (4, *shape), (8, *shape)))],
        optimization_level=1,
    )
    return engine_path


def _assert_rows_match(rows, image, boxes, reference) -> None:
    """Each row embeds like end2end() on a contiguous copy of its crop."""
    assert rows.shape == (len(boxes), _VITS_WIDTH)
    assert rows.dtype == np.float32
    for row, crop in zip(rows, crop_boxes(image, boxes)):
        assert _cosine(row, reference.end2end(np.ascontiguousarray(crop))) > 0.9999


@pytest.mark.parametrize(
    ("count", "chunks"),
    [
        pytest.param(1, [4], id="1-box"),
        pytest.param(4, [4], id="4-boxes"),
        pytest.param(6, [4, 4], id="6-boxes"),
        pytest.param(9, [4, 4, 4], id="9-boxes"),
    ],
)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_end2end_boxes_static_batch(
    build_model_engine, images, preprocessor, cpu_resize, count, chunks
) -> None:
    """A static batch-4 engine gets full chunks, padded at the end, and rows match their crops."""
    kwargs = {"preprocessor": preprocessor, "cpu_resize": cpu_resize, "warmup": False}
    model = DINOv3(build_model_engine(DINOv3, _VITS, batch_size=4), **kwargs)
    reference = DINOv3(build_model_engine(DINOv3, _VITS), **kwargs)
    assert model.is_dynamic_batch is False
    image = images["horse"].array
    boxes = _BOXES[:count]

    with patch.object(model, "end2end", wraps=model.end2end) as spy:
        rows = model.end2end_boxes(image, boxes)
    assert [len(call.args[0]) for call in spy.call_args_list] == chunks
    _assert_rows_match(rows, image, boxes, reference)


@pytest.mark.parametrize(
    ("count", "chunks"),
    [
        pytest.param(3, [3], id="3-boxes"),
        pytest.param(10, [8, 2], id="10-boxes"),
    ],
)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_end2end_boxes_dynamic_batch(
    dynamic_vits_engine, images, preprocessor, cpu_resize, count, chunks
) -> None:
    """A dynamic-batch engine gets short chunks as they are, and rows match their crops."""
    model = DINOv3(
        dynamic_vits_engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False
    )
    assert model.is_dynamic_batch is True
    image = images["horse"].array
    boxes = _BOXES[:count]

    with patch.object(model, "end2end", wraps=model.end2end) as spy:
        rows = model.end2end_boxes(image, boxes)
    assert [len(call.args[0]) for call in spy.call_args_list] == chunks
    _assert_rows_match(rows, image, boxes, model)


def test_end2end_boxes_behaviour(build_model_engine, images) -> None:
    """No boxes skips the engine, normalize gives unit rows, rows survive later calls, Buffer frames work."""
    engine = build_model_engine(DINOv3, _VITS, batch_size=4)
    model = DINOv3(engine, warmup=False)
    image = images["horse"].array

    with patch.object(model, "end2end", wraps=model.end2end) as spy:
        empty = model.end2end_boxes(image, [])
    spy.assert_not_called()
    assert empty.shape == (0, _VITS_WIDTH)
    assert empty.dtype == np.float32

    boxes = [
        (50, 40, 250, 260),
        (10, 10, 74, 202),
        (0, 0, 447, 288),
        (150, 50, 182, 82),
        (5, 5, 40, 40),
    ]
    rows = model.end2end_boxes(image, boxes)
    kept = rows.copy()
    model.end2end_boxes(images["people"].array, [(0, 0, 100, 100)] * 6)
    np.testing.assert_array_equal(rows, kept)

    normed = model.end2end_boxes(image, boxes, normalize=True)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    np.testing.assert_allclose(np.linalg.norm(normed, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(normed, rows / norms, rtol=1e-5, atol=1e-7)
    normed_model = DINOv3(engine, normalize=True, warmup=False)
    np.testing.assert_allclose(
        normed_model.end2end_boxes(image, boxes), normed, rtol=1e-5, atol=1e-7
    )

    device = Buffer.from_array(image, MemoryLocation.DEVICE)
    np.testing.assert_allclose(model.end2end_boxes(device, boxes), rows, rtol=1e-5, atol=1e-5)
