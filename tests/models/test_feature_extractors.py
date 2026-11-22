# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the feature extractor wrappers in src/trtutils/models/feature_extractors/."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import DATA_DIR, PREPROCESSORS
from trtutils.image._schema import FeatureExtractorOutputSchema
from trtutils.models import DINOv3
from trtutils.models._hooks import vit_attention_fp32_build_hook

_DINOV3 = DATA_DIR / "dinov3"

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


@pytest.mark.parametrize(("model_cls", "onnx_path", "width", "depth"), FEATURE_EXTRACTORS)
def test_attention_hook_matches(model_cls, onnx_path, width, depth) -> None:
    """The fp16 build hook pins the QK matmul and softmax of every block, so it is not a no-op."""
    del width
    if not onnx_path.exists():
        pytest.skip(f"missing {onnx_path}")
    assert vit_attention_fp32_build_hook in model_cls._build_hooks
    overrides = vit_attention_fp32_build_hook(onnx=onnx_path)
    assert len(overrides["layer_precision"]) == 2 * depth
