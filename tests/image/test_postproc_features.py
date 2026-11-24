# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_cpu_features.py -- feature extractor postprocessing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trtutils.image import FeatureExtractor
from trtutils.image._schema import FeatureExtractorOutputSchema
from trtutils.image.postprocessors import (
    get_embeddings,
    get_feature_maps,
    postprocess_features,
)

pytestmark = pytest.mark.cpu

_RNG = np.random.default_rng(0)
_B, _C, _H, _W = 3, 8, 4, 5


@pytest.mark.parametrize(
    "embedding_shape",
    [
        pytest.param((_B, _C), id="flat"),
        pytest.param((_B, 1, _C), id="token-axis"),
        pytest.param((_B, _C, 1, 1), id="pooled-map"),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("normalize", [False, True], ids=["raw", "l2"])
def test_postprocess_roundtrip(embedding_shape, dtype, normalize) -> None:
    """Embeddings come back (C,) float32 per image (L2 normalized on request), maps untouched."""
    embedding = _RNG.standard_normal(embedding_shape).astype(dtype)
    feature_map = _RNG.standard_normal((_B, _C, _H, _W)).astype(dtype)

    outputs = postprocess_features([embedding, feature_map], normalize=normalize)
    embeddings = get_embeddings(outputs)
    maps = get_feature_maps(outputs)

    assert len(outputs) == len(embeddings) == len(maps) == _B
    expected = embedding.reshape(_B, _C).astype(np.float32)
    if normalize:
        expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
    for i in range(_B):
        assert embeddings[i].shape == (_C,)
        assert embeddings[i].dtype == np.float32
        np.testing.assert_allclose(embeddings[i], expected[i], rtol=1e-6)
        assert maps[i].shape == (_C, _H, _W)
        assert maps[i].dtype == np.float32
        np.testing.assert_array_equal(maps[i], feature_map[i].astype(np.float32))
    if normalize:
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=-1), 1.0, rtol=1e-6)


def test_normalize_keeps_zero_embedding() -> None:
    """An all-zero embedding stays zero instead of turning into NaN."""
    outputs = postprocess_features([np.zeros((1, _C), dtype=np.float32)], normalize=True)
    np.testing.assert_array_equal(get_embeddings(outputs)[0], np.zeros(_C, dtype=np.float32))


@pytest.mark.parametrize(
    ("no_copy", "shares"),
    [
        pytest.param(True, True, id="no-copy-view"),
        pytest.param(False, False, id="copy"),
        pytest.param(None, False, id="default-copy"),
    ],
)
def test_no_copy(no_copy, shares) -> None:
    """no_copy returns views of float32 feature maps; embeddings always own their data."""
    embedding = _RNG.standard_normal((_B, _C)).astype(np.float32)
    feature_map = _RNG.standard_normal((_B, _C, _H, _W)).astype(np.float32)
    outputs = postprocess_features([embedding, feature_map], no_copy=no_copy)
    assert not np.shares_memory(outputs[0][0], embedding)
    assert np.shares_memory(outputs[0][1], feature_map) is shares


def test_get_feature_maps_without_map() -> None:
    """get_feature_maps on embedding-only outputs raises instead of indexing past the list."""
    outputs = postprocess_features([_RNG.standard_normal((_B, _C)).astype(np.float32)])
    assert all(len(image_outputs) == 1 for image_outputs in outputs)
    with pytest.raises(ValueError, match="no feature map"):
        get_feature_maps(outputs)


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(FeatureExtractorOutputSchema.EMBEDDING, id="embedding"),
        pytest.param(FeatureExtractorOutputSchema.GENERIC, id="generic"),
    ],
)
def test_model_get_feature_maps_names_schema(schema) -> None:
    """FeatureExtractor.get_feature_maps names the schema when the engine has no map."""
    model = SimpleNamespace(_output_schema=schema, _tag="dinov3")
    with pytest.raises(ValueError, match=f"{schema.name}.*EMBEDDING_MAP"):
        FeatureExtractor.get_feature_maps(model, [np.zeros(_C, dtype=np.float32)])  # ty: ignore[invalid-argument-type]
