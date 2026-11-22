# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/postprocessors/_cpu_semantic.py and _semantic_segmenter.py -- class maps back at the original resolution."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from trtutils.image import ImageModel, SemanticSegmenter
from trtutils.image._schema import SemanticSegmenterOutputSchema
from trtutils.image.postprocessors import (
    get_class_maps,
    postprocess_class_map,
    postprocess_semantic_logits,
)
from trtutils.image.preprocessors._cpu import resize_image

pytestmark = pytest.mark.cpu

_INPUT = (160, 160)
_CLASSES = 21


def _ground_truth(height: int, width: int, block: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    return (((yy // block) * 7 + (xx // block) * 3) % _CLASSES).astype(np.uint8)


def _engine_map(gt: np.ndarray, method: str) -> tuple[np.ndarray, tuple, tuple]:
    """The class map an ideal engine returns for gt, with the preprocessors' geometry."""
    dummy = np.zeros((*gt.shape, 3), dtype=np.uint8)
    _, ratios, padding = resize_image(dummy, _INPUT, method)
    content_w = round(_INPUT[0] - 2 * padding[0])
    content_h = round(_INPUT[1] - 2 * padding[1])
    left, top = round(padding[0] - 0.1), round(padding[1] - 0.1)
    out = np.zeros((_INPUT[1], _INPUT[0]), dtype=np.int32)
    out[top : top + content_h, left : left + content_w] = cv2.resize(
        gt, (content_w, content_h), interpolation=cv2.INTER_NEAREST
    )
    return out, ratios, padding


def _as_schema(class_map: np.ndarray, schema: str) -> np.ndarray:
    if schema == "class_map":
        return class_map[None, None]
    # one-hot logits whose argmax is the class map
    return np.eye(_CLASSES, dtype=np.float32)[class_map].transpose(2, 0, 1)[None]


_POST = {"class_map": postprocess_class_map, "logits": postprocess_semantic_logits}


@pytest.mark.parametrize("schema", ["class_map", "logits"])
@pytest.mark.parametrize("method", ["letterbox", "linear"])
@pytest.mark.parametrize(
    ("height", "width"),
    [
        pytest.param(37, 53, id="37x53"),
        pytest.param(31, 71, id="31x71"),
        pytest.param(64, 48, id="64x48"),
        pytest.param(160, 160, id="160x160"),
    ],
)
@pytest.mark.parametrize("shapes", [True, False], ids=["image-shapes", "from-ratios"])
def test_class_map_round_trip(schema, method, height, width, shapes) -> None:
    """A class map upscaled at least 2x maps back to the original pixels exactly, padding cropped."""
    gt = _ground_truth(height, width, block=1)
    class_map, ratios, padding = _engine_map(gt, method)

    result = _POST[schema](
        [_as_schema(class_map, schema)],
        [ratios],
        [padding],
        _INPUT,
        image_shapes=[(height, width)] if shapes else None,
    )

    assert len(result) == 1
    (mapped,) = result[0]
    assert mapped.dtype == np.uint8
    np.testing.assert_array_equal(mapped, gt)


@pytest.mark.parametrize("method", ["letterbox", "linear"])
@pytest.mark.parametrize(
    ("height", "width"),
    [
        pytest.param(1081, 1919, id="1081x1919"),
        pytest.param(479, 641, id="479x641"),
        pytest.param(997, 211, id="997x211"),
    ],
)
def test_class_map_downscaled_odd_sizes(method, height, width) -> None:
    """A downscaled class map comes back at the image size and agrees away from block edges."""
    gt = _ground_truth(height, width, block=max(height, width) // 4)
    class_map, ratios, padding = _engine_map(gt, method)

    (mapped,) = postprocess_class_map(
        [class_map[None]], [ratios], [padding], _INPUT, image_shapes=[(height, width)]
    )[0]

    assert mapped.shape == (height, width)
    assert (mapped == gt).mean() >= 0.9


def test_class_map_batch_keeps_each_image_size() -> None:
    """Each image of a batch gets its own geometry and size."""
    sizes = [(37, 53), (64, 48)]
    maps, ratios, padding = [], [], []
    for height, width in sizes:
        class_map, ratio, pad = _engine_map(_ground_truth(height, width, block=1), "letterbox")
        maps.append(class_map)
        ratios.append(ratio)
        padding.append(pad)

    result = postprocess_class_map(
        [np.stack(maps)[:, None]], ratios, padding, _INPUT, image_shapes=sizes
    )

    for (height, width), (mapped,) in zip(sizes, result):
        np.testing.assert_array_equal(mapped, _ground_truth(height, width, block=1))


def test_center_crop_border_takes_nearest_class() -> None:
    """With center_crop the class map covers the whole image, the cropped border included."""
    height, width = 120, 200
    _, ratios, padding = resize_image(np.zeros((height, width, 3), np.uint8), _INPUT, "center_crop")
    class_map = np.zeros(_INPUT[::-1], dtype=np.int32)
    class_map[:, : _INPUT[0] // 2] = 1
    class_map[:, _INPUT[0] // 2 :] = 2

    (mapped,) = postprocess_class_map(
        [class_map[None]], [ratios], [padding], _INPUT, image_shapes=[(height, width)]
    )[0]

    assert mapped.shape == (height, width)
    assert (mapped[:, : width // 2 - 2] == 1).all()
    assert (mapped[:, width // 2 + 2 :] == 2).all()


def test_logits_over_256_classes_rejected() -> None:
    """Logits with more classes than uint8 holds raise instead of wrapping."""
    logits = np.zeros((1, 257, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="max 256"):
        postprocess_semantic_logits([logits], [(1.0, 1.0)], [(0.0, 0.0)], (4, 4))


def _stub(schema: SemanticSegmenterOutputSchema) -> SimpleNamespace:
    """The state SemanticSegmenter.postprocess / get_class_maps read, without an engine."""
    return SimpleNamespace(
        _output_schema=schema,
        _input_size=_INPUT,
        _tag="stub",
        _nvtx_tags={},
        _host_outputs=lambda outputs: outputs,
        _as_batch=ImageModel._as_batch,
    )


@pytest.mark.parametrize(
    ("schema", "kind"),
    [
        pytest.param(SemanticSegmenterOutputSchema.CLASS_MAP, "class_map", id="class-map"),
        pytest.param(SemanticSegmenterOutputSchema.LOGITS, "logits", id="logits"),
        pytest.param(SemanticSegmenterOutputSchema.GENERIC, "logits", id="generic-as-logits"),
    ],
)
def test_model_postprocess_dispatch_and_get_class_maps(schema, kind) -> None:
    """Postprocess dispatches on the schema; get_class_maps unwraps a single image."""
    gt = _ground_truth(37, 53, block=1)
    class_map, ratios, padding = _engine_map(gt, "letterbox")
    model = _stub(schema)

    post = SemanticSegmenter.postprocess(
        model,  # ty: ignore[invalid-argument-type]
        [_as_schema(class_map, kind)],
        [ratios],
        [padding],
        image_shapes=[gt.shape],
    )
    single = SemanticSegmenter.get_class_maps(model, post[0])  # ty: ignore[invalid-argument-type]
    batch = SemanticSegmenter.get_class_maps(model, post)  # ty: ignore[invalid-argument-type]

    np.testing.assert_array_equal(single, gt)
    assert len(batch) == 1
    np.testing.assert_array_equal(batch[0], gt)
    np.testing.assert_array_equal(get_class_maps(post)[0], gt)
