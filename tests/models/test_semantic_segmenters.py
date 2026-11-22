# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the semantic segmenter wrappers in src/trtutils/models/semantic_segmenters/."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import DATA_DIR, PREPROCESSORS
from trtutils.image._schema import SemanticSegmenterOutputSchema
from trtutils.models import DeepLabV3

# Pascal VOC ids of the default torchvision weights
_VOC_CLASSES = 21
_HORSE = 13
_PERSON = 15

_MOBILENET = DATA_DIR / "deeplabv3" / "deeplabv3_mobilenet_v3_large_520.onnx"

SEMANTIC_SEGMENTERS = [
    pytest.param(DeepLabV3, _MOBILENET, id="deeplabv3-mobilenet-v3-large"),
    pytest.param(
        DeepLabV3, DATA_DIR / "deeplabv3" / "deeplabv3_resnet50_520.onnx", id="deeplabv3-resnet50"
    ),
    pytest.param(
        DeepLabV3,
        DATA_DIR / "deeplabv3" / "deeplabv3_resnet101_520.onnx",
        id="deeplabv3-resnet101",
    ),
]


@pytest.mark.parametrize(("model_cls", "onnx_path"), SEMANTIC_SEGMENTERS)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_semantic_segmenter_end2end(
    build_model_engine, images, model_cls, onnx_path, preprocessor, cpu_resize
) -> None:
    """run() + get_class_maps() equals end2end(): a uint8 VOC class map at the image size."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False)
    assert model.output_schema is SemanticSegmenterOutputSchema.CLASS_MAP

    for name, expected in (("horse", _HORSE), ("people", _PERSON)):
        image = images[name].array
        via_run = model.get_class_maps(model.run(image))
        via_e2e = model.end2end(image)
        np.testing.assert_array_equal(via_run, via_e2e)

        assert via_e2e.shape == image.shape[:2]
        assert via_e2e.dtype == np.uint8
        assert via_e2e.max() < _VOC_CLASSES
        foreground = via_e2e[via_e2e > 0]
        # horse.jpg is one horse; people.jpeg is people
        assert (foreground == expected).mean() >= 0.9


@pytest.mark.correctness
@pytest.mark.parametrize("resize_method", ["linear", "letterbox"])
def test_argmax_hook_matches_logits_engine(build_model_engine, images, resize_method) -> None:
    """The in-engine argmax and the CPU argmax of a logits engine give the same class maps."""
    hooked = DeepLabV3(build_model_engine(DeepLabV3, _MOBILENET), resize_method=resize_method)
    logits = DeepLabV3(
        build_model_engine(DeepLabV3, _MOBILENET, tag="logits", argmax=False),
        resize_method=resize_method,
    )
    assert logits.output_schema is SemanticSegmenterOutputSchema.LOGITS

    for name in ("horse", "people"):
        image = images[name].array
        a, b = hooked.end2end(image), logits.end2end(image)
        assert a.shape == b.shape == image.shape[:2]
        assert (a == b).mean() >= 0.995
