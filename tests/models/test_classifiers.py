# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the classifier wrappers in src/trtutils/models/classifiers/."""

from __future__ import annotations

import pytest

from tests.conftest import DATA_DIR
from trtutils.models import EfficientNet, MobileNetV3, ResNet

CLASSIFIERS = [
    pytest.param(ResNet, DATA_DIR / "resnet" / "resnet18_224.onnx", id="resnet18"),
    pytest.param(
        EfficientNet, DATA_DIR / "efficientnet" / "efficientnet_b0_224.onnx", id="efficientnet-b0"
    ),
    pytest.param(
        MobileNetV3,
        DATA_DIR / "mobilenet" / "mobilenet_v3_small_224.onnx",
        id="mobilenet-v3-small",
    ),
]


@pytest.mark.parametrize(("model_cls", "onnx_path"), CLASSIFIERS)
@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
def test_classifier_end2end(build_model_engine, images, model_cls, onnx_path, preprocessor) -> None:
    """run(), get_classifications(), and end2end() agree and rank the ground-truth class in the top 5."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, warmup=False)
    image = images["horse"]

    postprocessed = model.run(image.array)
    via_run = model.get_classifications(postprocessed, top_k=5)
    via_e2e = model.end2end(image.array, top_k=5)
    assert via_run == via_e2e

    assert len(via_run) == 5
    scores = [score for _cls_id, score in via_run]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert image.gt_cls_id in [cls_id for cls_id, _score in via_run]
