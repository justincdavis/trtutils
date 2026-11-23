# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the classifier wrappers in src/trtutils/models/classifiers/."""

from __future__ import annotations

import cv2
import pytest

from tests.conftest import DATA_DIR, PREPROCESSORS
from trtutils.image._schema import ClassifierOutputSchema
from trtutils.models import DeiT, DeiT3, EfficientNet, MobileNetV3, MobileNetV4, ResNet

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
    pytest.param(DeiT, DATA_DIR / "deit" / "deit_tiny_224.onnx", id="deit-tiny"),
    pytest.param(
        DeiT, DATA_DIR / "deit" / "deit_small_distilled_224.onnx", id="deit-small-distilled"
    ),
    pytest.param(DeiT3, DATA_DIR / "deit3" / "deit3_small_224.onnx", id="deit3-small"),
    *[
        pytest.param(
            MobileNetV4,
            DATA_DIR / "mobilenetv4" / f"mobilenetv4_{size}.onnx",
            id=f"mobilenetv4-{size.rsplit('_', 1)[0].replace('_', '-')}",
        )
        for size in [
            "conv_small_224",
            "conv_medium_256",
            "conv_large_384",
            "hybrid_medium_256",
            "hybrid_large_384",
        ]
    ],
]


@pytest.mark.parametrize(("model_cls", "onnx_path"), CLASSIFIERS)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_classifier_end2end(
    build_model_engine, images, model_cls, onnx_path, preprocessor, cpu_resize
) -> None:
    """run(), get_classifications(), and end2end() agree and rank the ground-truth class in the top 5."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False)
    assert model.output_schema is ClassifierOutputSchema.TORCHVISION
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


@pytest.mark.parametrize(
    "embed_sst", [pytest.param(False, id="float"), pytest.param(True, id="sst")]
)
def test_dynamic_batch_matches_static(build_model_engine, tmp_path, images, embed_sst) -> None:
    """A dynamic-batch build runs batch 1 and 3 of a max 4 and matches the static batch-1 engine."""
    onnx_path = DATA_DIR / "resnet" / "resnet18_224.onnx"
    static = ResNet(build_model_engine(ResNet, onnx_path, embed_sst=embed_sst), warmup=False)
    engine_path = tmp_path / "dynamic.engine"
    ResNet.build(
        onnx_path, engine_path, batch_size=4, opt_level=1, embed_sst=embed_sst, dynamic_batch=True
    )
    dynamic = ResNet(engine_path, warmup=False)
    assert dynamic.engine.is_dynamic_batch is True
    assert dynamic.engine.batch_size == 4

    frames = [
        images["horse"].array,
        images["people"].array,
        cv2.imread(str(DATA_DIR / "street.jpg")),
    ]
    expected = [static.end2end(frame, top_k=1)[0] for frame in frames]
    got = [dynamic.end2end(frames[0], top_k=1)[0]]
    got += [result[0] for result in dynamic.end2end(frames, top_k=1)]
    for (cls_id, score), (ref_id, ref_score) in zip(got, [expected[0], *expected]):
        assert cls_id == ref_id
        assert score == pytest.approx(ref_score, abs=1e-2)
