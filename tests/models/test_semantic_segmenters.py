# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the semantic segmenter wrappers in src/trtutils/models/semantic_segmenters/."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tests._gpu_fixtures import engine_cache_path
from tests.conftest import DATA_DIR, PREPROCESSORS
from trtutils import TRTEngine
from trtutils.builder import build_engine
from trtutils.core import Buffer
from trtutils.image._schema import SemanticSegmenterOutputSchema
from trtutils.models import DeepLabV3, SegFormer
from trtutils.models._hooks import argmax_build_hook

# (number of classes, background id or None, ((image, class id, min share of the
# non-background pixels), ...)); Pascal VOC 13 is horse, 15 person
_VOC = (21, 0, (("horse", 13, 0.9), ("people", 15, 0.9)))
_ADE = (150, None, (("people", 12, 0.15),))
_CITYSCAPES = (19, None, (("people", 11, 0.15),))

_MOBILENET = DATA_DIR / "deeplabv3" / "deeplabv3_mobilenet_v3_large_520.onnx"
_SEGFORMER_B0 = DATA_DIR / "segformer" / "segformer_b0_ade_512.onnx"


def _segformer(variant: str, imgsz: int, labels: tuple, *, slow: bool = False) -> object:
    onnx_path = DATA_DIR / "segformer" / f"segformer_{variant}_{imgsz}.onnx"
    marks = [pytest.mark.slow] if slow else []
    return pytest.param(SegFormer, onnx_path, labels, id=f"segformer-{variant}", marks=marks)


SEMANTIC_SEGMENTERS = [
    pytest.param(DeepLabV3, _MOBILENET, _VOC, id="deeplabv3-mobilenet-v3-large"),
    pytest.param(
        DeepLabV3,
        DATA_DIR / "deeplabv3" / "deeplabv3_resnet50_520.onnx",
        _VOC,
        id="deeplabv3-resnet50",
    ),
    pytest.param(
        DeepLabV3,
        DATA_DIR / "deeplabv3" / "deeplabv3_resnet101_520.onnx",
        _VOC,
        id="deeplabv3-resnet101",
    ),
    _segformer("b0_ade", 512, _ADE),
    _segformer("b1_ade", 512, _ADE),
    _segformer("b2_ade", 512, _ADE),
    _segformer("b3_ade", 512, _ADE),
    _segformer("b4_ade", 512, _ADE, slow=True),
    _segformer("b5_ade", 640, _ADE, slow=True),
    _segformer("b0_cityscapes", 1024, _CITYSCAPES),
    _segformer("b1_cityscapes", 1024, _CITYSCAPES),
    _segformer("b2_cityscapes", 1024, _CITYSCAPES),
    _segformer("b3_cityscapes", 1024, _CITYSCAPES, slow=True),
    _segformer("b4_cityscapes", 1024, _CITYSCAPES, slow=True),
    _segformer("b5_cityscapes", 1024, _CITYSCAPES, slow=True),
]


@pytest.mark.parametrize(("model_cls", "onnx_path", "labels"), SEMANTIC_SEGMENTERS)
@pytest.mark.parametrize(("preprocessor", "cpu_resize"), PREPROCESSORS)
def test_semantic_segmenter_end2end(
    build_model_engine, images, model_cls, onnx_path, labels, preprocessor, cpu_resize
) -> None:
    """run() + get_class_maps() equals end2end(): a uint8 class map at the image size."""
    num_classes, background, checks = labels
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, cpu_resize=cpu_resize, warmup=False)
    assert model.output_schema is SemanticSegmenterOutputSchema.CLASS_MAP

    for name, expected, min_share in checks:
        image = images[name].array
        via_run = model.get_class_maps(model.run(image))
        via_e2e = model.end2end(image)
        np.testing.assert_array_equal(via_run, via_e2e)

        assert via_e2e.shape == image.shape[:2]
        assert via_e2e.dtype == np.uint8
        assert via_e2e.max() < num_classes
        pixels = via_e2e[via_e2e != background] if background is not None else via_e2e
        # horse.jpg is one horse; people.jpeg is people
        assert (pixels == expected).mean() >= min_share


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


@pytest.mark.correctness
def test_argmax_hook_on_quarter_res_logits(build_model_engine, images) -> None:
    """A dynamic-batch SegFormer class map is the bilinear upsample of its 1/4 logits, argmaxed."""
    if not _SEGFORMER_B0.exists():
        pytest.skip(f"missing {_SEGFORMER_B0}")
    engine_path = engine_cache_path(f"{_SEGFORMER_B0.stem}_argmax_dynamic", "1-4")
    if not engine_path.exists():
        shape = (3, 512, 512)
        build_engine(
            _SEGFORMER_B0,
            engine_path,
            shapes=[("input", ((1, *shape), (2, *shape), (4, *shape)))],
            fp16=True,
            optimization_level=1,
            **argmax_build_hook(imgsz=512),
        )
    logits = SegFormer(
        build_model_engine(SegFormer, _SEGFORMER_B0, tag="logits", argmax=False), warmup=False
    )
    tensor, _, _ = logits.preprocess([images["horse"].array, images["people"].array], method="cpu")
    engine = TRTEngine(engine_path, warmup=False)
    (class_maps,) = engine.execute([Buffer.wrap(np.ascontiguousarray(tensor))])
    assert engine.output_names == ["class_map"]
    assert class_maps.shape == (2, 1, 512, 512)
    for idx in range(2):
        (scores,) = logits.run(tensor[idx : idx + 1], preprocessed=True, postprocess=False)
        assert scores.shape == (1, 150, 128, 128)
        upsampled = np.stack(
            [cv2.resize(c, (512, 512), interpolation=cv2.INTER_LINEAR) for c in scores[0]]
        )
        assert (class_maps[idx, 0] == upsampled.argmax(axis=0)).mean() >= 0.99
    del engine
