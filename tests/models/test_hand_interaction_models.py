# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/image/_hand_interaction.py -- HOIDETR and Hands23 engines end to end."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import DATA_DIR
from trtutils._flags import FLAGS
from trtutils.image._hand_interaction import HandInteractionDetector
from trtutils.image._schema import HandInteractionOutputSchema
from trtutils.models import HOIDETR, Hands23

HOI_DETR_ONNX = DATA_DIR / "hoi_detr" / "hoi_detr_vitl_640.onnx"
HANDS23_ONNX = DATA_DIR / "hands23" / "hands23_x101_800.onnx"
_trt86_hoi = pytest.mark.skipif(
    FLAGS.TRT_VERSION < (10, 0), reason="TensorRT 8.6 cannot build HOI-DETR (Einsum ellipsis)"
)
_trt86_hands23 = pytest.mark.skipif(
    FLAGS.TRT_VERSION < (10, 0), reason="TensorRT 8.6 cannot build Hands23 (OneHot shape tensor)"
)
MODELS = [
    pytest.param(HOIDETR, HOI_DETR_ONNX, id="hoi-detr", marks=_trt86_hoi),
    pytest.param(Hands23, HANDS23_ONNX, id="hands23", marks=_trt86_hands23),
]


def test_output_contract_validation(build_test_engine) -> None:
    """An engine matching no hand interaction schema raises, naming the valid outputs."""
    engine = build_test_engine(DATA_DIR / "simple.onnx")
    with pytest.raises(ValueError, match=r"match no HandInteractionOutputSchema.*pair_probs"):
        HandInteractionDetector(engine, warmup=False)


@pytest.mark.parametrize(
    ("model_cls", "onnx_path", "expected"),
    [
        pytest.param(
            HOIDETR,
            HOI_DETR_ONNX,
            HandInteractionOutputSchema.HOI_DETR,
            id="hoi-detr",
            marks=_trt86_hoi,
        ),
        pytest.param(
            Hands23,
            HANDS23_ONNX,
            HandInteractionOutputSchema.HANDS23,
            id="hands23",
            marks=_trt86_hands23,
        ),
    ],
)
@pytest.mark.parametrize(
    "override",
    [
        pytest.param(None, id="auto"),
        pytest.param("enum", id="enum"),
        pytest.param("str", id="str"),
    ],
)
def test_output_schema(build_model_engine, model_cls, onnx_path, expected, override) -> None:
    """output_schema is auto-detected and accepts an enum member or its name as override."""
    engine = build_model_engine(model_cls, onnx_path)
    schema = {None: None, "enum": expected, "str": expected.name}[override]
    model = model_cls(engine, output_schema=schema, warmup=False)
    assert model.output_schema is expected


def _assert_interactions_close(left, right) -> None:
    # run() and end2end() preprocess on different devices, fp16 scores drift slightly
    assert len(left) == len(right)
    for a, b in zip(left, right):
        for entry_a, entry_b in zip(a[:3], b[:3]):
            assert (entry_a is None) == (entry_b is None)
            if entry_a is not None:
                assert entry_a[0] == pytest.approx(entry_b[0], abs=1)
                assert entry_a[1] == pytest.approx(entry_b[1], abs=0.02)
        assert a[3:] == b[3:]


@pytest.mark.parametrize(
    ("model_cls", "onnx_path", "attr_type"),
    [
        # hoi-detr has no side head and two pair classes, so side and contact stay None
        pytest.param(HOIDETR, HOI_DETR_ONNX, type(None), id="hoi-detr", marks=_trt86_hoi),
        pytest.param(Hands23, HANDS23_ONNX, int, id="hands23", marks=_trt86_hands23),
    ],
)
@pytest.mark.parametrize(
    "preprocessor",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param("cuda", id="cuda"),
        pytest.param("trt", id="trt"),
    ],
)
def test_hand_interaction_end2end(
    build_model_engine,
    images,
    model_cls,
    onnx_path,
    attr_type,
    preprocessor,
) -> None:
    """run(), get_interactions(), and end2end() agree and produce valid interactions."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, warmup=False)

    img = images["people"].array
    height, width = img.shape[:2]

    raw = model.run(img)
    assert len(raw) == 5
    num_dets = raw[0].shape[0]
    assert raw[0].shape == (num_dets, 4)
    assert raw[3].shape[:2] == (num_dets, num_dets)

    via_run = model.get_interactions(raw)
    via_e2e = model.end2end(img)
    assert len(via_run) > 0
    _assert_interactions_close(via_run, via_e2e)

    for hand, obj, second, side, contact in via_run:
        for bbox, _score in (b for b in (hand, obj, second) if b is not None):
            x1, y1, x2, y2 = bbox
            assert all(isinstance(coord, int) for coord in bbox)
            assert 0 <= x1 <= x2 <= width
            assert 0 <= y1 <= y2 <= height
        assert isinstance(side, attr_type)
        assert isinstance(contact, attr_type)


OVERRIDES = [
    ({}, {"conf_thres": 0.0}, "conf-zero"),
    ({"conf_thres": 0.1}, {"pair_thres": 0.0}, "pair-zero"),
    ({"conf_thres": 0.0}, {"second_pair_thres": 0.0}, "second-pair-zero"),
    ({"conf_thres": 0.0}, {"nms_iou_thres": 0.0}, "nms-iou-zero"),
    ({}, {"conf_thres": 0.9, "pair_thres": 0.9}, "conf-pair-strict"),
]


@pytest.mark.parametrize(
    ("model_cls", "onnx_path", "base", "override"),
    [
        pytest.param(model_cls, onnx_path, base, override, id=f"{model_id}-{case_id}", marks=mark)
        for model_cls, onnx_path, model_id, mark in (
            (HOIDETR, HOI_DETR_ONNX, "hoi-detr", _trt86_hoi),
            (Hands23, HANDS23_ONNX, "hands23", _trt86_hands23),
        )
        for base, override, case_id in OVERRIDES
        # hands23 finds no second objects in this image, so the override cannot show
        if not (model_cls is Hands23 and "second_pair_thres" in override)
    ],
)
def test_per_call_thresholds_match_constructor(
    build_model_engine,
    images,
    model_cls,
    onnx_path,
    base,
    override,
) -> None:
    """Per-call thresholds in end2end() (conf and NMS IoU positional) equal passing them to the constructor."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, **base, warmup=False)
    reference = model_cls(engine, **base, **override, warmup=False)

    img = images["people"].array
    per_call = model.end2end(
        img,
        override.get("conf_thres"),
        override.get("nms_iou_thres"),
        pair_thres=override.get("pair_thres"),
        second_pair_thres=override.get("second_pair_thres"),
    )

    assert per_call == reference.end2end(img)
    assert per_call != model.end2end(img)


@pytest.mark.parametrize(("model_cls", "onnx_path"), MODELS)
def test_batch_input(build_model_engine, images, model_cls, onnx_path) -> None:
    """List input returns per-image results equal to the single-image calls."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, warmup=False)

    img = images["people"].array
    single = model.end2end(img)
    assert len(single) > 0
    assert model.end2end([img]) == [single]

    raw = model.run(img)
    raw_batch = model.run([img])
    assert len(raw_batch) == 1
    for a, b in zip(raw_batch[0], raw):
        np.testing.assert_array_equal(a, b)
    assert model.get_interactions(raw_batch) == [model.get_interactions(raw)]


@pytest.mark.parametrize(
    ("names", "ref_names"),
    [
        pytest.param(("people", "people"), ("people", "people"), id="duplicate"),
        pytest.param(("people", "horse"), ("people", "horse"), id="mixed"),
        pytest.param(("horse", "people"), ("horse", "people"), id="swapped"),
    ],
)
@_trt86_hoi
def test_batch_slots_match_single(build_model_engine, images, names, ref_names) -> None:
    """Every slot of a batch-2 engine matches the batch-1 result for its image."""
    single = HOIDETR(build_model_engine(HOIDETR, HOI_DETR_ONNX), warmup=False)
    batched = HOIDETR(build_model_engine(HOIDETR, HOI_DETR_ONNX, batch_size=2), warmup=False)

    # low thresholds so objects and links show up, people has none at the defaults
    kwargs = {"conf_thres": 0.1, "pair_thres": 0.3}
    results = batched.end2end([images[n].array for n in names], **kwargs)
    assert len(results) == 2
    for result, ref_name in zip(results, ref_names):
        expected = single.end2end(images[ref_name].array, **kwargs)
        # fp16 near-ties can reorder equal scores between engines
        _assert_interactions_close(
            sorted(result, key=lambda t: t[0][0]), sorted(expected, key=lambda t: t[0][0])
        )
