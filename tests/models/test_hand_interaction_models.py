# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the HOIDETR and Hands23 model wrappers."""

from __future__ import annotations

import pytest

from tests.conftest import DATA_DIR
from trtutils.core import Buffer, MemoryLocation
from trtutils.image._hand_interaction import HandInteractionDetector
from trtutils.models import HOIDETR, Hands23

HAND_INTERACTION_MODELS = [
    pytest.param(HOIDETR, DATA_DIR / "hoi_detr" / "hoi_detr_vitl_640.onnx", id="hoi-detr"),
    pytest.param(Hands23, DATA_DIR / "hands23" / "hands23_x101_800.onnx", id="hands23"),
]


def test_output_contract_validation(build_test_engine) -> None:
    """An engine that does not follow the unified output contract raises ValueError."""
    engine = build_test_engine(DATA_DIR / "simple.onnx")
    with pytest.raises(ValueError, match="boxes"):
        HandInteractionDetector(engine, warmup=False)


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


@pytest.mark.parametrize(("model_cls", "onnx_path"), HAND_INTERACTION_MODELS)
@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
def test_hand_interaction_end2end(
    build_test_engine,
    images,
    model_cls,
    onnx_path,
    preprocessor,
) -> None:
    """run(), get_interactions(), and end2end() agree and produce valid interactions."""
    if not onnx_path.exists():
        pytest.skip(f"missing {onnx_path}")

    engine = build_test_engine(onnx_path)
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
    _assert_interactions_close(via_run, via_e2e)

    for hand, obj, second, side, contact in via_run:
        for bbox, _score in (b for b in (hand, obj, second) if b is not None):
            x1, y1, x2, y2 = bbox
            assert all(isinstance(coord, int) for coord in bbox)
            assert 0 <= x1 <= width
            assert 0 <= x2 <= width
            assert 0 <= y1 <= height
            assert 0 <= y2 <= height
            assert x1 <= x2
            assert y1 <= y2

        if model_cls is HOIDETR:
            assert side is None
            assert contact is None
        else:
            assert isinstance(side, int)
            assert isinstance(contact, int)


@pytest.mark.parametrize(("model_cls", "onnx_path"), HAND_INTERACTION_MODELS)
@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
def test_hand_interaction_device_buffer_matches_ndarray(
    build_test_engine, images, model_cls, onnx_path, preprocessor
) -> None:
    """A device Buffer passes through preprocess() and run(postprocess=False) like an ndarray."""
    if not onnx_path.exists():
        pytest.skip(f"missing {onnx_path}")

    engine = build_test_engine(onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, warmup=False)
    image = images["people"].array

    expected_tensor, expected_ratios, expected_padding = model.preprocess(image)
    expected_raw = model.run(image, postprocess=False)

    buf = Buffer.from_array(image, MemoryLocation.DEVICE)
    try:
        tensor, ratios, padding = model.preprocess(buf)
        raw = model.run(buf, postprocess=False)
    finally:
        buf.free()

    assert tensor.shape == expected_tensor.shape
    assert ratios == expected_ratios
    assert padding == expected_padding
    assert [o.shape for o in raw] == [o.shape for o in expected_raw]
