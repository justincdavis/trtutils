# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the HOIDETR and Hands23 model wrappers."""

from __future__ import annotations

import pytest

from tests.conftest import DATA_DIR
from trtutils.image._hand_interaction import HandInteractionDetector
from trtutils.models import HOIDETR, Hands23


def test_output_contract_validation(build_test_engine) -> None:
    """An engine that does not follow the unified output contract raises ValueError."""
    engine = build_test_engine(DATA_DIR / "simple.onnx")
    with pytest.raises(ValueError, match="boxes"):
        HandInteractionDetector(engine, warmup=False)


@pytest.mark.parametrize(
    ("model_cls", "onnx_path"),
    [
        pytest.param(HOIDETR, DATA_DIR / "hoi_detr" / "hoi_detr_vitl_640.onnx", id="hoi-detr"),
        pytest.param(Hands23, DATA_DIR / "hands23" / "hands23_x101_800.onnx", id="hands23"),
    ],
)
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
    assert via_run == via_e2e

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
