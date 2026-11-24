# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the segmenter wrappers in src/trtutils/models/segmenters/."""

from __future__ import annotations

import pytest

from tests.conftest import DATA_DIR
from trtutils.models import YOLOE

_YOLOE = DATA_DIR / "yoloe"

# text-prompt exports with the prompts below, the default COCO names, and prompt-free models
SEGMENTERS = [
    pytest.param(YOLOE, _YOLOE / "yoloe-v8s-seg_640.onnx", id="yoloe-v8"),
    pytest.param(YOLOE, _YOLOE / "yoloe-11s-seg_640.onnx", id="yoloe-11"),
    pytest.param(YOLOE, _YOLOE / "yoloe-26n-seg_640.onnx", id="yoloe-26-coco"),
    pytest.param(YOLOE, _YOLOE / "yoloe-11s-seg-pf_640.onnx", id="yoloe-11-prompt-free"),
    pytest.param(YOLOE, _YOLOE / "yoloe-26n-seg-pf_640.onnx", id="yoloe-26-prompt-free"),
]

# the prompts of yoloe-11s-seg_640.onnx and, reversed, of yoloe-11s-seg-rev_640.onnx
_PROMPTS = ["person", "car", "bus", "truck", "horse", "traffic light"]


def _names(onnx_path) -> list[str]:
    return onnx_path.with_suffix(".classes.txt").read_text().splitlines()


@pytest.mark.parametrize(("model_cls", "onnx_path"), SEGMENTERS)
@pytest.mark.parametrize(("image_key", "label"), [("horse", "horse"), ("people", "person")])
def test_segmenter_end2end(
    build_model_engine, images, model_cls, onnx_path, image_key, label
) -> None:
    """end2end() finds the labelled object, named through the exported class names file."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, warmup=False)
    image = images[image_key]
    height, width = image.array.shape[:2]
    names = _names(onnx_path)
    # by name: TensorRT 8.6 lists the mask prototypes (output1) first
    detections = model.engine.output_names.index("output0")
    assert model.engine.output_spec[detections][0][1] == 4 + len(names) + 32

    segmentations = model.end2end(image.array, conf_thres=0.25)
    assert label in {names[cls_id] for _bbox, _score, cls_id, _mask in segmentations}
    for (x1, y1, x2, y2), score, _cls_id, mask in segmentations:
        assert 0 <= x1 <= x2 <= width
        assert 0 <= y1 <= y2 <= height
        assert 0.0 <= score <= 1.0
        assert mask.shape[:2] == (height, width)


@pytest.mark.correctness
@pytest.mark.parametrize(
    ("onnx_name", "prompts"),
    [
        pytest.param("yoloe-11s-seg_640.onnx", _PROMPTS, id="given-order"),
        pytest.param("yoloe-11s-seg-rev_640.onnx", _PROMPTS[::-1], id="reversed"),
    ],
)
def test_class_ids_follow_prompt_order(build_model_engine, images, onnx_name, prompts) -> None:
    """Class ids index the prompts in the order given at export, as the names file lists them."""
    onnx_path = _YOLOE / onnx_name
    engine = build_model_engine(YOLOE, onnx_path)
    assert _names(onnx_path) == prompts
    model = YOLOE(engine, warmup=False)
    for image_key, label in (("horse", "horse"), ("people", "person")):
        segmentations = model.end2end(images[image_key].array, conf_thres=0.5)
        _bbox, _score, top_cls, _mask = max(segmentations, key=lambda seg: seg[1])
        assert top_cls == prompts.index(label)
