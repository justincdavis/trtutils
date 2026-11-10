# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""An engine built with embed_sst matches the plain engine for every family with data/ ONNX."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import DATA_DIR
from tests.models.test_detectors import _assert_same_detections
from tests.models.test_hand_interaction_models import _assert_interactions_close
from trtutils.models import RFDETR, DepthAnythingV2, Hands23, ResNet, RTDETRv3, YOLOv10


def _assert_same_classes(left, right) -> None:
    assert [cls_id for cls_id, _ in left] == [cls_id for cls_id, _ in right]
    assert [score for _, score in left] == pytest.approx([score for _, score in right], abs=0.02)


def _assert_close_depth(left, right) -> None:
    # normalized to [0, 1]
    assert np.abs(left - right).mean() < 0.01


def _assert_same_interactions(left, right) -> None:
    # fp16 drift moves a hand at 0.302 across the 0.3 threshold and reorders near-tied scores,
    # so drop borderline hands and compare in left-to-right order
    def stable(interactions):
        kept = [i for i in interactions if i[0][1] > 0.35]
        return sorted(kept, key=lambda i: i[0][0][0])

    _assert_interactions_close(stable(left), stable(right))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model_cls", "onnx_path", "image_name", "assert_same"),
    [
        pytest.param(
            YOLOv10,
            DATA_DIR / "yolov10" / "yolov10n_640.onnx",
            "horse",
            _assert_same_detections,
            id="yolov10",
        ),
        # image is the second of three inputs
        pytest.param(
            RTDETRv3,
            DATA_DIR / "rtdetrv3" / "rtdetrv3_r18_640.onnx",
            "horse",
            _assert_same_detections,
            id="rtdetrv3",
        ),
        pytest.param(
            RFDETR,
            DATA_DIR / "rfdetr" / "rfdetr_n_384.onnx",
            "horse",
            _assert_same_detections,
            id="rfdetr",
        ),
        pytest.param(
            ResNet,
            DATA_DIR / "resnet" / "resnet18_224.onnx",
            "horse",
            _assert_same_classes,
            id="resnet",
        ),
        pytest.param(
            DepthAnythingV2,
            DATA_DIR / "depth_anything_v2" / "depth_anything_v2_small_518.onnx",
            "horse",
            _assert_close_depth,
            id="depth-anything-v2",
        ),
        pytest.param(
            Hands23,
            DATA_DIR / "hands23" / "hands23_x101_800.onnx",
            "people",
            _assert_same_interactions,
            id="hands23",
        ),
    ],
)
# trt resolves to the CUDA preprocessor: its engine only normalizes
@pytest.mark.parametrize("preprocessor", ["cuda", "trt"])
def test_embedded_engine_matches_plain(
    build_model_engine, images, model_cls, onnx_path, image_name, assert_same, preprocessor
) -> None:
    """end2end on the uint8 NHWC engine matches the float NCHW engine on the same image."""
    plain = model_cls(build_model_engine(model_cls, onnx_path), warmup=False)
    model = model_cls(
        build_model_engine(model_cls, onnx_path, embed_sst=True),
        preprocessor=preprocessor,
        warmup=False,
    )
    image = images[image_name].array
    assert_same(plain.end2end(image), model.end2end(image))

    tensor, _, _ = model.preprocess(image)
    assert tensor.dtype == np.uint8
    assert tensor.shape == (1, *model.input_shape[::-1], 3)
    assert int(tensor.max()) > 0


def test_embedded_engine_fixes_the_norm(build_model_engine) -> None:
    """input_range, mean, and std are baked at build time, and CPU preprocessing is refused."""
    engine = build_model_engine(YOLOv10, DATA_DIR / "yolov10" / "yolov10n_640.onnx", embed_sst=True)
    model = YOLOv10(engine, warmup=False)
    with pytest.raises(ValueError, match="fixed at build time"):
        model.update_input_range((0, 255))
    with pytest.raises(ValueError, match="fixed at build time"):
        model.update_mean_std((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    with pytest.raises(ValueError, match="cuda preprocessor"):
        YOLOv10(engine, preprocessor="cpu", warmup=False)
