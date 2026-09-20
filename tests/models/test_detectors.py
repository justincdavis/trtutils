# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the detector wrappers in src/trtutils/models/detectors/."""

from __future__ import annotations

import pytest

from tests.conftest import DATA_DIR
from trtutils.models import (
    DEIM,
    DFINE,
    RFDETR,
    YOLOX,
    DEIMv2,
    RTDETRv1,
    RTDETRv2,
    RTDETRv3,
    YOLOv3,
    YOLOv5,
    YOLOv7,
    YOLOv8,
    YOLOv9,
    YOLOv10,
    YOLOv11,
    YOLOv12,
    YOLOv13,
    YOLOv26,
)

DETECTORS = [
    pytest.param(YOLOv3, DATA_DIR / "yolov3" / "yolov3t_640.onnx", id="yolov3"),
    pytest.param(YOLOv5, DATA_DIR / "yolov5" / "yolov5n_640.onnx", id="yolov5"),
    pytest.param(YOLOv7, DATA_DIR / "yolov7" / "yolov7t_640.onnx", id="yolov7"),
    pytest.param(YOLOv8, DATA_DIR / "yolov8" / "yolov8n_640.onnx", id="yolov8"),
    pytest.param(YOLOv9, DATA_DIR / "yolov9" / "yolov9t_640.onnx", id="yolov9"),
    pytest.param(YOLOv10, DATA_DIR / "yolov10" / "yolov10n_640.onnx", id="yolov10"),
    pytest.param(YOLOv11, DATA_DIR / "yolov11" / "yolov11n_640.onnx", id="yolov11"),
    pytest.param(YOLOv12, DATA_DIR / "yolov12" / "yolov12n_640.onnx", id="yolov12"),
    pytest.param(YOLOv13, DATA_DIR / "yolov13" / "yolov13n_640.onnx", id="yolov13"),
    pytest.param(YOLOv26, DATA_DIR / "yolov26" / "yolov26n_640.onnx", id="yolov26"),
    pytest.param(YOLOX, DATA_DIR / "yolox" / "yoloxs_640.onnx", id="yolox"),
    pytest.param(RTDETRv1, DATA_DIR / "rtdetrv1" / "rtdetrv1_r18_640.onnx", id="rtdetrv1"),
    pytest.param(RTDETRv2, DATA_DIR / "rtdetrv2" / "rtdetrv2_r18_640.onnx", id="rtdetrv2"),
    pytest.param(RTDETRv3, DATA_DIR / "rtdetrv3" / "rtdetrv3_r18_640.onnx", id="rtdetrv3"),
    pytest.param(DFINE, DATA_DIR / "dfine" / "dfine_n_640.onnx", id="dfine"),
    pytest.param(DEIM, DATA_DIR / "deim" / "deim_dfine_n_640.onnx", id="deim"),
    pytest.param(DEIMv2, DATA_DIR / "deimv2" / "deimv2_atto_320.onnx", id="deimv2"),
    pytest.param(RFDETR, DATA_DIR / "rfdetr" / "rfdetr_n_576.onnx", id="rfdetr"),
]


@pytest.mark.parametrize(("model_cls", "onnx_path"), DETECTORS)
@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt"])
def test_detector_end2end(build_model_engine, images, model_cls, onnx_path, preprocessor) -> None:
    """run(), get_detections(), and end2end() agree and find the ground-truth classes in bounds."""
    engine = build_model_engine(model_cls, onnx_path)
    model = model_cls(engine, preprocessor=preprocessor, warmup=False)
    image = images["horse"]
    height, width = image.array.shape[:2]

    postprocessed = model.run(image.array)
    assert len(postprocessed) == 3
    via_run = model.get_detections(postprocessed)
    via_e2e = model.end2end(image.array)
    assert via_run == via_e2e

    assert len(via_run) >= image.gt_det_min
    assert set(image.gt_det_classes) <= {cls_id for _bbox, _score, cls_id in via_run}
    for (x1, y1, x2, y2), score, _cls_id in via_run:
        assert 0 <= x1 <= x2 <= width
        assert 0 <= y1 <= y2 <= height
        assert 0.0 <= score <= 1.0
