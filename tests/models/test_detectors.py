# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Engine-level tests for the detector wrappers in src/trtutils/models/detectors/."""

from __future__ import annotations

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from tests.conftest import DATA_DIR
from trtutils._flags import FLAGS
from trtutils.builder._onnx import read_onnx
from trtutils.compat._libs import trt
from trtutils.image import COCO91_TO_COCO80
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
from trtutils.models._hooks import (
    depthwise_fp32_build_hook,
    gridsample_debug_build_hook,
    nms_build_hook,
    rtdetr_build_hook,
    vit_attention_fp32_build_hook,
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
    pytest.param(RFDETR, DATA_DIR / "rfdetr" / "rfdetr_n_384.onnx", id="rfdetr"),
]

# models that return their own class index instead of the 80-class COCO index
CLASS_ID_OVERRIDES = {RFDETR: {17: 19}}

# builder opt level 1 miscompiles these (horse 0.10 / absent vs 0.93 in onnxruntime, random per
# build); the default level 3 matches onnxruntime in every build (0.85-0.93 over 8 builds)
OPT_LEVELS = {DFINE: 3, DEIM: 3}


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0) * max(ih, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _assert_same_detections(left, right) -> None:
    # EfficientNMS breaks exact fp16 score ties nondeterministically (a different equivalent box
    # can survive, equal scores can swap order), so match by class + IoU, not by position
    assert len(left) == len(right)
    unmatched = list(right)
    for bbox, score, cls_id in left:
        match = next(
            (
                d
                for d in unmatched
                if d[2] == cls_id and _iou(bbox, d[0]) >= 0.9 and abs(score - d[1]) < 1e-2
            ),
            None,
        )
        assert match is not None, f"no match for {cls_id} {bbox} {score}"
        unmatched.remove(match)


@pytest.mark.parametrize(("model_cls", "onnx_path"), DETECTORS)
@pytest.mark.parametrize("preprocessor", ["cpu", "cuda", "trt", "hybrid"])
def test_detector_end2end(build_model_engine, images, model_cls, onnx_path, preprocessor) -> None:
    """run(), get_detections(), and end2end() agree and find the ground-truth classes in bounds."""
    engine = build_model_engine(model_cls, onnx_path, opt_level=OPT_LEVELS.get(model_cls, 1))
    model = model_cls(engine, preprocessor=preprocessor, warmup=False)
    image = images["horse"]
    height, width = image.array.shape[:2]

    postprocessed = model.run(image.array)
    assert len(postprocessed) == 3
    via_run = model.get_detections(postprocessed)
    via_e2e = model.end2end(image.array)
    _assert_same_detections(via_run, via_e2e)

    assert len(via_run) >= image.gt_det_min
    overrides = CLASS_ID_OVERRIDES.get(model_cls, {})
    expected = {overrides.get(cls_id, cls_id) for cls_id in image.gt_det_classes}
    assert expected <= {cls_id for _bbox, _score, cls_id in via_run}
    for (x1, y1, x2, y2), score, _cls_id in via_run:
        assert 0 <= x1 <= x2 <= width
        assert 0 <= y1 <= y2 <= height
        assert 0.0 <= score <= 1.0


@pytest.mark.parametrize(
    ("model_cls", "onnx_path", "num_depthwise"),
    [
        pytest.param(YOLOv10, DATA_DIR / "yolov10" / "yolov10n_640.onnx", 13, id="yolov10"),
        pytest.param(YOLOv13, DATA_DIR / "yolov13" / "yolov13n_640.onnx", 42, id="yolov13"),
    ],
)
@pytest.mark.skipif(not FLAGS.WEAK_TYPING, reason="per-layer precision was removed in TensorRT 11")
def test_build_pins_depthwise_convs_fp32(model_cls, onnx_path, num_depthwise) -> None:
    """Builds force FP32 on every depthwise conv and make TensorRT obey it."""
    if not onnx_path.exists():
        pytest.skip(f"missing {onnx_path}")
    assert depthwise_fp32_build_hook in model_cls._build_hooks
    overrides = depthwise_fp32_build_hook(onnx=onnx_path)
    assert overrides["prefer_precision_constraints"] is True
    assert len(overrides["layer_precision"]) == num_depthwise
    assert {dtype for _idx, dtype in overrides["layer_precision"]} == {trt.DataType.FLOAT}


def test_yolov9_adds_nms_for_ultralytics_exports() -> None:
    """The yolov9u models export a raw output0 head, so YOLOv9 builds must add EfficientNMS."""
    assert nms_build_hook in YOLOv9._build_hooks


def test_rfdetr_class_map_horse(build_model_engine, images) -> None:
    """COCO91_TO_COCO80 turns RF-DETR's horse (19) into the 80-class index (17)."""
    engine = build_model_engine(RFDETR, DATA_DIR / "rfdetr" / "rfdetr_n_384.onnx")
    model = RFDETR(engine, class_map=COCO91_TO_COCO80, warmup=False)
    classes = {cls_id for _bbox, _score, cls_id in model.end2end(images["horse"].array)}
    assert images["horse"].gt_det_classes[0] in classes
    assert 19 not in classes


@pytest.mark.parametrize("name", ["street.jpg", "cars.jpeg"])
def test_rtdetrv3_large_image_boxes(build_model_engine, name) -> None:
    """RT-DETRv3 boxes on large images are real boxes, not garbage clipped to the image edge."""
    engine = build_model_engine(RTDETRv3, DATA_DIR / "rtdetrv3" / "rtdetrv3_r18_640.onnx")
    model = RTDETRv3(engine, warmup=False)
    image = cv2.imread(str(DATA_DIR / name))
    height, width = image.shape[:2]

    detections = model.end2end(image)
    assert detections
    for (x1, y1, x2, y2), _score, _cls_id in detections:
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
        # a garbage box would be clipped to a sliver or to the whole image corner
        assert (x2 - x1) * (y2 - y1) < 0.5 * width * height


@pytest.mark.parametrize(
    ("model_cls", "onnx_path"),
    [
        pytest.param(DEIM, DATA_DIR / "deim" / "deim_dfine_n_640.onnx", id="deim"),
        pytest.param(DEIMv2, DATA_DIR / "deimv2" / "deimv2_n_640.onnx", id="deimv2"),
    ],
)
def test_deim_boxes_non_square_match_onnxruntime(build_model_engine, model_cls, onnx_path) -> None:
    """DEIM boxes on a non-square image match onnxruntime fed the upstream stretched input."""
    engine = build_model_engine(model_cls, onnx_path, opt_level=3)
    model = model_cls(engine, warmup=False, conf_thres=0.5)
    image = cv2.imread(str(DATA_DIR / "horse.jpg"))
    height, width = image.shape[:2]
    assert height != width

    stretched = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 640))
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    blob = ((stretched.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _labels, boxes, scores = session.run(
        None, {"images": blob, "orig_target_sizes": np.array([[width, height]], np.int64)}
    )
    reference = boxes[0][scores[0] > 0.5]
    assert len(reference) == 1

    detections = model.end2end(image)
    assert detections
    assert max(_iou(bbox, reference[0]) for bbox, _score, _cls in detections) > 0.85


def _write_head_onnx(path, output_name):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["images"], [output_name])],
        "head",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 300, 84])],
        [onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 300, 84])],
    )
    onnx.save(onnx.helper.make_model(graph), path)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("output_name", "adds_nms"),
    [
        pytest.param("output0", True, id="ultralytics-single-head"),
        pytest.param("labels", False, id="original-decoded"),
    ],
)
def test_rtdetr_hook_only_for_ultralytics_exports(tmp_path, output_name, adds_nms) -> None:
    """Only the single-head Ultralytics rtdetr-l/x exports get EfficientNMS added."""
    assert rtdetr_build_hook in RTDETRv1._build_hooks
    assert depthwise_fp32_build_hook in RTDETRv1._build_hooks
    path = tmp_path / "head.onnx"
    _write_head_onnx(path, output_name)
    assert bool(rtdetr_build_hook(onnx=path, imgsz=640)) is adds_nms


def _write_gridsample_onnx(path):
    value = onnx.helper.make_tensor_value_info("value", onnx.TensorProto.FLOAT, [1, 1, 4, 4])
    grid = onnx.helper.make_tensor_value_info("grid", onnx.TensorProto.FLOAT, [1, 2, 2, 2])
    out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 1, 2, 2])
    node = onnx.helper.make_node("GridSample", ["value", "grid"], ["out"], name="gs")
    graph = onnx.helper.make_graph([node], "gs", [value, grid], [out])
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 16)]), path)


@pytest.mark.cpu
def test_gridsample_hook_marks_value_input_as_debug(tmp_path) -> None:
    """The hook turns each GridSample value tensor into a debug tensor, leaving the grid alone."""
    for model_cls in (DFINE, DEIM, DEIMv2):
        assert gridsample_debug_build_hook in model_cls._build_hooks
    # the HGNetv2 backbones of D-FINE/DEIM hit the broken FP16 depthwise kernel too
    for model_cls in (DFINE, DEIM):
        assert depthwise_fp32_build_hook in model_cls._build_hooks
    path = tmp_path / "gs.onnx"
    _write_gridsample_onnx(path)
    network, _, _, _ = read_onnx(path)
    (hook,) = gridsample_debug_build_hook()["hooks"]
    hook(network)
    sample = next(
        network.get_layer(i)
        for i in range(network.num_layers)
        if network.get_layer(i).type == trt.LayerType.GRID_SAMPLE
    )
    assert network.is_debug_tensor(sample.get_input(0))
    assert not network.is_debug_tensor(sample.get_input(1))


def _write_attention_onnx(path):
    shape = [1, 4, 4]
    info = onnx.helper.make_tensor_value_info
    names = ["blocks.0/attn/MatMul", "blocks.0/attn/Softmax", "blocks.0/attn/MatMul_1", "mlp/MatMul"]
    nodes = [
        onnx.helper.make_node("MatMul", ["q", "k"], ["s"], name=names[0]),
        onnx.helper.make_node("Softmax", ["s"], ["p"], name=names[1], axis=-1),
        onnx.helper.make_node("MatMul", ["p", "v"], ["a"], name=names[2]),
        onnx.helper.make_node("MatMul", ["a", "v"], ["out"], name=names[3]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "attn",
        [info(n, onnx.TensorProto.FLOAT, shape) for n in ("q", "k", "v")],
        [info("out", onnx.TensorProto.FLOAT, shape)],
    )
    onnx.save(onnx.helper.make_model(graph), path)


@pytest.mark.cpu
def test_vit_attention_hook_pins_only_qk_matmul_and_softmax(tmp_path) -> None:
    """Only the attention logits (QK matmul and softmax) are pinned, not the AV or MLP matmuls."""
    assert vit_attention_fp32_build_hook in DEIMv2._build_hooks
    path = tmp_path / "attn.onnx"
    _write_attention_onnx(path)
    overrides = vit_attention_fp32_build_hook(onnx=path)
    network, _, _, _ = read_onnx(path)
    pinned = {network.get_layer(idx).name for idx, _dtype in overrides["layer_precision"]}
    assert pinned == {"blocks.0/attn/MatMul", "blocks.0/attn/Softmax"}
    assert {dtype for _idx, dtype in overrides["layer_precision"]} == {trt.DataType.FLOAT}
    assert overrides["prefer_precision_constraints"] is True
    assert vit_attention_fp32_build_hook(onnx=path, strongly_typed=True) == {}
