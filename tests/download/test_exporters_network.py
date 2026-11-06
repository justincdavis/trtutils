# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/download -- real downloads and exports per exporter family (opt-in)."""

from __future__ import annotations

import shutil

import onnx
import pytest

from trtutils.download import download
from trtutils.download._make_static import make_static
from trtutils.download._simplify import simplify

pytestmark = [pytest.mark.download, pytest.mark.slow]

# None marks a dimension that must be symbolic
_BATCH = None
_YOLO = {"images": [1, 3, 320, 320]}
_YOLO_DYN = {"images": [_BATCH, 3, None, None]}
_YOLO_B2 = {"images": [2, 3, 320, 320]}

# (model, download kwargs, expected input shapes keyed by input name)
EXPORT_CASES = [
    pytest.param("yolov8n", {"imgsz": 320}, _YOLO, id="ultralytics-static"),
    pytest.param("yolov8n", {"imgsz": 320, "dynamic": True}, _YOLO_DYN, id="ultralytics-dynamic"),
    pytest.param("yolov8n", {"imgsz": 320, "batch": 2}, _YOLO_B2, id="ultralytics-batch2"),
    pytest.param("yolov5nu", {"imgsz": 320}, _YOLO, id="ultralytics-yolov5"),
    pytest.param("yolov3tu", {"imgsz": 320}, _YOLO, id="ultralytics-yolov3"),
    pytest.param("yolov26n", {"imgsz": 320}, _YOLO, id="ultralytics-yolov26"),
    pytest.param("yolov11n-seg", {"imgsz": 320}, _YOLO, id="ultralytics-seg"),
    pytest.param("yolov8n-pose", {"imgsz": 320}, _YOLO, id="ultralytics-pose"),
    pytest.param(
        "yolov26n-obb", {}, {"images": [1, 3, 1024, 1024]}, id="ultralytics-obb-default-imgsz"
    ),
    pytest.param(
        "yolov8n-cls", {}, {"images": [1, 3, 224, 224]}, id="ultralytics-cls-default-imgsz"
    ),
    pytest.param("yolov7t", {"imgsz": 320}, _YOLO, id="yolov7"),
    pytest.param("yolov9t", {"imgsz": 320}, _YOLO, id="yolov9"),
    pytest.param("yolov10n", {"imgsz": 320}, _YOLO, id="yolov10-static"),
    pytest.param("yolov10n", {"imgsz": 320, "dynamic": True}, _YOLO_DYN, id="yolov10-dynamic"),
    pytest.param("yolov10n", {"imgsz": 320, "batch": 2}, _YOLO_B2, id="yolov10-batch2"),
    pytest.param("yolov12n", {"imgsz": 320, "dynamic": True}, _YOLO_DYN, id="yolov12-dynamic"),
    pytest.param("yolov13n", {"imgsz": 320, "batch": 2}, _YOLO_B2, id="yolov13-batch2"),
    pytest.param("yoloxn", {"imgsz": 320}, _YOLO, id="yolox"),
    pytest.param(
        "rtdetrv1_r18",
        {"imgsz": 640},
        {"images": [None, 3, 640, 640], "orig_target_sizes": [None, 2]},
        id="rtdetrv1",
    ),
    pytest.param(
        "rtdetrv2_r18",
        {"imgsz": 640},
        {"images": [1, 3, 640, 640], "orig_target_sizes": [1, 2]},
        id="rtdetrv2",
    ),
    pytest.param(
        "rtdetrv3_r18",
        {"imgsz": 640},
        {"image": [None, 3, 640, 640], "im_shape": [None, 2], "scale_factor": [None, 2]},
        id="rtdetrv3",
    ),
    pytest.param(
        "dfine_n",
        {"imgsz": 640},
        {"images": [None, 3, 640, 640], "orig_target_sizes": [None, 2]},
        id="dfine",
    ),
    pytest.param(
        "deim_dfine_n",
        {"imgsz": 640},
        {"images": [None, 3, 640, 640], "orig_target_sizes": [None, 2]},
        id="deim",
    ),
    pytest.param(
        "deimv2_atto",
        {"imgsz": 320},
        {"images": [None, 3, 320, 320], "orig_target_sizes": [None, 2]},
        id="deimv2",
    ),
    pytest.param("rfdetr_n", {}, {"input": [1, 3, 384, 384]}, id="rfdetr"),
    pytest.param(
        "mobilenet_v3_small",
        {"imgsz": 224},
        {"input": [_BATCH, 3, 224, 224]},
        id="torchvision-default-dynamic",
    ),
    pytest.param(
        "mobilenet_v3_small",
        {"imgsz": 224, "batch": 2},
        {"input": [2, 3, 224, 224]},
        id="torchvision-batch2",
    ),
    pytest.param(
        "depth_anything_v1_small",
        {},
        {"input": [_BATCH, 3, 518, 518]},
        id="depth_anything_v1",
    ),
    pytest.param(
        "depth_anything_v2_small",
        {},
        {"input": [_BATCH, 3, 518, 518]},
        id="depth_anything_v2",
    ),
    pytest.param(
        "depth_anything_v3_mono_large",
        {"imgsz": 252},
        {"input": [_BATCH, 3, 252, 252]},
        id="depth_anything_v3",
    ),
    pytest.param("hands23_x101", {}, {"input": [1, 3, 800, 800]}, id="hands23"),
    pytest.param("ppocrv4_det", {}, {"x": [None, 3, None, None]}, id="ppocr-det"),
    pytest.param("ppocrv4_rec", {}, {"x": [None, 3, None, None]}, id="ppocr-rec"),
]


def _input_shapes(path):
    model = onnx.load(str(path), load_external_data=False)
    return {
        v.name: [
            d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim
        ]
        for v in model.graph.input
    }


def _check_shapes(path, expected) -> None:
    onnx.checker.check_model(str(path))
    actual = _input_shapes(path)
    assert actual.keys() == expected.keys()
    for name, dims in expected.items():
        assert len(actual[name]) == len(dims)
        for got, want in zip(actual[name], dims):
            if want is None:
                assert got is None
            else:
                assert got == want


def _node_count(path) -> int:
    return len(onnx.load(str(path), load_external_data=False).graph.node)


@pytest.mark.parametrize(("model", "kwargs", "expected"), EXPORT_CASES)
def test_export_family(tmp_path, model, kwargs, expected) -> None:
    """A real download + export produces a valid ONNX with the requested input shapes."""
    out = tmp_path / f"{model}.onnx"
    download(model, out, **kwargs)
    _check_shapes(out, expected)


@pytest.fixture(scope="module")
def baseline_onnx(tmp_path_factory):
    """yolov8n exported without any post-processing."""
    out = tmp_path_factory.mktemp("baseline") / "yolov8n.onnx"
    download("yolov8n", out, imgsz=320)
    return out


@pytest.mark.parametrize(
    "tools",
    [
        pytest.param(True, id="default-tools"),
        pytest.param(["polygraphy"], id="polygraphy"),
        pytest.param(["onnxslim"], id="onnxslim"),
        pytest.param(["onnxsim"], id="onnxsim"),
        pytest.param(["onnxslim", "polygraphy"], id="chain-reversed"),
    ],
)
def test_simplify_via_download(tmp_path, baseline_onnx, tools) -> None:
    """download(simplify=...) runs each tool and leaves a valid, no larger ONNX."""
    out = tmp_path / "simplified.onnx"
    download("yolov8n", out, imgsz=320, simplify=tools)
    _check_shapes(out, _YOLO)
    assert _node_count(out) <= _node_count(baseline_onnx)


def test_simplify_standalone_and_unknown_tool(tmp_path, baseline_onnx) -> None:
    """simplify() builds its own venv without a directory and rejects unknown tools."""
    model = tmp_path / "model.onnx"
    shutil.copy(baseline_onnx, model)
    simplify(model, tools=["onnxslim"])
    _check_shapes(model, _YOLO)
    with pytest.raises(ValueError, match="Unknown simplification tools"):
        simplify(model, tools=["not_a_tool"])


@pytest.mark.parametrize(
    ("model", "kwargs", "expected"),
    [
        pytest.param(
            "mobilenet_v3_small",
            {"imgsz": 224},
            {"input": [1, 3, 224, 224]},
            id="torchvision-batch",
        ),
        pytest.param(
            "rtdetrv1_r18",
            {"imgsz": 640},
            {"images": [1, 3, 640, 640], "orig_target_sizes": [1, 2]},
            id="rtdetrv1-batch",
        ),
        pytest.param(
            "ppocrv4_det",
            {},
            {"x": [1, 3, 1, 1]},
            id="ppocr-all-dims",
        ),
    ],
)
def test_make_static_via_download(tmp_path, model, kwargs, expected) -> None:
    """download(make_static=True) fixes every dynamic input dimension to 1."""
    out = tmp_path / f"{model}.onnx"
    download(model, out, make_static=True, **kwargs)
    _check_shapes(out, expected)


def test_make_static_standalone(tmp_path) -> None:
    """make_static() builds its own venv without a directory and fixes dynamic dims."""
    model = tmp_path / "model.onnx"
    download("mobilenet_v3_small", model, imgsz=224)
    make_static(model)
    _check_shapes(model, {"input": [1, 3, 224, 224]})


def test_requirements_export_and_no_cache(tmp_path) -> None:
    """requirements_export writes the venv freeze and no_cache re-fetches the weights."""
    out = tmp_path / "yolov8n.onnx"
    reqs = tmp_path / "requirements.txt"
    download("yolov8n", out, imgsz=320, requirements_export=reqs, no_cache=True, verbose=True)
    _check_shapes(out, _YOLO)
    assert "ultralytics" in reqs.read_text().lower()
