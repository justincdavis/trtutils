# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Download test fixtures."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from trtutils.download import get_supported_models, load_model_configs
from trtutils.download._models import EXPORTERS

# (model, EXPORTERS key) for one model of every exporter
ROUTING_CASES = [
    pytest.param("yolov8n", "ultralytics", id="ultralytics-yolov8"),
    pytest.param("yolov11n", "ultralytics", id="ultralytics-yolov11"),
    pytest.param("yolov7t", "yolov7", id="yolov7"),
    pytest.param("yolov9t", "yolov9", id="yolov9"),
    pytest.param("yolov10n", "yolov10", id="yolov10"),
    pytest.param("yolov12n", "yolov12", id="yolov12"),
    pytest.param("yolov13n", "yolov13", id="yolov13"),
    pytest.param("yoloxn", "yolox", id="yolox"),
    pytest.param("rtdetrv1_r18", "rtdetrv1", id="rtdetrv1"),
    pytest.param("rtdetrv2_r18", "rtdetrv2", id="rtdetrv2"),
    pytest.param("rtdetrv3_r18", "rtdetrv3", id="rtdetrv3"),
    pytest.param("dfine_n", "dfine", id="dfine"),
    pytest.param("deim_dfine_n", "deim", id="deim"),
    pytest.param("deimv2_atto", "deimv2", id="deimv2"),
    pytest.param("rfdetr_n", "rfdetr", id="rfdetr"),
    pytest.param("alexnet", "torchvision", id="torchvision"),
    pytest.param("deit_tiny", "timm", id="timm-deit"),
    pytest.param("deit3_small", "timm", id="timm-deit3"),
    pytest.param("depth_anything_v1_small", "depth_anything_v1", id="depth_anything_v1"),
    pytest.param("depth_anything_v2_small", "depth_anything_v2", id="depth_anything"),
    pytest.param("depth_anything_v3_mono_large", "depth_anything_v3", id="depth_anything_v3"),
    pytest.param("dinov3_vits16", "timm_features", id="timm-features-dinov3"),
    pytest.param("dinov3_vith16plus", "timm_features", id="timm-features-dinov3-vith16plus"),
    pytest.param("dinov3_convnext_tiny", "timm_features", id="timm-features-dinov3-convnext"),
    pytest.param("hoi_detr_vitl", "hoi_detr", id="hoi_detr"),
    pytest.param("hands23_x101", "hands23", id="hands23"),
    pytest.param("ppocrv4_det", "ppocr", id="ppocr_det"),
    pytest.param("ppocrv4_rec", "ppocr", id="ppocr_rec"),
    pytest.param("yolov8n-seg", "ultralytics", id="ultralytics-yolov8-seg"),
    pytest.param("yolov8n-cls", "ultralytics", id="ultralytics-yolov8-cls"),
    pytest.param("yolov8n-pose", "ultralytics", id="ultralytics-yolov8-pose"),
    pytest.param("yolov8n-obb", "ultralytics", id="ultralytics-yolov8-obb"),
    pytest.param("yolov11n-seg", "ultralytics", id="ultralytics-yolov11-seg"),
    pytest.param("yolov11n-cls", "ultralytics", id="ultralytics-yolov11-cls"),
    pytest.param("yolov11n-pose", "ultralytics", id="ultralytics-yolov11-pose"),
    pytest.param("yolov11n-obb", "ultralytics", id="ultralytics-yolov11-obb"),
    pytest.param("yolov26n-seg", "ultralytics", id="ultralytics-yolov26-seg"),
    pytest.param("yolov26n-cls", "ultralytics", id="ultralytics-yolov26-cls"),
    pytest.param("yolov26n-pose", "ultralytics", id="ultralytics-yolov26-pose"),
    pytest.param("yolov26n-obb", "ultralytics", id="ultralytics-yolov26-obb"),
]


@contextmanager
def patch_exporter(name, return_value):
    """Swap the function of EXPORTERS[name] for a mock, keeping its capability flags."""
    mock = MagicMock(return_value=return_value)
    with patch.dict(EXPORTERS, {name: EXPORTERS[name]._replace(fn=mock)}):
        yield mock


@pytest.fixture(scope="session")
def model_configs():
    """All model configurations loaded from JSON files."""
    return load_model_configs()


@pytest.fixture(scope="session")
def supported_models():
    """All supported model names."""
    return get_supported_models()


@pytest.fixture
def fake_venv(tmp_path):
    """Fake venv layout: touched python binary, bin dir, and a placeholder ONNX file."""
    fake_python = tmp_path / ".venv" / "bin" / "python"
    fake_bin = tmp_path / ".venv" / "bin"
    fake_python.parent.mkdir(parents=True, exist_ok=True)
    fake_python.touch()
    fake_output = tmp_path / "model.onnx"
    fake_output.touch()
    return fake_python, fake_bin, fake_output


@pytest.fixture
def patched_yolov10_export(fake_venv):
    """Patch make_venv and the yolov10 exporter so download_model('yolov10n', ...) is a no-op."""
    fake_python, fake_bin, fake_output = fake_venv
    with patch(
        "trtutils.download._download.make_venv",
        return_value=(fake_python, fake_bin),
    ), patch_exporter("yolov10", fake_output) as mock_export:
        yield mock_export
