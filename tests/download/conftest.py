# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Download test fixtures."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trtutils.download import get_supported_models, load_model_configs

# (model, export function name) for one model of every exporter
ROUTING_CASES = [
    pytest.param("yolov8n", "export_ultralytics", id="ultralytics-yolov8"),
    pytest.param("yolov11n", "export_ultralytics", id="ultralytics-yolov11"),
    pytest.param("yolov7t", "export_yolov7", id="yolov7"),
    pytest.param("yolov9t", "export_yolov9", id="yolov9"),
    pytest.param("yolov10n", "export_yolov10", id="yolov10"),
    pytest.param("yolov12n", "export_yolov12", id="yolov12"),
    pytest.param("yolov13n", "export_yolov13", id="yolov13"),
    pytest.param("yoloxn", "export_yolox", id="yolox"),
    pytest.param("rtdetrv1_r18", "export_rtdetrv1", id="rtdetrv1"),
    pytest.param("rtdetrv2_r18", "export_rtdetrv2", id="rtdetrv2"),
    pytest.param("rtdetrv3_r18", "export_rtdetrv3", id="rtdetrv3"),
    pytest.param("dfine_n", "export_dfine", id="dfine"),
    pytest.param("deim_dfine_n", "export_deim", id="deim"),
    pytest.param("deimv2_atto", "export_deimv2", id="deimv2"),
    pytest.param("rfdetr_n", "export_rfdetr", id="rfdetr"),
    pytest.param("alexnet", "export_torchvision_classifier", id="torchvision"),
    pytest.param("depth_anything_v1_small", "export_depth_anything_v1", id="depth_anything_v1"),
    pytest.param("depth_anything_v2_small", "export_depth_anything_v2", id="depth_anything"),
    pytest.param("depth_anything_v3_mono_large", "export_depth_anything_v3", id="depth_anything_v3"),
    pytest.param("hoi_detr_vitl", "export_hoi_detr", id="hoi_detr"),
    pytest.param("hands23_x101", "export_hands23", id="hands23"),
    pytest.param("ppocrv4_det", "export_ppocr", id="ppocr_det"),
    pytest.param("ppocrv4_rec", "export_ppocr", id="ppocr_rec"),
    pytest.param("yolov8n-seg", "export_ultralytics", id="ultralytics-yolov8-seg"),
    pytest.param("yolov8n-cls", "export_ultralytics", id="ultralytics-yolov8-cls"),
    pytest.param("yolov8n-pose", "export_ultralytics", id="ultralytics-yolov8-pose"),
    pytest.param("yolov8n-obb", "export_ultralytics", id="ultralytics-yolov8-obb"),
    pytest.param("yolov11n-seg", "export_ultralytics", id="ultralytics-yolov11-seg"),
    pytest.param("yolov11n-cls", "export_ultralytics", id="ultralytics-yolov11-cls"),
    pytest.param("yolov11n-pose", "export_ultralytics", id="ultralytics-yolov11-pose"),
    pytest.param("yolov11n-obb", "export_ultralytics", id="ultralytics-yolov11-obb"),
    pytest.param("yolov26n-seg", "export_ultralytics", id="ultralytics-yolov26-seg"),
    pytest.param("yolov26n-cls", "export_ultralytics", id="ultralytics-yolov26-cls"),
    pytest.param("yolov26n-pose", "export_ultralytics", id="ultralytics-yolov26-pose"),
    pytest.param("yolov26n-obb", "export_ultralytics", id="ultralytics-yolov26-obb"),
]


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
    """Patch make_venv and export_yolov10 so download_model('yolov10n', ...) is a no-op pipeline."""
    fake_python, fake_bin, fake_output = fake_venv
    with patch(
        "trtutils.download._download.make_venv",
        return_value=(fake_python, fake_bin),
    ), patch(
        "trtutils.download._download.export_yolov10",
        return_value=fake_output,
    ) as mock_export:
        yield mock_export
