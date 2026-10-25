# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/download/_download.py -- config loading and model listing."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cpu

_EXPECTED_FAMILIES = [
    "yolov3",
    "yolov5",
    "yolov7",
    "yolov8",
    "yolov9",
    "yolov10",
    "yolov11",
    "yolov12",
    "yolov13",
    "yolov26",
    "yolox",
    "rtdetrv1",
    "rtdetrv2",
    "rtdetrv3",
    "dfine",
    "deim",
    "deimv2",
    "rfdetr",
    "torchvision_classifier",
    "depth_anything_v1",
    "depth_anything_v2",
    "depth_anything_v3",
    "hoi_detr",
    "hands23",
    "ppocr_det",
    "ppocr_rec",
    "yolov8_seg",
    "yolov8_cls",
    "yolov8_pose",
    "yolov8_obb",
    "yolov11_seg",
    "yolov11_cls",
    "yolov11_pose",
    "yolov11_obb",
    "yolov26_seg",
    "yolov26_cls",
    "yolov26_pose",
    "yolov26_obb",
]

_EXPECTED_MODELS = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolov10n",
    "yolov11n",
    "yolov12n",
    "yolov13n",
    "yolov26n",
    "rtdetrv1_r18",
    "rtdetrv2_r18",
    "rtdetrv3_r18",
    "dfine_n",
    "deim_dfine_n",
    "deimv2_atto",
    "rfdetr_n",
    "rfdetr_s",
    "rfdetr_m",
    "yoloxn",
    "yolov3tu",
    "alexnet",
    "resnet18",
    "vgg11",
    "depth_anything_v1_small",
    "depth_anything_v2_small",
    "depth_anything_v3_mono_large",
    "hoi_detr_vitl",
    "hands23_x101",
    "ppocrv4_det",
    "ppocrv4_rec",
    "yolov8n-seg",
    "yolov8n-cls",
    "yolov8n-pose",
    "yolov8n-obb",
    "yolov11n-seg",
    "yolov11n-cls",
    "yolov11n-pose",
    "yolov11n-obb",
    "yolov26n-seg",
    "yolov26n-cls",
    "yolov26n-pose",
    "yolov26n-obb",
]


def test_config_structure(model_configs) -> None:
    """Every family is a non-empty dict-of-dicts with string keys."""
    for family_name, family in model_configs.items():
        assert isinstance(family_name, str)
        assert isinstance(family, dict)
        assert len(family) > 0, f"Family {family_name} is empty"
        for model_name, config in family.items():
            assert isinstance(model_name, str)
            assert isinstance(config, dict)


@pytest.mark.parametrize("family", _EXPECTED_FAMILIES)
def test_family_loaded(model_configs, family) -> None:
    """Each expected family exists in the loaded configs."""
    assert family in model_configs


@pytest.mark.parametrize("model_name", _EXPECTED_MODELS)
def test_supported_model_present(supported_models, model_name) -> None:
    """Each expected model name is present in the supported list."""
    assert model_name in supported_models


def test_supported_models_unique(supported_models) -> None:
    """get_supported_models returns no duplicate names."""
    assert len(supported_models) == len(set(supported_models))


@pytest.mark.parametrize(
    "version",
    [
        pytest.param("yolov8", id="yolov8"),
        pytest.param("yolov11", id="yolov11"),
        pytest.param("yolov26", id="yolov26"),
    ],
)
def test_task_config_export_args(model_configs, version) -> None:
    """Task heads export at their trained size, and YOLOv26 task heads export raw (no end2end)."""
    for task, imgsz in [("seg", None), ("pose", None), ("obb", "1024"), ("cls", "224")]:
        for config in model_configs[f"{version}_{task}"].values():
            assert config.get("imgsz") == imgsz
            raw_head = version == "yolov26" and task != "cls"
            assert config.get("end2end") == ("False" if raw_head else None)
