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
    "depth_anything_v2",
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
    "depth_anything_v2_small",
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
