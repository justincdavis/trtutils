# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for download config loading and model listing."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# TestLoadModelConfigs
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestLoadModelConfigs:
    """Tests for load_model_configs()."""

    def test_returns_dict(self, model_configs: dict) -> None:
        """load_model_configs returns a dict."""
        assert isinstance(model_configs, dict)

    def test_all_config_files_loaded(self, model_configs: dict) -> None:
        """All 20 JSON config files should be loaded as top-level keys."""
        # There are 20 config JSON files in configs/
        assert len(model_configs) >= 20

    def test_config_structure(self, model_configs: dict) -> None:
        """Each family entry maps model names to dicts with string values."""
        for family_name, family in model_configs.items():
            assert isinstance(family_name, str)
            assert isinstance(family, dict), f"Family {family_name} is not a dict"
            for model_name, config in family.items():
                assert isinstance(model_name, str), (
                    f"Model name {model_name!r} in {family_name} is not a str"
                )
                assert isinstance(config, dict), (
                    f"Config for {model_name} in {family_name} is not a dict"
                )

    def test_caching(self) -> None:
        """Second call to load_model_configs returns the same object (lru_cache)."""
        from trtutils.download import load_model_configs

        a = load_model_configs()
        b = load_model_configs()
        assert a is b

    def test_no_empty_families(self, model_configs: dict) -> None:
        """Every family has at least one model."""
        for family_name, family in model_configs.items():
            assert len(family) > 0, f"Family {family_name} is empty"

    @pytest.mark.parametrize(
        "family",
        [
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
        ],
    )
    def test_family_loaded(self, model_configs: dict, family: str) -> None:
        """Each expected family exists in loaded configs."""
        assert family in model_configs, f"Family {family!r} not found in configs"

    def test_config_values_have_name_key(self, model_configs: dict) -> None:
        """Most model configs should have a 'name' key."""
        # Not all configs have 'name' (rfdetr uses 'class'/'weights'),
        # but check at least some do.
        has_name = 0
        total = 0
        for family in model_configs.values():
            for config in family.values():
                total += 1
                if "name" in config:
                    has_name += 1
        assert has_name > 0, "No model configs have a 'name' key"


# ---------------------------------------------------------------------------
# TestGetSupportedModels
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestGetSupportedModels:
    """Tests for get_supported_models()."""

    def test_returns_list(self, supported_models: list) -> None:
        """get_supported_models returns a list."""
        assert isinstance(supported_models, list)

    def test_all_elements_are_strings(self, supported_models: list) -> None:
        """All model names are strings."""
        for name in supported_models:
            assert isinstance(name, str)

    def test_no_duplicates(self, supported_models: list) -> None:
        """All model names are unique."""
        assert len(supported_models) == len(set(supported_models))

    def test_count_at_least_100(self, supported_models: list) -> None:
        """At least 100 models should be supported."""
        assert len(supported_models) >= 100

    def test_caching(self) -> None:
        """lru_cache returns the same object on second call."""
        from trtutils.download import get_supported_models

        a = get_supported_models()
        b = get_supported_models()
        assert a is b

    @pytest.mark.parametrize(
        "model_name",
        [
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
        ],
    )
    def test_specific_model_present(self, supported_models: list, model_name: str) -> None:
        """Known model names should be present in the supported list."""
        assert model_name in supported_models, f"{model_name!r} not in supported models"

    def test_known_models_present(self, supported_models: list) -> None:
        """Spot check several well-known models."""
        for name in ["yolov8n", "rfdetr_n", "deimv2_atto", "resnet50"]:
            assert name in supported_models
