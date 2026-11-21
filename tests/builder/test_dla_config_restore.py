# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Mock-based tests that can_run_on_dla restores the caller's config; no Jetson required."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from trtutils.builder._dla import can_run_on_dla
from trtutils.compat._libs import trt


def _mock_config(*, side_effect: Exception | None = None) -> MagicMock:
    """Build a config at the unset default (GPU device, DLA core -1)."""
    config = MagicMock()
    config.default_device_type = trt.DeviceType.GPU
    config.DLA_core = -1
    # cover both the new and the legacy TensorRT spelling
    config.can_run_on_DLA.side_effect = side_effect
    config.canRunOnDLA.side_effect = side_effect
    return config


def _mock_network(num_layers: int = 2) -> MagicMock:
    network = MagicMock(spec=trt.INetworkDefinition)
    network.num_layers = num_layers
    return network


def test_can_run_on_dla_restores_unset_default() -> None:
    """The unset default (DLA core -1, default device type) is restored, not left at DLA 0."""
    config = _mock_config()
    can_run_on_dla(_mock_network(), config=config)
    assert config.default_device_type == trt.DeviceType.GPU
    assert config.DLA_core == -1


def test_can_run_on_dla_restores_config_on_error() -> None:
    """The config is restored even when the per-layer check raises."""
    config = _mock_config(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        can_run_on_dla(_mock_network(), config=config)
    assert config.default_device_type == trt.DeviceType.GPU
    assert config.DLA_core == -1
