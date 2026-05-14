# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Download test fixtures."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trtutils.download import get_supported_models, load_model_configs


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
    ):
        yield fake_output
