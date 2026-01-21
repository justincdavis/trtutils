# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# mypy: disable-error-code="misc,no-any-return"
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from trtutils.download import _download as dl  # noqa: PLC2701
from trtutils.download import download

from .common import MODEL_CONFIGS, TEST_MODELS

if TYPE_CHECKING:
    from collections.abc import Generator


def _expected_cache_file(model: str) -> Path | None:
    config = None
    for model_set in MODEL_CONFIGS.values():
        if model in model_set:
            config = model_set[model]
            break
    if config is None:
        return None

    weights_cache = dl._get_weights_cache_dir()  # noqa: SLF001
    if "weights" in config:
        return weights_cache / config["weights"]
    if config.get("url") == "ultralytics":
        return weights_cache / f"ultralytics_{config['name']}.pt"
    if "id" in config:
        extension = config.get("extension", "").strip(".")
        suffix = f".{extension}" if extension else ""
        return weights_cache / f"gdown_{config['id']}_{config['name']}{suffix}"
    url = config.get("url")
    if url:
        filename = url.rstrip("/").split("/")[-1]
        return weights_cache / f"wget_{filename}"
    return None


@pytest.fixture(scope="session")
def cache_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Provide an isolated cache home directory for downloads.

    Returns
    -------
    Path
        The temporary cache home directory.

    """
    return tmp_path_factory.mktemp("trtutils_cache_home")


@pytest.fixture(autouse=True)
def patch_home(monkeypatch: pytest.MonkeyPatch, cache_home: Path) -> Generator[None, None, None]:
    """
    Patch Path.home to use the temporary cache directory.

    Yields
    ------
    None
        Control back to the test.

    """
    dl._get_cache_dir.cache_clear()  # noqa: SLF001
    dl._get_repo_cache_dir.cache_clear()  # noqa: SLF001
    dl._get_weights_cache_dir.cache_clear()  # noqa: SLF001
    monkeypatch.setattr(Path, "home", lambda: cache_home)
    yield
    dl._get_cache_dir.cache_clear()  # noqa: SLF001
    dl._get_repo_cache_dir.cache_clear()  # noqa: SLF001
    dl._get_weights_cache_dir.cache_clear()  # noqa: SLF001


@pytest.mark.parametrize("model", TEST_MODELS)
def test_download_all_models_real_downloads(model: str, tmp_path: Path) -> None:
    """Download models without cache and verify outputs."""
    output = tmp_path / f"{model}.onnx"

    download(model, output, accept=True)

    assert output.exists()
    cache_file = _expected_cache_file(model)
    if cache_file is not None:
        assert cache_file.exists()
