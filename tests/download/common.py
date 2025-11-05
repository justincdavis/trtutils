# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import tempfile
from pathlib import Path

from trtutils.download import download


def download_with_args(
    model: str,
    *args,
    **kwargs,
) -> None:
    """Download a model with the given arguments."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "model.onnx"
        download(
            model,
            output,
            *args,
            **kwargs,
            accept=True,
        )
        assert output.exists()
