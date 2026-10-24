# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlsplit

from trtutils._log import LOG
from trtutils.download._tools import run_download


def export_ppocr(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,  # noqa: ARG001
    model: str,
    opset: int,  # noqa: ARG001
    imgsz: int | None = None,  # noqa: ARG001
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,  # noqa: ARG001
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning("PP-OCR is a Apache-2.0 licensed model, be aware of license restrictions")
    # PP-OCR models are distributed as ready-made ONNX with dynamic input shapes,
    # so there is nothing to export: download the file as-is
    run_download(directory, config, python_path, no_cache=no_cache, verbose=verbose)
    model_path = directory / Path(urlsplit(config["url"]).path).name
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path
