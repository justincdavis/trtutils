# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from trtutils.download._tools import run_download

from ._ppocr_pass import scale_ppocrv5_server_det_neck


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
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    # PP-OCR models are distributed as ready-made ONNX with dynamic input shapes,
    # so there is nothing to export: download the file as-is (the v5 server detector
    # gets its neck rescaled so it builds in fp16)
    run_download(directory, config, python_path, no_cache=no_cache, verbose=verbose)
    model_path = directory / Path(urlsplit(config["url"]).path).name
    if "sha256" in config:
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != config["sha256"]:
            err_msg = f"{model_path.name} has sha256 {digest}, expected {config['sha256']}"
            raise RuntimeError(err_msg)
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    if model == "ppocrv5_server_det":
        scale_ppocrv5_server_det_neck(new_model_path)
    return new_model_path
