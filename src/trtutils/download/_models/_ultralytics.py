# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from trtutils._log import LOG
from trtutils.download._tools import (
    _640,
    get_weights_cache_dir,
    handle_imgsz,
    run_cmd,
    run_uv_pip_install,
)

if TYPE_CHECKING:
    from pathlib import Path


def export_ultralytics(
    directory: Path,
    config: dict[str, str],
    python_path: Path,  # noqa: ARG001
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None = None,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning(
            "Ultralytics is a AGPL-3.0 and commercial licensed model, be aware of license restrictions"
        )
    imgsz = handle_imgsz(imgsz, _640, "Ultralytics")
    run_uv_pip_install(
        directory,
        bin_path.parent,
        "ultralytics",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    modelname = f"model={config['name']}"

    # Handle caching of ultralytics downloaded .pt weights
    weights_cache_dir = get_weights_cache_dir()
    pt_filename = config["name"] + ".pt"
    cache_key = f"ultralytics_{pt_filename}"
    cached_pt_file = weights_cache_dir / cache_key
    target_pt_file = directory / pt_filename

    # Check if .pt file exists in cache
    if cached_pt_file.exists() and not no_cache:
        LOG.info(f"Using cached ultralytics weights: {pt_filename}")
        shutil.copy(cached_pt_file, target_pt_file)
        modelname = f"model={pt_filename}"

    run_cmd(
        [
            str(bin_path / "yolo"),
            "export",
            modelname,
            "format=onnx",
            f"opset={opset}",
            f"imgsz={imgsz}",
            "simplify=True",
        ],
        cwd=directory,
        verbose=verbose,
    )

    # Cache the .pt file if it was downloaded and not already cached
    if target_pt_file.exists() and not cached_pt_file.exists():
        shutil.copy(target_pt_file, cached_pt_file)
        LOG.info(f"Cached ultralytics weights: {pt_filename}")

    model_path = directory / (config["name"] + ".onnx")
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path
