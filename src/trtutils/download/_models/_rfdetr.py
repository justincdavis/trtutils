# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from trtutils._log import LOG
from trtutils.download._tools import (
    get_weights_cache_dir,
    handle_imgsz,
    run_cmd,
    run_uv_pip_install,
)

if TYPE_CHECKING:
    from pathlib import Path


def export_rfdetr(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
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
        LOG.warning("RF-DETR is a Apache-2.0 licensed model, be aware of license restrictions")
    rfdetr_imgszs = {
        "rfdetr_n": 384,
        "rfdetr_s": 512,
        "rfdetr_m": 576,
    }
    required_imgsz = rfdetr_imgszs.get(model)
    if required_imgsz is None:
        err_msg = f"RF-DETR does not support model {model}"
        raise ValueError(err_msg)
    imgsz = handle_imgsz(imgsz, required_imgsz, model, enforce=True, adjust_div=32)

    run_uv_pip_install(
        directory,
        bin_path.parent,
        "rfdetr",
        no_cache=no_uv_cache,
        verbose=verbose,
    )

    # Handle caching of rfdetr downloaded .pth weights
    weights_cache_dir = get_weights_cache_dir()
    cached_pth_file = weights_cache_dir / config["weights"]
    target_pth_file = directory / config["weights"]

    # Check if .pth file exists in cache
    if cached_pth_file.exists() and not no_cache:
        LOG.info(f"Using cached rfdetr weights: {config['weights']}")
        shutil.copy(cached_pth_file, target_pth_file)

    program = f"""
import rfdetr
model = rfdetr.{config["class"]}(resolution={imgsz})
model.export(
    opset_version={opset},
    simplify=True,
)
    """
    run_cmd(
        [
            python_path,
            "-c",
            program,
        ],
        cwd=directory,
        verbose=verbose,
    )

    # Cache the .pth file if it was downloaded and not already cached
    if target_pth_file.exists() and not cached_pth_file.exists():
        shutil.copy(target_pth_file, cached_pth_file)
        LOG.info(f"Cached rfdetr weights: {config['weights']}")

    model_path = directory / "output" / "inference_model.sim.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path
