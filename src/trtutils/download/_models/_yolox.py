# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from trtutils._log import LOG
from trtutils.download._tools import (
    _640,
    get_patches_dir,
    git_clone,
    handle_imgsz,
    run_cmd,
    run_download,
    run_patch,
    run_uv_pip_install,
)

if TYPE_CHECKING:
    from pathlib import Path


def export_yolox(
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
        LOG.warning("YOLOX is a Apache-2.0 licensed model, be aware of license restrictions")
    imgsz = handle_imgsz(imgsz, _640, "YOLOX")
    git_clone(
        "https://github.com/Megvii-BaseDetection/YOLOX",
        directory,
        "6ddff4824372906469a7fae2dc3206c7aa4bbaee",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolox_dir = directory / "YOLOX"
    run_patch(
        yolox_dir,
        str((get_patches_dir() / "yolox_requirements.patch").resolve()),
        "requirements.txt",
        verbose=verbose,
    )
    run_patch(
        yolox_dir,
        str((get_patches_dir() / "yolox_export_onnx.patch").resolve()),
        "tools/export_onnx.py",
        verbose=verbose,
    )
    run_uv_pip_install(
        yolox_dir,
        bin_path.parent,
        model="yolox",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(yolox_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    # Use exp_file instead of name to avoid module import issues
    exp_file = yolox_dir / "exps" / "default" / f"{config['name']}.py"
    # Set PYTHONPATH so yolox imports work without installing the package
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{yolox_dir}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(yolox_dir)
    run_cmd(
        [
            python_path,
            "tools/export_onnx.py",
            "--output-name",
            f"{config['name']}.onnx",
            "-f",
            str(exp_file),
            "--ckpt",
            config["name"] + ".pth",
            "--opset",
            str(opset),
            "--imgsz",
            str(imgsz),
            str(imgsz),
            "--decode_in_inference",
            "--no-onnxsim",  # Disable onnxslim due to compatibility issues with newer onnx
        ],
        cwd=yolox_dir,
        env=env,
        verbose=verbose,
    )
    model_path = yolox_dir / f"{config['name']}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path
