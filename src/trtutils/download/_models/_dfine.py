# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import shutil
from pathlib import Path

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


def export_dfine(
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
        LOG.warning("D-FINE is a Apache-2.0 licensed model, be aware of license restrictions")
    imgsz = handle_imgsz(imgsz, _640, "D-FINE", enforce=True)
    git_clone(
        "https://github.com/Peterande/D-FINE",
        directory,
        "d6694750683b0c7e9f523ba6953d16f112a376ae",
        no_cache=no_cache,
        verbose=verbose,
    )
    dfine_dir = directory / "D-FINE"
    run_uv_pip_install(
        dfine_dir,
        bin_path.parent,
        "dfine",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(dfine_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    run_patch(
        dfine_dir,
        str((get_patches_dir() / "dfine_export_onnx.patch").resolve()),
        "tools/deployment/export_onnx.py",
        verbose=verbose,
    )
    run_cmd(
        [
            python_path,
            "tools/deployment/export_onnx.py",
            "-c",
            str(Path("configs") / "dfine" / config["config"]),
            "-r",
            config["name"] + ".pth",
            "--opset",
            str(opset),
            "--imgsz",
            str(imgsz),
        ],
        cwd=dfine_dir,
        verbose=verbose,
    )
    model_path = dfine_dir / f"{config['name']}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path
