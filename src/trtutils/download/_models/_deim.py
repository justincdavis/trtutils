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


def export_deim(
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
        LOG.warning("DEIM is a Apache-2.0 licensed model, be aware of license restrictions")
    imgsz = handle_imgsz(imgsz, _640, "DEIM", enforce=True)
    git_clone(
        "https://github.com/Intellindust-AI-Lab/DEIM",
        directory,
        "8f28fe63cca4bd2a0f4abaf9b0814b69d5abb658",
        no_cache=no_cache,
        verbose=verbose,
    )
    deim_dir = directory / "DEIM"
    run_uv_pip_install(
        deim_dir,
        bin_path.parent,
        "deim",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(deim_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    run_patch(
        deim_dir,
        str((get_patches_dir() / "deim_export_onnx.patch").resolve()),
        "tools/deployment/export_onnx.py",
        verbose=verbose,
    )
    config_folder = "deim_dfine"
    if "rtdetrv2" in config["name"]:
        config_folder = "deim_rtdetrv2"
    run_cmd(
        [
            python_path,
            "tools/deployment/export_onnx.py",
            "-c",
            str(Path("configs") / config_folder / config["config"]),
            "-r",
            config["name"] + ".pth",
            "--opset",
            str(opset),
            "--imgsz",
            str(imgsz),
            "--simplify",
        ],
        cwd=deim_dir,
        verbose=verbose,
    )
    model_path = deim_dir / f"{config['name']}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def export_deimv2(
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
        LOG.warning("DEIMv2 is a Apache-2.0 licensed model, be aware of license restrictions")
    deimv2_imgszs = {
        "deimv2_atto": 320,
        "deimv2_femto": 416,
        "deimv2_pico": 640,
        "deimv2_n": 640,
        "deimv2_s": 640,
        "deimv2_m": 640,
        "deimv2_l": 640,
        "deimv2_x": 640,
    }
    required_imgsz = deimv2_imgszs.get(model)
    if required_imgsz is None:
        err_msg = f"DEIMv2 does not support model {model}"
        raise ValueError(err_msg)
    imgsz = handle_imgsz(imgsz, required_imgsz, model, enforce=True)
    git_clone(
        "https://github.com/Intellindust-AI-Lab/DEIMv2",
        directory,
        "19d5b19a58c229dd7ad5f079947bbe398e005d01",
        no_cache=no_cache,
        verbose=verbose,
    )
    deim_dir = directory / "DEIMv2"
    run_uv_pip_install(
        deim_dir,
        bin_path.parent,
        "deimv2",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(deim_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    config_folder = "deimv2"
    run_cmd(
        [
            python_path,
            "tools/deployment/export_onnx.py",
            "-c",
            str(Path("configs") / config_folder / config["config"]),
            "-r",
            config["name"] + ".pth",
            "--opset",
            str(opset),
            "--simplify",
        ],
        cwd=deim_dir,
        verbose=verbose,
    )
    model_path = deim_dir / f"{config['name']}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path
