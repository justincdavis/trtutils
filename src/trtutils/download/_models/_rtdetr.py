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


def export_rtdetrv1(
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
        LOG.warning("RT-DETRv1 is a Apache-2.0 licensed model, be aware of license restrictions")
    imgsz = handle_imgsz(imgsz, _640, "RT-DETRv1", enforce=True)
    git_clone(
        "https://github.com/lyuwenyu/RT-DETR",
        directory,
        "f9417e3acfa48bcb649e5ec0bc3de1e8677c8961",
        no_cache=no_cache,
        verbose=verbose,
    )
    rtdetr_dir = directory / "RT-DETR" / "rtdetr_pytorch"
    run_uv_pip_install(
        rtdetr_dir,
        bin_path.parent,
        "rtdetrv1",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(rtdetr_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    run_patch(
        rtdetr_dir,
        str((get_patches_dir() / "rtdetrv1_export_onnx.patch").resolve()),
        "tools/export_onnx.py",
        verbose=verbose,
    )
    run_cmd(
        [
            python_path,
            "tools/export_onnx.py",
            "-c",
            str(Path("configs") / "rtdetr" / config["config"]),
            "-r",
            config["name"] + ".pth",
            "--opset",
            str(opset),
            "--imgsz",
            str(imgsz),
        ],
        cwd=rtdetr_dir,
        verbose=verbose,
    )
    model_path = rtdetr_dir / "model.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def export_rtdetrv2(
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
        LOG.warning("RT-DETRv2 is a Apache-2.0 licensed model, be aware of license restrictions")
    imgsz = handle_imgsz(imgsz, _640, "RT-DETRv2", enforce=True)
    git_clone(
        "https://github.com/lyuwenyu/RT-DETR",
        directory,
        "f9417e3acfa48bcb649e5ec0bc3de1e8677c8961",
        no_cache=no_cache,
        verbose=verbose,
    )
    rtdetrv2_dir = directory / "RT-DETR" / "rtdetrv2_pytorch"
    run_uv_pip_install(
        rtdetrv2_dir,
        bin_path.parent,
        "rtdetrv2",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(rtdetrv2_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    run_patch(
        rtdetrv2_dir,
        str((get_patches_dir() / "rtdetrv2_export_onnx.patch").resolve()),
        "tools/export_onnx.py",
        verbose=verbose,
    )
    run_cmd(
        [
            python_path,
            "tools/export_onnx.py",
            "-c",
            str(Path("configs") / "rtdetrv2" / config["config"]),
            "-r",
            config["name"] + ".pth",
            "--opset",
            str(opset),
            "--input_size",
            str(imgsz),
            "--simplify",
        ],
        cwd=rtdetrv2_dir,
        verbose=verbose,
    )
    model_path = rtdetrv2_dir / "model.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def export_rtdetrv3(
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
        LOG.warning("RT-DETRv3 is a Apache-2.0 licensed model, be aware of license restrictions")
    imgsz = handle_imgsz(imgsz, _640, "RT-DETRv3", enforce=True)
    paddle2onnx_max_opset = 16
    if opset > paddle2onnx_max_opset:
        LOG.warning(
            f"RT-DETRv3 only supports opset <{paddle2onnx_max_opset}, using opset {paddle2onnx_max_opset}"
        )
        opset = paddle2onnx_max_opset
    git_clone(
        "https://github.com/clxia12/RT-DETRv3",
        directory,
        "349e7d99a5065e7b684118912e6a74178d4f4625",
        no_cache=no_cache,
        verbose=verbose,
    )
    rtdetrv3_dir = directory / "RT-DETRv3"
    run_uv_pip_install(
        rtdetrv3_dir,
        bin_path.parent,
        "rtdetrv3",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(rtdetrv3_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    run_cmd(
        [
            python_path,
            "tools/export_model.py",
            "-c",
            str(Path("configs") / "rtdetrv3" / config["config"]),
            "-o",
            f"weights={config['name']}.pdparams",
            "use_gpu=False",
            "trt=True",
            "--output_dir",
            "output_weights",
        ],
        cwd=rtdetrv3_dir,
        verbose=verbose,
    )
    run_cmd(
        [
            bin_path / "paddle2onnx",
            "--model_dir",
            f"./output_weights/{config['name']}/",
            "--model_filename",
            "model.pdmodel",
            "--params_filename",
            "model.pdiparams",
            "--opset_version",
            str(opset),
            "--save_file",
            f"{config['name']}.onnx",
        ],
        cwd=rtdetrv3_dir,
        verbose=verbose,
    )
    model_path = rtdetrv3_dir / f"{config['name']}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path
