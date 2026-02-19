# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

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


def export_yolov7(
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
        LOG.warning("YOLOv7 is a GPL-3.0 licensed model, be aware of license restrictions")
    imgsz = handle_imgsz(imgsz, _640, "YOLOv7")
    git_clone(
        "https://github.com/WongKinYiu/yolov7",
        directory,
        "a207844b1ce82d204ab36d87d496728d3d2348e7",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolov7_dir = directory / "yolov7"
    run_uv_pip_install(
        yolov7_dir,
        bin_path.parent,
        "yolov7",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(yolov7_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    run_patch(
        yolov7_dir,
        str((get_patches_dir() / "yolov7_export.patch").resolve()),
        "export.py",
        verbose=verbose,
    )
    run_cmd(
        [
            python_path,
            "export.py",
            "--weights",
            config["name"] + ".pt",
            "--grid",
            "--end2end",
            "--simplify",
            "--img-size",
            str(imgsz),
            "--opset",
            str(opset),
        ],
        cwd=yolov7_dir,
        verbose=verbose,
    )
    model_path = yolov7_dir / (config["name"] + ".onnx")

    # patch names
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def export_yolov9(
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
        LOG.warning("YOLOv9 is a GPL-3.0 licensed model, be aware of license restrictions")
    imgsz = handle_imgsz(imgsz, _640, "YOLOv9")
    git_clone(
        "https://github.com/WongKinYiu/yolov9",
        directory,
        "5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolov9_dir = directory / "yolov9"
    run_patch(
        yolov9_dir,
        str((get_patches_dir() / "yolov9_export.patch").resolve()),
        "export.py",
        verbose=verbose,
    )
    run_uv_pip_install(
        yolov9_dir,
        bin_path.parent,
        "yolov9",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(yolov9_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    run_cmd(
        [
            python_path,
            "export.py",
            "--weights",
            config["name"] + ".pt",
            "--include",
            "onnx_end2end",
            "--simplify",
            "--img-size",
            str(imgsz),
            str(imgsz),
            "--opset",
            str(opset),
        ],
        cwd=yolov9_dir,
        verbose=verbose,
    )
    model_path = yolov9_dir / (config["name"] + "-end2end.onnx")
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def export_yolov10(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,  # noqa: ARG001
    opset: int,
    imgsz: int | None = None,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning("YOLOv10 is a AGPL-3.0 licensed model, be aware of license restrictions")
    imgsz = handle_imgsz(imgsz, _640, "YOLOv10")
    git_clone(
        "https://github.com/THU-MIG/yolov10",
        directory,
        "453c6e38a51e9d1d5a2aa5fb7f1014a711913397",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolov10_dir = directory / "yolov10"
    run_uv_pip_install(
        yolov10_dir,
        bin_path.parent,
        "yolov10",
        packages=["."],
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(yolov10_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    run_cmd(
        [
            str(bin_path / "yolo"),
            "export",
            f"model={config['name'] + '.pt'}",
            "format=onnx",
            f"opset={opset}",
            f"imgsz={imgsz}",
        ],
        cwd=yolov10_dir,
        verbose=verbose,
    )
    return yolov10_dir / (config["name"] + ".onnx")


def export_yolov12(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,  # noqa: ARG001
    opset: int,
    imgsz: int | None = None,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning("YOLOv12 is a AGPL-3.0 licensed model, be aware of license restrictions")
    imgsz = handle_imgsz(imgsz, _640, "YOLOv12")
    git_clone(
        "https://github.com/sunsmarterjie/yolov12",
        directory,
        "3bca22b336e96cfdabfec4c062b84eef210e9563",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolov12_dir = directory / "yolov12"
    run_uv_pip_install(
        yolov12_dir,
        bin_path.parent,
        "yolov12",
        packages=["."],
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(yolov12_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    run_cmd(
        [
            str(bin_path / "yolo"),
            "export",
            f"model={config['name'] + '.pt'}",
            "format=onnx",
            f"opset={opset}",
            f"imgsz={imgsz}",
            "simplify",
        ],
        cwd=yolov12_dir,
        verbose=verbose,
    )
    return yolov12_dir / (config["name"] + ".onnx")


def export_yolov13(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,  # noqa: ARG001
    opset: int,
    imgsz: int | None = None,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning("YOLOv13 is a AGPL-3.0 licensed model, be aware of license restrictions")
    imgsz = handle_imgsz(imgsz, _640, "YOLOv13")
    git_clone(
        "https://github.com/iMoonLab/yolov13",
        directory,
        "d09f2efa512bff266d558b562ae06b41e3af00d8",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolov13_dir = directory / "yolov13"
    run_uv_pip_install(
        yolov13_dir,
        bin_path.parent,
        "yolov13",
        packages=["."],
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(yolov13_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    run_cmd(
        [
            str(bin_path / "yolo"),
            "export",
            f"model={config['name'] + '.pt'}",
            "format=onnx",
            f"opset={opset}",
            f"imgsz={imgsz}",
        ],
        cwd=yolov13_dir,
        verbose=verbose,
    )
    return yolov13_dir / (config["name"] + ".onnx")
