# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import shutil
from pathlib import Path

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


def _export_dfine_style(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    repo: str,
    commit: str,
    family: str,
    config_dir: str,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    # repos forked from D-FINE share its export script, so only the names differ
    git_clone(repo, directory, commit, no_cache=no_cache, verbose=verbose)
    repo_dir = directory / repo.rsplit("/", 1)[-1]
    run_uv_pip_install(
        repo_dir,
        bin_path.parent,
        family,
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    run_download(repo_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    run_patch(
        repo_dir,
        str((get_patches_dir() / f"{family}_export_onnx.patch").resolve()),
        "tools/deployment/export_onnx.py",
        verbose=verbose,
    )
    run_cmd(
        [
            python_path,
            "tools/deployment/export_onnx.py",
            "-c",
            str(Path("configs") / config_dir / config["config"]),
            "-r",
            config["name"] + ".pth",
            "--opset",
            str(opset),
            "--imgsz",
            str(imgsz),
        ],
        cwd=repo_dir,
        verbose=verbose,
    )
    model_path = repo_dir / f"{config['name']}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


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
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    return _export_dfine_style(
        directory,
        config,
        python_path,
        bin_path,
        model,
        opset,
        handle_imgsz(imgsz, _640, "D-FINE", enforce=True),
        repo="https://github.com/Peterande/D-FINE",
        commit="d6694750683b0c7e9f523ba6953d16f112a376ae",
        family="dfine",
        config_dir="dfine",
        no_cache=no_cache,
        no_uv_cache=no_uv_cache,
        verbose=verbose,
    )


def export_rtdetrv4(
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
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    return _export_dfine_style(
        directory,
        config,
        python_path,
        bin_path,
        model,
        opset,
        handle_imgsz(imgsz, _640, "RT-DETRv4", enforce=True),
        repo="https://github.com/RT-DETRs/RT-DETRv4",
        commit="55fefaaed7efe2a5f72d0a18fd4e05965e35c292",
        family="rtdetrv4",
        config_dir="rtv4",
        no_cache=no_cache,
        no_uv_cache=no_uv_cache,
        verbose=verbose,
    )
