# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S607, S603, S404
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from trtutils._log import LOG


@lru_cache(maxsize=1)
def _get_cache_dir() -> Path:
    """Get or create the cache directory for trtutils downloads."""
    cache_dir = Path.home() / ".cache" / "trtutils"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def _get_repo_cache_dir() -> Path:
    """Get or create the repository cache directory."""
    repo_cache = _get_cache_dir() / "repos"
    repo_cache.mkdir(parents=True, exist_ok=True)
    return repo_cache


@lru_cache(maxsize=1)
def _get_weights_cache_dir() -> Path:
    """Get or create the weights cache directory."""
    weights_cache = _get_cache_dir() / "weights"
    weights_cache.mkdir(parents=True, exist_ok=True)
    return weights_cache


@lru_cache(maxsize=0)
def _get_model_requirements(model: str) -> str:
    """Get the requirements file for the model."""
    file_path = Path(__file__).parent / "requirements" / f"{model}.txt"
    return str(file_path.resolve())


@lru_cache(maxsize=1)
def load_model_configs() -> dict[str, dict[str, dict[str, str]]]:
    configs_dir = Path(__file__).parent / "configs"
    model_configs: dict[str, dict[str, dict[str, str]]] = {}

    for config_path in configs_dir.glob("*.json"):
        model_type = config_path.stem
        try:
            with config_path.open() as f:
                model_configs[model_type] = json.load(f)
        except Exception as e:
            LOG.warning(f"Failed to load configuration file {config_path.name}: {e}")

    return model_configs


def _git_clone(
    url: str,
    directory: Path,
    commit: str | None = None,
    *,
    no_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    # Extract repo name from URL
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    cache_repo_path = _get_repo_cache_dir() / repo_name
    target_path = directory / repo_name

    # Check if repo exists in cache
    if cache_repo_path.exists() and not no_cache:
        # Copy from cache
        LOG.info(f"Using cached repository: {repo_name}")
        shutil.copytree(cache_repo_path, target_path)
    else:
        LOG.info(f"Cloning repository: {repo_name}")
        # Clone fresh
        subprocess.run(
            ["git", "clone", url],
            cwd=directory,
            check=True,
            stdout=subprocess.DEVNULL if not verbose else None,
            stderr=subprocess.STDOUT if not verbose else None,
        )
        # Cache the cloned repo
        shutil.copytree(target_path, cache_repo_path)
        LOG.info(f"Cached repository: {repo_name}")

    if commit:
        subprocess.run(
            ["git", "checkout", commit],
            cwd=target_path,
            check=True,
            stdout=subprocess.DEVNULL if not verbose else None,
            stderr=subprocess.STDOUT if not verbose else None,
        )


def _run_uv_pip_install(
    directory: Path,
    venv_path: Path,
    model: str | None,
    packages: list[str] | None = None,
    *,
    no_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    cmd = ["uv", "pip", "install", "-p", str(venv_path)]
    if model is not None:
        LOG.info(f"Creating venv for model: {model}")
        cmd.extend(
            [
                "-r",
                _get_model_requirements(model),
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
                "--index-strategy",
                "unsafe-best-match",
            ]
        )
    if packages is not None:
        cmd.extend(packages)
    if no_cache:
        cmd.extend(["--no-cache"])
    LOG.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(
        cmd,
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )


def _export_requirements(
    venv_path: Path,
    output_path: Path,
    *,
    verbose: bool | None = None,
) -> None:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        LOG.info(f"Exporting virtual environment requirements to {output_path}")
    with output_path.open("w", encoding="utf-8") as requirements_file:
        subprocess.run(
            ["uv", "pip", "freeze", "-p", str(venv_path)],
            check=True,
            stdout=requirements_file,
            stderr=subprocess.DEVNULL if not verbose else None,
        )


def _make_venv(
    directory: Path,
    *,
    no_cache: bool | None = None,
    verbose: bool | None = None,
) -> tuple[Path, Path]:
    subprocess.run(
        ["uv", "venv", ".venv", "--python=3.10", "--clear"],
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    # Important: do NOT resolve symlinks here; keep .venv/bin/python so uv targets the venv
    bin_path: Path = directory / ".venv" / "bin"
    python_path: Path = bin_path / "python"
    std_packages = ["--upgrade", "pip", "setuptools", "wheel"]
    if no_cache:
        std_packages.append("--no-cache")
    _run_uv_pip_install(
        directory,
        bin_path.parent,
        model=None,
        packages=std_packages,
        verbose=verbose,
    )
    return python_path, bin_path


def _run_download(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    *,
    no_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    weights_cache_dir = _get_weights_cache_dir()

    # Determine the filename based on download method
    if "id" in config:
        filename = config["name"] + "." + config["extension"]
        cache_key = f"gdown_{config['id']}_{filename}"
    # RF-DETR and Ultralytics based models are downloaded via their packages
    # Handle caching inside their own functions
    elif config.get("url") is None or config["url"] == "ultralytics":
        seed = time.monotonic_ns()  # use this to get unique filename
        cache_key = f"unused_{seed}"
        filename = f"unused_{seed}"
    # Handle URL based models
    else:
        filename = config["url"].rstrip("/").split("/")[-1]
        cache_key = f"wget_{filename}"

    cached_file = weights_cache_dir / cache_key
    target_file = directory / filename

    # Check if file exists in cache
    if cached_file.exists() and not no_cache:
        LOG.info(f"Using cached weights: {filename}")
        shutil.copy(cached_file, target_file)
    else:
        LOG.info(f"Downloading weights: {filename}")
        # Download fresh
        if "id" in config:
            subprocess.run(
                [
                    python_path,
                    "-m",
                    "gdown",
                    "--id",
                    config["id"],
                    "-O",
                    filename,
                ],
                cwd=directory,
                check=True,
                stdout=subprocess.DEVNULL if not verbose else None,
                stderr=subprocess.STDOUT if not verbose else None,
            )
        else:
            subprocess.run(
                ["wget", "-nc", config["url"]],
                cwd=directory,
                check=True,
                stdout=subprocess.DEVNULL if not verbose else None,
                stderr=subprocess.STDOUT if not verbose else None,
            )

        # Cache the downloaded file
        shutil.copy(target_file, cached_file)
        LOG.info(f"Cached weights: {filename}")


def _run_patch(
    directory: Path,
    patch_file: str,
    file_to_patch: str,
    *,
    verbose: bool | None = None,
) -> None:
    LOG.info(f"Patching {file_to_patch} with {patch_file}")
    subprocess.run(
        ["patch", file_to_patch, "-i", patch_file],
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )


def _export_yolov7(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning("YOLOv7 is a GPL-3.0 licensed model, be aware of license restrictions")
    _git_clone(
        "https://github.com/WongKinYiu/yolov7",
        directory,
        "a207844b1ce82d204ab36d87d496728d3d2348e7",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolov7_dir = directory / "yolov7"
    _run_uv_pip_install(
        yolov7_dir,
        bin_path.parent,
        "yolov7",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(yolov7_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    _run_patch(
        yolov7_dir,
        str((Path(__file__).parent / "patches" / "yolov7_export.patch").resolve()),
        "export.py",
        verbose=verbose,
    )
    subprocess.run(
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
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = yolov7_dir / (config["name"] + ".onnx")

    # patch names
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_ultralytics(
    directory: Path,
    config: dict[str, str],
    python_path: Path,  # noqa: ARG001
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
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
    _run_uv_pip_install(
        directory,
        bin_path.parent,
        "ultralytics",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    modelname = f"model={config['name']}"

    # Handle caching of ultralytics downloaded .pt weights
    weights_cache_dir = _get_weights_cache_dir()
    pt_filename = config["name"] + ".pt"
    cache_key = f"ultralytics_{pt_filename}"
    cached_pt_file = weights_cache_dir / cache_key
    target_pt_file = directory / pt_filename

    # Check if .pt file exists in cache
    if cached_pt_file.exists() and not no_cache:
        LOG.info(f"Using cached ultralytics weights: {pt_filename}")
        shutil.copy(cached_pt_file, target_pt_file)
        modelname = f"model={pt_filename}"

    subprocess.run(
        [
            str(bin_path / "yolo"),
            "export",
            modelname,
            "format=onnx",
            f"opset={opset}",
            f"imgsz={imgsz}",
        ],
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )

    # Cache the .pt file if it was downloaded and not already cached
    if target_pt_file.exists() and not cached_pt_file.exists():
        shutil.copy(target_pt_file, cached_pt_file)
        LOG.info(f"Cached ultralytics weights: {pt_filename}")

    model_path = directory / (config["name"] + ".onnx")
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_yolov9(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning("YOLOv9 is a GPL-3.0 licensed model, be aware of license restrictions")
    _git_clone(
        "https://github.com/WongKinYiu/yolov9",
        directory,
        "5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolov9_dir = directory / "yolov9"
    _run_patch(
        yolov9_dir,
        str((Path(__file__).parent / "patches" / "yolov9_export.patch").resolve()),
        "export.py",
        verbose=verbose,
    )
    _run_uv_pip_install(
        yolov9_dir,
        bin_path.parent,
        "yolov9",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(yolov9_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    subprocess.run(
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
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = yolov9_dir / (config["name"] + "-end2end.onnx")
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_yolov10(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,  # noqa: ARG001
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning(
            "YOLOv10 is a AGPL-3.0 licensed model, be aware of license restrictions"
        )
    _git_clone(
        "https://github.com/THU-MIG/yolov10",
        directory,
        "453c6e38a51e9d1d5a2aa5fb7f1014a711913397",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolov10_dir = directory / "yolov10"
    _run_uv_pip_install(
        yolov10_dir,
        bin_path.parent,
        "yolov10",
        packages=["."],
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(yolov10_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    subprocess.run(
        [
            str(bin_path / "yolo"),
            "export",
            f"model={config['name'] + '.pt'}",
            "format=onnx",
            f"opset={opset}",
            f"imgsz={imgsz}",
        ],
        cwd=yolov10_dir,
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    return yolov10_dir / (config["name"] + ".onnx")


def _export_yolov12(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,  # noqa: ARG001
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning(
            "YOLOv12 is a AGPL-3.0 licensed model, be aware of license restrictions"
        )
    _git_clone(
        "https://github.com/sunsmarterjie/yolov12",
        directory,
        "3bca22b336e96cfdabfec4c062b84eef210e9563",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolov12_dir = directory / "yolov12"
    _run_uv_pip_install(
        yolov12_dir,
        bin_path.parent,
        "yolov12",
        packages=["."],
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(yolov12_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    subprocess.run(
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
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    return yolov12_dir / (config["name"] + ".onnx")


def _export_yolov13(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,  # noqa: ARG001
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning(
            "YOLOv13 is a AGPL-3.0 licensed model, be aware of license restrictions"
        )
    _git_clone(
        "https://github.com/iMoonLab/yolov13",
        directory,
        "d09f2efa512bff266d558b562ae06b41e3af00d8",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolov13_dir = directory / "yolov13"
    _run_uv_pip_install(
        yolov13_dir,
        bin_path.parent,
        "yolov13",
        packages=["."],
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(yolov13_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    subprocess.run(
        [
            str(bin_path / "yolo"),
            "export",
            f"model={config['name'] + '.pt'}",
            "format=onnx",
            f"opset={opset}",
            f"imgsz={imgsz}",
        ],
        cwd=yolov13_dir,
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    return yolov13_dir / (config["name"] + ".onnx")


def _export_rtdetrv1(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning(
            "RT-DETRv1 is a Apache-2.0 licensed model, be aware of license restrictions"
        )
    if imgsz != 640:
        err_msg = f"RT-DETRv1 supports only an imgsz of 640, got {imgsz}"
        raise ValueError(err_msg)
    _git_clone(
        "https://github.com/lyuwenyu/RT-DETR",
        directory,
        "f9417e3acfa48bcb649e5ec0bc3de1e8677c8961",
        no_cache=no_cache,
        verbose=verbose,
    )
    rtdetr_dir = directory / "RT-DETR" / "rtdetr_pytorch"
    _run_uv_pip_install(
        rtdetr_dir,
        bin_path.parent,
        "rtdetrv1",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(rtdetr_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    _run_patch(
        rtdetr_dir,
        str(
            (Path(__file__).parent / "patches" / "rtdetrv1_export_onnx.patch").resolve()
        ),
        "tools/export_onnx.py",
        verbose=verbose,
    )
    subprocess.run(
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
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = rtdetr_dir / "model.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_rtdetrv2(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning(
            "RT-DETRv2 is a Apache-2.0 licensed model, be aware of license restrictions"
        )
    if imgsz != 640:
        err_msg = f"RT-DETRv2 supports only an imgsz of 640, got {imgsz}"
        raise ValueError(err_msg)
    _git_clone(
        "https://github.com/lyuwenyu/RT-DETR",
        directory,
        "f9417e3acfa48bcb649e5ec0bc3de1e8677c8961",
        no_cache=no_cache,
        verbose=verbose,
    )
    rtdetrv2_dir = directory / "RT-DETR" / "rtdetrv2_pytorch"
    _run_uv_pip_install(
        rtdetrv2_dir,
        bin_path.parent,
        "rtdetrv2",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(rtdetrv2_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    _run_patch(
        rtdetrv2_dir,
        str(
            (Path(__file__).parent / "patches" / "rtdetrv2_export_onnx.patch").resolve()
        ),
        "tools/export_onnx.py",
        verbose=verbose,
    )
    subprocess.run(
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
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = rtdetrv2_dir / "model.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_rtdetrv3(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning(
            "RT-DETRv3 is a Apache-2.0 licensed model, be aware of license restrictions"
        )
    if imgsz != 640:
        err_msg = f"RT-DETRv3 supports only an imgsz of 640, got {imgsz}"
        raise ValueError(err_msg)
    paddle2onnx_max_opset = 16
    if opset > paddle2onnx_max_opset:
        LOG.warning(
            f"RT-DETRv3 only supports opset <{paddle2onnx_max_opset}, using opset {paddle2onnx_max_opset}"
        )
        opset = paddle2onnx_max_opset
    _git_clone(
        "https://github.com/clxia12/RT-DETRv3",
        directory,
        "349e7d99a5065e7b684118912e6a74178d4f4625",
        no_cache=no_cache,
        verbose=verbose,
    )
    rtdetrv3_dir = directory / "RT-DETRv3"
    _run_uv_pip_install(
        rtdetrv3_dir,
        bin_path.parent,
        "rtdetrv3",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(rtdetrv3_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    subprocess.run(
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
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
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
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = rtdetrv3_dir / f"{config['name']}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_dfine(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning(
            "D-FINE is a Apache-2.0 licensed model, be aware of license restrictions"
        )
    if imgsz != 640:
        err_msg = f"D-FINE supports only an imgsz of 640, got {imgsz}"
        raise ValueError(err_msg)
    _git_clone(
        "https://github.com/Peterande/D-FINE",
        directory,
        "d6694750683b0c7e9f523ba6953d16f112a376ae",
        no_cache=no_cache,
        verbose=verbose,
    )
    dfine_dir = directory / "D-FINE"
    _run_uv_pip_install(
        dfine_dir,
        bin_path.parent,
        "dfine",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(dfine_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    _run_patch(
        dfine_dir,
        str((Path(__file__).parent / "patches" / "dfine_export_onnx.patch").resolve()),
        "tools/deployment/export_onnx.py",
        verbose=verbose,
    )
    subprocess.run(
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
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = dfine_dir / f"{config['name']}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_deim(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning("DEIM is a Apache-2.0 licensed model, be aware of license restrictions")
    if imgsz != 640:
        err_msg = f"DEIM supports only an imgsz of 640, got {imgsz}"
        raise ValueError(err_msg)
    _git_clone(
        "https://github.com/Intellindust-AI-Lab/DEIM",
        directory,
        "8f28fe63cca4bd2a0f4abaf9b0814b69d5abb658",
        no_cache=no_cache,
        verbose=verbose,
    )
    deim_dir = directory / "DEIM"
    _run_uv_pip_install(
        deim_dir,
        bin_path.parent,
        "deim",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(deim_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    _run_patch(
        deim_dir,
        str((Path(__file__).parent / "patches" / "deim_export_onnx.patch").resolve()),
        "tools/deployment/export_onnx.py",
        verbose=verbose,
    )
    config_folder = "deim_dfine"
    if "rtdetrv2" in config["name"]:
        config_folder = "deim_rtdetrv2"
    subprocess.run(
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
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = deim_dir / f"{config['name']}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_rfdetr(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning(
            "RF-DETR is a Apache-2.0 licensed model, be aware of license restrictions"
        )
    if imgsz % 32 != 0:
        new_imgsz = max(imgsz // 32, 1) * 32
        wrn_msg = f"RF-DETR does not support input size {imgsz}, "
        wrn_msg += f"using {new_imgsz} (closest divisible by 32)"
        LOG.warning(wrn_msg)
        imgsz = new_imgsz

    _run_uv_pip_install(
        directory,
        bin_path.parent,
        "rfdetr",
        no_cache=no_uv_cache,
        verbose=verbose,
    )

    # Handle caching of rfdetr downloaded .pth weights
    weights_cache_dir = _get_weights_cache_dir()
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
    subprocess.run(
        [
            python_path,
            "-c",
            program,
        ],
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )

    # Cache the .pth file if it was downloaded and not already cached
    if target_pth_file.exists() and not cached_pth_file.exists():
        shutil.copy(target_pth_file, cached_pth_file)
        LOG.info(f"Cached rfdetr weights: {config['weights']}")

    model_path = directory / "output" / "inference_model.sim.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_deimv2(
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
        LOG.warning(
            "DEIMv2 is a Apache-2.0 licensed model, be aware of license restrictions"
        )
    # atto is 320, femto is 416, all others 640
    if imgsz is not None:
        if "atto" in model:
            if imgsz != 320:
                err_msg = f"DEIMv2 atto model requires an imgsz of 320, got {imgsz}"
                raise ValueError(err_msg)
        elif "femto" in model:
            if imgsz != 416:
                err_msg = f"DEIMv2 femto model requires an imgsz of 416, got {imgsz}"
                raise ValueError(err_msg)
        else:
            if imgsz != 640:
                err_msg = f"DEIMv2 models (excluding atto/femto) require an imgsz of 640, got {imgsz}"
                raise ValueError(err_msg)
    _git_clone(
        "https://github.com/Intellindust-AI-Lab/DEIMv2",
        directory,
        "19d5b19a58c229dd7ad5f079947bbe398e005d01",
        no_cache=no_cache,
        verbose=verbose,
    )
    deim_dir = directory / "DEIMv2"
    _run_uv_pip_install(
        deim_dir,
        bin_path.parent,
        "deimv2",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(deim_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    config_folder = "deimv2"
    subprocess.run(
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
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = deim_dir / f"{config['name']}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_yolox(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    if not no_warn:
        LOG.warning(
            "YOLOX is a Apache-2.0 licensed model, be aware of license restrictions"
        )
    _git_clone(
        "https://github.com/Megvii-BaseDetection/YOLOX",
        directory,
        "6ddff4824372906469a7fae2dc3206c7aa4bbaee",
        no_cache=no_cache,
        verbose=verbose,
    )
    yolox_dir = directory / "YOLOX"
    _run_patch(
        yolox_dir,
        str((Path(__file__).parent / "patches" / "yolox_requirements.patch").resolve()),
        "requirements.txt",
        verbose=verbose,
    )
    _run_patch(
        yolox_dir,
        str((Path(__file__).parent / "patches" / "yolox_export_onnx.patch").resolve()),
        "tools/export_onnx.py",
        verbose=verbose,
    )
    _run_uv_pip_install(
        yolox_dir,
        bin_path.parent,
        model="yolox",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    _run_download(yolox_dir, config, python_path, no_cache=no_cache, verbose=verbose)
    # Use exp_file instead of name to avoid module import issues
    exp_file = yolox_dir / "exps" / "default" / f"{config['name']}.py"
    # Set PYTHONPATH so yolox imports work without installing the package
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{yolox_dir}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(yolox_dir)
    subprocess.run(
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
            "--decode_in_inference",
            "--no-onnxsim",  # Disable onnxslim due to compatibility issues with newer onnx
        ],
        cwd=yolox_dir,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = yolox_dir / f"{config['name']}.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def download_model(
    model: str,
    directory: Path,
    opset: int = 17,
    imgsz: int = 640,
    requirements_export: Path | None = None,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    accept: bool | None = None,
    verbose: bool | None = None,
) -> Path:
    """
    Download a model from remote source and convert to ONNX.

    Parameters
    ----------
    model : str
        The name of the model to download.
    directory : Path
        The directory to save the model and working files.
    opset : int, optional
        The ONNX opset version to use.
    imgsz : int, optional
        The image size to use for the model.
    requirements_export : Path, optional
        Export the created virtual environment's requirements to this path using uv pip freeze.
    no_cache : bool, optional
        Whether to disable caching of downloaded weights and repos.
    no_uv_cache : bool, optional
        Whether to disable caching of uv packages.
    accept : bool, optional
        Whether to accept the license terms for the model. If None or False, will raise an error.
        Must be True to proceed with the download.
    verbose : bool, optional
        Whether to print verbose output.

    Returns
    -------
    Path
        The path to the exported model inside the directory.

    Raises
    ------
    ValueError
        If the model is not supported or license is not accepted.

    """
    if not accept:
        err_msg = f"License acceptance required for model '{model}'. Please accept the license terms."
        raise ValueError(err_msg)

    model_configs: dict[str, dict[str, dict[str, str]]] = load_model_configs()
    config: dict[str, str] | None = None
    for model_set in model_configs.values():
        for model_name in model_set:
            if model_name == model:
                config = model_set[model_name]
                break
        if config is not None:
            break
    if config is None:
        err_msg = f"Model {model} is not supported"
        raise ValueError(err_msg)

    python_path, bin_path = _make_venv(directory, no_cache=no_uv_cache, verbose=verbose)
    requirements_export_path = (
        Path(requirements_export) if requirements_export is not None else None
    )
    packet = (
        directory,
        config,
        python_path,
        bin_path,
        model,
        opset,
        imgsz,
    )
    model_path: Path | None = None
    if config["url"] == "ultralytics":
        model_path = _export_ultralytics(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "deim" in model and "deimv2" not in model:
        model_path = _export_deim(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "deimv2" in model:
        model_path = _export_deimv2(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "yolox" in model:
        model_path = _export_yolox(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "yolov7" in model:
        model_path = _export_yolov7(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "yolov9" in model:
        model_path = _export_yolov9(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "yolov10" in model:
        model_path = _export_yolov10(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "yolov12" in model:
        model_path = _export_yolov12(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "yolov13" in model:
        model_path = _export_yolov13(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "rtdetrv1" in model:
        model_path = _export_rtdetrv1(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "rtdetrv2" in model:
        model_path = _export_rtdetrv2(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "rtdetrv3" in model:
        model_path = _export_rtdetrv3(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "dfine" in model:
        model_path = _export_dfine(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    elif "rfdetr" in model:
        model_path = _export_rfdetr(
            *packet, no_cache=no_cache, no_uv_cache=no_uv_cache, no_warn=no_warn, verbose=verbose
        )
    if model_path is None:
        err_msg = f"Model {model} is not supported"
        raise ValueError(err_msg)
    if requirements_export_path is not None:
        _export_requirements(bin_path.parent, requirements_export_path, verbose=verbose)
    return model_path.with_name(model + model_path.suffix)


def download(
    model: str,
    output: Path,
    opset: int = 17,
    imgsz: int = 640,
    requirements_export: Path | None = None,
    *,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
    accept: bool | None = None,
    verbose: bool | None = None,
) -> None:
    """
    Download a model from remote source and convert to ONNX.

    Parameters
    ----------
    model : str
        The name of the model to download.
    output : Path
        The path to save the model.
    opset : int, optional
        The ONNX opset version to use.
    imgsz : int, optional
        The image size to use for the model.
    requirements_export : Path, optional
        Export the created virtual environment's requirements to this path using uv pip freeze.
    no_cache : bool, optional
        Whether to disable caching of downloaded weights and repos.
    no_uv_cache : bool, optional
        Whether to disable caching of uv packages.
    accept : bool, optional
        Whether to accept the license terms for the model. If None, will prompt the user.
        If False, will raise an error. If True, will proceed without prompting.
    verbose : bool, optional
        Whether to print verbose output.

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = download_model(
            model,
            Path(temp_dir),
            opset,
            imgsz,
            requirements_export=requirements_export,
            no_cache=no_cache,
            no_uv_cache=no_uv_cache,
            no_warn=no_warn,
            accept=accept,
            verbose=verbose,
        )
        shutil.copy(model_path, output)

    if verbose is not None:
        LOG.info(f"Model {model} downloaded and converted to ONNX.")
