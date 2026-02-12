# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S607, S603
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import IO, TYPE_CHECKING

from trtutils._log import LOG

if TYPE_CHECKING:
    from collections.abc import Sequence

_MIN_UV_VERSION = (0, 9, 0)
_MIN_UV_VERSION_PARTS = 2
_EXPECTED_UV_PARTS = 3
_640 = 640


def _handle_imgsz(
    imgsz: int | None,
    default: int,
    model_name: str,
    *,
    enforce: bool = False,
    adjust_div: int | None = None,
) -> int:
    """Handle image size validation and adjustment."""
    if imgsz is None:
        imgsz = default
    elif enforce and imgsz != default:
        LOG.warning(
            f"{model_name} supports only an imgsz of {default}, got {imgsz}. Using {default}."
        )
        imgsz = default

    if adjust_div is not None and imgsz % adjust_div != 0:
        new_imgsz = max(imgsz // adjust_div, 1) * adjust_div
        LOG.warning(
            f"{model_name} requires imgsz divisible by {adjust_div}, got {imgsz}. Using {new_imgsz}."
        )
        imgsz = new_imgsz

    return imgsz


def _kill_process_group(pid: int | None, cmd: Sequence[str]) -> None:
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError as e:
        LOG.warning(f"Failed to kill process group for {cmd}: {e}")


def _run_cmd(
    cmd: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    verbose: bool | None = None,
    timeout: float | None = None,
    stdout: IO[str] | IO[bytes] | int | None = None,
    stderr: IO[str] | IO[bytes] | int | None = None,
    check: bool = True,
) -> int:
    final_stdout = stdout if stdout is not None else (None if verbose else subprocess.DEVNULL)
    final_stderr = stderr if stderr is not None else (None if verbose else subprocess.STDOUT)
    cmd_list = [str(part) for part in cmd]
    proc = subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=final_stdout,
        stderr=final_stderr,
        start_new_session=True,
    )
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        LOG.error(f"Command timed out, killing process group: {' '.join(cmd_list)}")
        _kill_process_group(proc.pid, cmd_list)
        raise
    if check and proc.returncode:
        sigkill_code = -9
        if proc.returncode == sigkill_code:
            LOG.error(
                "Command was killed with SIGKILL (exit code -9). This usually means the system "
                "ran out of memory (OOM). Try closing other applications, increasing available RAM, "
                "or on WSL2, increasing memory in .wslconfig."
            )
        LOG.error(f"Command failed with code {proc.returncode}: {' '.join(cmd_list)}")
        _kill_process_group(proc.pid, cmd_list)
        raise subprocess.CalledProcessError(proc.returncode, cmd_list)
    return proc.returncode


def _check_uv_available() -> None:
    if shutil.which("uv") is None:
        err_msg = (
            "uv is not available. Please install uv (version >= 0.9.0) to use the download function. "
            "See https://docs.astral.sh/uv/getting-started/installation/ for installation instructions."
        )
        raise RuntimeError(err_msg)


def _check_uv_version() -> None:
    _check_uv_available()

    def _failed_to_parse_uv_version() -> None:
        err_msg = f"Failed to parse uv version. Please ensure uv (version >= {_MIN_UV_VERSION}) is installed correctly."
        raise RuntimeError(err_msg)

    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )

        version_output = result.stdout.strip()
        parts = version_output.split()
        if len(parts) < _MIN_UV_VERSION_PARTS:
            _failed_to_parse_uv_version()

        uv_version_str = parts[1]
        version_parts = [int(x) for x in uv_version_str.split(".")]
        if len(version_parts) != _EXPECTED_UV_PARTS:
            _failed_to_parse_uv_version()

        if tuple(version_parts) < _MIN_UV_VERSION:
            err_msg = (
                f"uv version {uv_version_str} is too old. "
                f"Minimum required version is {_MIN_UV_VERSION}. Please upgrade uv."
            )
            raise RuntimeError(err_msg)
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        err_msg = (
            "Failed to determine uv version. Please ensure uv is installed correctly "
            f"and version >= {_MIN_UV_VERSION}."
        )
        raise RuntimeError(err_msg) from e


@lru_cache(maxsize=1)
def _get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "trtutils"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def _get_repo_cache_dir() -> Path:
    repo_cache = _get_cache_dir() / "repos"
    repo_cache.mkdir(parents=True, exist_ok=True)
    return repo_cache


@lru_cache(maxsize=1)
def _get_weights_cache_dir() -> Path:
    weights_cache = _get_cache_dir() / "weights"
    weights_cache.mkdir(parents=True, exist_ok=True)
    return weights_cache


@lru_cache(maxsize=0)
def _get_model_requirements(model: str) -> str:
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
        except (FileNotFoundError, KeyError) as e:
            LOG.warning(f"Failed to load configuration file {config_path.name}: {e}")

    return model_configs


@lru_cache(maxsize=1)
def get_supported_models() -> list[str]:
    """
    Return a list of supported model names.

    Returns
    -------
    list[str]
        A list of supported model names.

    """
    model_configs = load_model_configs()
    names: list[str] = []
    for model_set in model_configs.values():
        names.extend(model_set.keys())
    return names


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
        _run_cmd(
            ["git", "clone", url],
            cwd=directory,
            verbose=verbose,
        )
        # Cache the cloned repo, removing existing version if it exists
        if cache_repo_path.exists():
            shutil.rmtree(cache_repo_path)
        shutil.copytree(target_path, cache_repo_path)
        LOG.info(f"Cached repository: {repo_name}")

    if commit:
        _run_cmd(
            ["git", "checkout", commit],
            cwd=target_path,
            verbose=verbose,
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
    _run_cmd(
        cmd,
        cwd=directory,
        verbose=verbose,
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
        _run_cmd(
            ["uv", "pip", "freeze", "-p", str(venv_path)],
            stdout=requirements_file,
            stderr=subprocess.DEVNULL if not verbose else None,
        )


def _make_venv(
    directory: Path,
    *,
    no_cache: bool | None = None,
    verbose: bool | None = None,
) -> tuple[Path, Path]:
    _run_cmd(
        ["uv", "venv", ".venv", "--python=3.10", "--clear"],
        cwd=directory,
        verbose=verbose,
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
            _run_cmd(
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
                verbose=verbose,
            )
        else:
            _run_cmd(
                ["wget", "-nc", config["url"]],
                cwd=directory,
                verbose=verbose,
            )

        # Cache the downloaded file, removing existing version if it exists
        if cached_file.exists():
            cached_file.unlink()
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
    _run_cmd(
        ["patch", file_to_patch, "-i", patch_file],
        cwd=directory,
        verbose=verbose,
    )


def _export_yolov7(
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
    imgsz = _handle_imgsz(imgsz, _640, "YOLOv7")
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
    _run_cmd(
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


def _export_ultralytics(
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
    imgsz = _handle_imgsz(imgsz, _640, "Ultralytics")
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

    _run_cmd(
        [
            str(bin_path / "yolo"),
            "export",
            modelname,
            "format=onnx",
            f"opset={opset}",
            f"imgsz={imgsz}",
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


def _export_yolov9(
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
    imgsz = _handle_imgsz(imgsz, _640, "YOLOv9")
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
    _run_cmd(
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


def _export_yolov10(
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
    imgsz = _handle_imgsz(imgsz, _640, "YOLOv10")
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
    _run_cmd(
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


def _export_yolov12(
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
    imgsz = _handle_imgsz(imgsz, _640, "YOLOv12")
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
    _run_cmd(
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


def _export_yolov13(
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
    imgsz = _handle_imgsz(imgsz, _640, "YOLOv13")
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
    _run_cmd(
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


def _export_rtdetrv1(
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
    imgsz = _handle_imgsz(imgsz, _640, "RT-DETRv1", enforce=True)
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
        str((Path(__file__).parent / "patches" / "rtdetrv1_export_onnx.patch").resolve()),
        "tools/export_onnx.py",
        verbose=verbose,
    )
    _run_cmd(
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


def _export_rtdetrv2(
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
    imgsz = _handle_imgsz(imgsz, _640, "RT-DETRv2", enforce=True)
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
        str((Path(__file__).parent / "patches" / "rtdetrv2_export_onnx.patch").resolve()),
        "tools/export_onnx.py",
        verbose=verbose,
    )
    _run_cmd(
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


def _export_rtdetrv3(
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
    imgsz = _handle_imgsz(imgsz, _640, "RT-DETRv3", enforce=True)
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
    _run_cmd(
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
    _run_cmd(
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


def _export_dfine(
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
    imgsz = _handle_imgsz(imgsz, _640, "D-FINE", enforce=True)
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
    _run_cmd(
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


def _export_deim(
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
    imgsz = _handle_imgsz(imgsz, _640, "DEIM", enforce=True)
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
    _run_cmd(
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


def _export_rfdetr(
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
    imgsz = _handle_imgsz(imgsz, required_imgsz, model, enforce=True, adjust_div=32)

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
    _run_cmd(
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
    imgsz = _handle_imgsz(imgsz, required_imgsz, model, enforce=True)
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
    _run_cmd(
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


def _export_yolox(
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
    imgsz = _handle_imgsz(imgsz, _640, "YOLOX")
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
    _run_cmd(
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


def _export_torchvision_classifier(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None = None,
    *,
    no_cache: bool | None = None,  # noqa: ARG001
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    _224 = 224
    if imgsz is None:
        imgsz = _224
    _run_uv_pip_install(
        directory,
        bin_path.parent,
        "torchvision_classifier",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    script_content = f"""\
import torch
import torchvision.models as models
import onnx
import onnxslim

model_name = "{config["name"]}"
opset = {opset}
imgsz = {imgsz}
output_path = model_name + ".onnx"

model = getattr(models, model_name)(weights="DEFAULT")
model.eval()
dummy_input = torch.randn(1, 3, imgsz, imgsz)
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    opset_version=opset,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={{"input": {{0: "batch_size"}}, "output": {{0: "batch_size"}}}},
)
slim_model = onnxslim.slim(output_path)
onnx.save(slim_model, output_path)
"""
    script_path = directory / "_export_torchvision_classifier.py"
    script_path.write_text(script_content)
    _run_cmd(
        [python_path, str(script_path)],
        cwd=directory,
        verbose=verbose,
    )
    model_path = directory / (config["name"] + ".onnx")
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def download_model(
    model: str,
    directory: Path,
    opset: int = 17,
    imgsz: int | None = None,
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
        By default, the model will use the default image size for the model.
    requirements_export : Path, optional
        Export the created virtual environment's requirements to this path using uv pip freeze.
    no_cache : bool, optional
        Whether to disable caching of downloaded weights and repos.
    no_uv_cache : bool, optional
        Whether to disable caching of uv packages.
    no_warn : bool, optional
        Whether to disable warnings for the model.
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
        err_msg = (
            f"License acceptance required for model '{model}'. Please accept the license terms."
        )
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
    requirements_export_path = Path(requirements_export) if requirements_export is not None else None

    # Determine which export function to use
    export_func = None
    if config.get("url") == "torchvision_classifier":
        export_func = _export_torchvision_classifier
    elif config.get("url") == "ultralytics":
        export_func = _export_ultralytics
    elif "deim" in model and "deimv2" not in model:
        export_func = _export_deim
    elif "deimv2" in model:
        export_func = _export_deimv2
    elif "yolox" in model:
        export_func = _export_yolox
    elif "yolov7" in model:
        export_func = _export_yolov7
    elif "yolov9" in model:
        export_func = _export_yolov9
    elif "yolov10" in model:
        export_func = _export_yolov10
    elif "yolov12" in model:
        export_func = _export_yolov12
    elif "yolov13" in model:
        export_func = _export_yolov13
    elif "rtdetrv1" in model:
        export_func = _export_rtdetrv1
    elif "rtdetrv2" in model:
        export_func = _export_rtdetrv2
    elif "rtdetrv3" in model:
        export_func = _export_rtdetrv3
    elif "dfine" in model:
        export_func = _export_dfine
    elif "rfdetr" in model:
        export_func = _export_rfdetr

    # Single call site
    if export_func is None:
        err_msg = f"Model {model} is not supported"
        raise ValueError(err_msg)

    model_path = export_func(
        directory,
        config,
        python_path,
        bin_path,
        model,
        opset,
        imgsz,
        no_cache=no_cache,
        no_uv_cache=no_uv_cache,
        no_warn=no_warn,
        verbose=verbose,
    )
    if requirements_export_path is not None:
        _export_requirements(bin_path.parent, requirements_export_path, verbose=verbose)
    return model_path.with_name(model + model_path.suffix)


def download(
    model: str,
    output: Path,
    opset: int = 17,
    imgsz: int | None = None,
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
        By default, the model will use the default image size for the model.
    requirements_export : Path, optional
        Export the created virtual environment's requirements to this path using uv pip freeze.
    no_cache : bool, optional
        Whether to disable caching of downloaded weights and repos.
    no_uv_cache : bool, optional
        Whether to disable caching of uv packages.
    no_warn : bool, optional
        Whether to disable warnings for the model.
    accept : bool, optional
        Whether to accept the license terms for the model. If None, will prompt the user.
        If False, will raise an error. If True, will proceed without prompting.
    verbose : bool, optional
        Whether to print verbose output.

    """
    _check_uv_version()

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
