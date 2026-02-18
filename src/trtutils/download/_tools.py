# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S607, S603
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import IO, TYPE_CHECKING
from urllib.parse import urlsplit

from trtutils._log import LOG

if TYPE_CHECKING:
    from collections.abc import Sequence

_MIN_UV_VERSION = (0, 9, 0)
_MIN_UV_VERSION_PARTS = 2
_EXPECTED_UV_PARTS = 3
_640 = 640


def get_patches_dir() -> Path:
    return Path(__file__).parent / "patches"


def handle_imgsz(
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


def kill_process_group(pid: int | None, cmd: Sequence[str]) -> None:
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError as e:
        LOG.warning(f"Failed to kill process group for {cmd}: {e}")


def run_cmd(
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
        kill_process_group(proc.pid, cmd_list)
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
        kill_process_group(proc.pid, cmd_list)
        raise subprocess.CalledProcessError(proc.returncode, cmd_list)
    return proc.returncode


def check_uv_available() -> None:
    if shutil.which("uv") is None:
        err_msg = (
            "uv is not available. Please install uv (version >= 0.9.0) to use the download function. "
            "See https://docs.astral.sh/uv/getting-started/installation/ for installation instructions."
        )
        raise RuntimeError(err_msg)


def check_uv_version() -> None:
    check_uv_available()

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
def get_cache_dir() -> Path:
    cache_dir = Path.home() / ".cache" / "trtutils"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def get_repo_cache_dir() -> Path:
    repo_cache = get_cache_dir() / "repos"
    repo_cache.mkdir(parents=True, exist_ok=True)
    return repo_cache


@lru_cache(maxsize=1)
def get_weights_cache_dir() -> Path:
    weights_cache = get_cache_dir() / "weights"
    weights_cache.mkdir(parents=True, exist_ok=True)
    return weights_cache


@lru_cache(maxsize=0)
def get_model_requirements(model: str) -> str:
    file_path = Path(__file__).parent / "requirements" / f"{model}.txt"
    return str(file_path.resolve())


@lru_cache(maxsize=0)
def get_tool_requirements(tool: str) -> str:
    file_path = Path(__file__).parent / "requirements_tools" / f"{tool}.txt"
    return str(file_path.resolve())


def git_clone(
    url: str,
    directory: Path,
    commit: str | None = None,
    *,
    no_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    # Extract repo name from URL
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    cache_repo_path = get_repo_cache_dir() / repo_name
    target_path = directory / repo_name

    # Check if repo exists in cache
    if cache_repo_path.exists() and not no_cache:
        # Copy from cache
        LOG.info(f"Using cached repository: {repo_name}")
        shutil.copytree(cache_repo_path, target_path)
    else:
        LOG.info(f"Cloning repository: {repo_name}")
        # Clone fresh
        run_cmd(
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
        run_cmd(
            ["git", "checkout", commit],
            cwd=target_path,
            verbose=verbose,
        )


def run_uv_pip_install(
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
                get_model_requirements(model),
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
    run_cmd(
        cmd,
        cwd=directory,
        verbose=verbose,
    )


def export_requirements(
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
        run_cmd(
            ["uv", "pip", "freeze", "-p", str(venv_path)],
            stdout=requirements_file,
            stderr=subprocess.DEVNULL if not verbose else None,
        )


def make_venv(
    directory: Path,
    *,
    no_cache: bool | None = None,
    verbose: bool | None = None,
) -> tuple[Path, Path]:
    run_cmd(
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
    run_uv_pip_install(
        directory,
        bin_path.parent,
        model=None,
        packages=std_packages,
        verbose=verbose,
    )
    return python_path, bin_path


def run_download(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    *,
    no_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    weights_cache_dir = get_weights_cache_dir()

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
        parsed_url = urlsplit(config["url"])
        filename = Path(parsed_url.path).name
        if not filename:
            err_msg = f"Unable to determine filename from URL: {config['url']}"
            raise ValueError(err_msg)
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
            run_cmd(
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
            run_cmd(
                ["wget", "-O", filename, config["url"]],
                cwd=directory,
                verbose=verbose,
            )

        # Cache the downloaded file, removing existing version if it exists
        if cached_file.exists():
            cached_file.unlink()
        shutil.copy(target_file, cached_file)
        LOG.info(f"Cached weights: {filename}")


def run_patch(
    directory: Path,
    patch_file: str,
    file_to_patch: str,
    *,
    verbose: bool | None = None,
) -> None:
    LOG.info(f"Patching {file_to_patch} with {patch_file}")
    run_cmd(
        ["patch", file_to_patch, "-i", patch_file],
        cwd=directory,
        verbose=verbose,
    )
