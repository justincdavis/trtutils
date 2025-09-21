# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S607, S603, S404
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from trtutils._log import LOG


def _make_venv(directory: Path) -> tuple[Path, Path]:
    subprocess.run(
        ["uv", "venv", ".venv", "--python=3.10", "--clear"],
        cwd=directory,
        check=True,
    )
    # Important: do NOT resolve symlinks here; keep .venv/bin/python so uv targets the venv
    bin_path: Path = directory / ".venv" / "bin"
    python_path: Path = bin_path / "python"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ],
        cwd=directory,
        check=True,
    )
    return python_path, bin_path


def _export_yolov7(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,  # noqa: ARG001
    model: str,
    opset: int,
    imgsz: int,
) -> Path:
    subprocess.run(
        ["git", "clone", "https://github.com/WongKinYiu/yolov7"],
        cwd=directory,
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            "-r",
            "requirements.txt",
            "torch==2.4.*",
            "onnx",
            "onnxruntime",
            "onnxslim",
            "onnxsim",
            "onnx_graphsurgeon",
        ],
        cwd=directory / "yolov7",
        check=True,
    )
    subprocess.run(
        ["wget", "-nc", config["url"]],
        cwd=directory / "yolov7",
        check=True,
    )
    subprocess.run(
        [
            "patch",
            "export.py",
            "-i",
            str((Path(__file__).parent / "patches" / "yolov7_export.patch").resolve()),
        ],
        cwd=directory / "yolov7",
        check=True,
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
        cwd=directory / "yolov7",
        check=True,
    )
    model_path = directory / "yolov7" / (config["name"] + ".onnx")

    # patch names
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_ultralytics(
    directory: Path,
    config: dict[str, str],
    python_path: Path,  # noqa: ARG001
    bin_path: Path,
    model: str,  # noqa: ARG001
    opset: int,
    imgsz: int,
) -> Path:
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            "ultralytics",
            "onnx",
            "onnxruntime",
            "onnxslim",
        ],
        cwd=directory,
        check=True,
    )
    subprocess.run(
        [
            str(bin_path / "yolo"),
            "export",
            f"model={config['name']}",
            "format=onnx",
            f"opset={opset}",
            f"imgsz={imgsz}",
        ],
        cwd=directory,
        check=True,
    )
    model_path = directory / (config["name"] + ".onnx")
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_yolov9(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,  # noqa: ARG001
    model: str,  # noqa: ARG001
    opset: int,
    imgsz: int,
) -> Path:
    subprocess.run(
        ["git", "clone", "https://github.com/WongKinYiu/yolov9"],
        cwd=directory,
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            "-r",
            "requirements.txt",
            "torch==2.4.*",
            "onnx",
            "onnxruntime",
            "onnx-simplifier>=0.4.1",
        ],
        cwd=directory / "yolov9",
        check=True,
    )
    subprocess.run(
        ["wget", "-nc", config["url"]],
        cwd=directory / "yolov9",
        check=True,
    )
    subprocess.run(
        [
            python_path,
            "export.py",
            "--weights",
            config["name"] + ".pt",
            "--include",
            "onnx",
            "--simplify",
            "--img-size",
            str(imgsz),
            str(imgsz),
            "--opset",
            str(opset),
        ],
        cwd=directory / "yolov9",
        check=True,
    )
    model_path = directory / "yolov9" / (config["name"] + ".onnx")
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _export_yolov10(
    directory: Path,
    config: dict[str, str],
    python_path: Path,  # noqa: ARG001
    bin_path: Path,
    model: str,  # noqa: ARG001
    opset: int,
    imgsz: int,
) -> Path:
    subprocess.run(
        ["git", "clone", "https://github.com/THU-MIG/yolov10"],
        cwd=directory,
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            ".",
            "torch==2.4.*",
            "onnx",
            "onnxsim",
            "huggingface_hub",
        ],
        cwd=directory / "yolov10",
        check=True,
    )
    subprocess.run(
        ["wget", "-nc", config["url"]],
        cwd=directory / "yolov10",
        check=True,
    )
    subprocess.run(
        [
            str(bin_path / "yolo"),
            "export",
            f"model={config['name'] + '.pt'}",
            "format=onnx",
            f"opset={opset}",
            f"imgsz={imgsz}",
        ],
        cwd=directory / "yolov10",
        check=True,
    )
    return directory / "yolov10" / (config["name"] + ".onnx")


def _export_yolov12(
    directory: Path,
    config: dict[str, str],
    python_path: Path,  # noqa: ARG001
    bin_path: Path,
    model: str,  # noqa: ARG001
    opset: int,
    imgsz: int,
) -> Path:
    subprocess.run(
        ["git", "clone", "https://github.com/sunsmarterjie/yolov12"],
        cwd=directory,
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            ".",
            "torch==2.4.*",
            "torchvision",
            "timm",
            "albumentations",
            "onnx",
            "onnxruntime",
            "pycocotools",
            "pyyaml",
            "scipy",
            "onnxslim",
            "onnxruntime-gpu",
            "gradio",
            "opencv-python",
            "psutil",
            "huggingface-hub",
            "safetensors",
            "numpy",
            "supervision",
        ],
        cwd=directory / "yolov12",
        check=True,
    )
    # subprocess.run(
    #     ["uv", "pip", "install", "-p", str(bin_path.parent), "flash_attn", "--no-build-isolation"],
    #     cwd=directory / "yolov12",
    #     check=True,
    # )
    subprocess.run(
        ["wget", "-nc", config["url"]],
        cwd=directory / "yolov12",
        check=True,
    )
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
        cwd=directory / "yolov12",
        check=True,
    )
    return directory / "yolov12" / (config["name"] + ".onnx")


def download_model(
    model: str,
    directory: Path,
    opset: int = 17,
    imgsz: int = 640,
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

    Returns
    -------
    Path
        The path to the exported model inside the directory.

    Raises
    ------
    ValueError
        If the model is not supported.

    """
    model_configs: dict[str, dict[str, dict[str, str]]] = {
        "yolov7": {
            "yolov7t": {
                "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
                "name": "yolov7-tiny",
            },
            "yolov7m": {
                "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
                "name": "yolov7",
            },
            "yolov7x": {
                "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt",
                "name": "yolov7x",
            },
            "yolov7w6": {
                "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt",
                "name": "yolov7-w6",
            },
            "yolov7e6": {
                "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt",
                "name": "yolov7-e6",
            },
            "yolov7d6": {
                "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt",
                "name": "yolov7-d6",
            },
            "yolov7e6e": {
                "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt",
                "name": "yolov7-e6e",
            },
        },
        "yolov8": {
            "yolov8n": {
                "url": "ultralytics",
                "name": "yolov8n",
            },
            "yolov8s": {
                "url": "ultralytics",
                "name": "yolov8s",
            },
            "yolov8m": {
                "url": "ultralytics",
                "name": "yolov8m",
            },
            "yolov8l": {
                "url": "ultralytics",
                "name": "yolov8l",
            },
            "yolov8x": {
                "url": "ultralytics",
                "name": "yolov8x",
            },
        },
        "yolov9": {
            "yolov9t": {
                "url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt",
                "name": "yolov9-t-converted",
            },
            "yolov9s": {
                "url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s-converted.pt",
                "name": "yolov9-s-converted",
            },
            "yolov9m": {
                "url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt",
                "name": "yolov9-m-converted",
            },
            "yolov9c": {
                "url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt",
                "name": "yolov9-c-converted",
            },
            "yolov9e": {
                "url": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt",
                "name": "yolov9-e-converted",
            },
        },
        "yolov10": {
            "yolov10n": {
                "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
                "name": "yolov10n",
            },
            "yolov10s": {
                "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
                "name": "yolov10s",
            },
            "yolov10m": {
                "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt",
                "name": "yolov10m",
            },
            "yolov10b": {
                "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt",
                "name": "yolov10b",
            },
            "yolov10l": {
                "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt",
                "name": "yolov10l",
            },
            "yolov10x": {
                "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt",
                "name": "yolov10x",
            },
        },
        "yolov11": {
            "yolov11n": {
                "url": "ultralytics",
                "name": "yolo11n",
            },
            "yolov11s": {
                "url": "ultralytics",
                "name": "yolo11s",
            },
            "yolov11m": {
                "url": "ultralytics",
                "name": "yolo11m",
            },
            "yolov11l": {
                "url": "ultralytics",
                "name": "yolo11l",
            },
            "yolov11x": {
                "url": "ultralytics",
                "name": "yolo11x",
            },
        },
        "yolov12": {
            "yolov12n": {
                "url": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt",
                "name": "yolov12n",
            },
            "yolov12s": {
                "url": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt",
                "name": "yolov12s",
            },
            "yolov12m": {
                "url": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12m.pt",
                "name": "yolov12m",
            },
            "yolov12l": {
                "url": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12l.pt",
                "name": "yolov12l",
            },
            "yolov12x": {
                "url": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12x.pt",
                "name": "yolov12x",
            },
        },
    }
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

    python_path, bin_path = _make_venv(directory)
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
    if "yolov7" in model:
        model_path = _export_yolov7(*packet)
    if "yolov8" in model or "yolov11" in model:
        model_path = _export_ultralytics(*packet)
    if "yolov9" in model:
        model_path = _export_yolov9(*packet)
    if "yolov10" in model:
        model_path = _export_yolov10(*packet)
    if "yolov12" in model:
        model_path = _export_yolov12(*packet)
    if model_path is None:
        err_msg = f"Model {model} is not supported"
        raise ValueError(err_msg)
    return model_path.with_name(model + model_path.suffix)


def download(
    model: str,
    output: Path,
    opset: int = 17,
    imgsz: int = 640,
    *,
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
    verbose : bool, optional
        Whether to print verbose output.

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = download_model(model, Path(temp_dir), opset, imgsz)
        shutil.copy(model_path, output)

    if verbose is not None:
        LOG.info(f"Model {model} downloaded and converted to ONNX.")
