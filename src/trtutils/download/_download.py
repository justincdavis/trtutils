# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: S607, S603, S404
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from trtutils._log import LOG


def _load_model_configs() -> dict[str, dict[str, dict[str, str]]]:
    configs_dir = Path(__file__).parent / "configs"
    model_configs: dict[str, dict[str, dict[str, str]]] = {}

    config_files = {
        "yolov7": "yolov7.json",
        "yolov8": "yolov8.json",
        "yolov9": "yolov9.json",
        "yolov10": "yolov10.json",
        "yolov11": "yolov11.json",
        "yolov12": "yolov12.json",
        "yolov13": "yolov13.json",
        "rtdetrv1": "rtdetrv1.json",
        "rtdetrv2": "rtdetrv2.json",
        "rtdetrv3": "rtdetrv3.json",
        "dfine": "dfine.json",
        "deim": "deim.json",
        "rfdetr": "rfdetr.json",
    }

    for model_type, config_file in config_files.items():
        config_path = configs_dir / config_file
        if config_path.exists():
            with config_path.open() as f:
                model_configs[model_type] = json.load(f)
        else:
            LOG.warning(f"Configuration file {config_file} not found in {configs_dir}")

    return model_configs


def _git_clone(
    url: str,
    directory: Path,
    *,
    verbose: bool | None = None,
) -> None:
    subprocess.run(
        ["git", "clone", url],
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )


def _make_venv(
    directory: Path,
    *,
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    return python_path, bin_path


def _export_yolov7(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    verbose: bool | None = None,
) -> Path:
    LOG.warning("YOLOv7 is a GPL-3.0 licensed model, be aware of license restrictions")
    _git_clone("https://github.com/WongKinYiu/yolov7", directory, verbose=verbose)
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        ["wget", "-nc", config["url"]],
        cwd=directory / "yolov7",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
    model: str,
    opset: int,
    imgsz: int,
    *,
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "Ultralytics is a AGPL-3.0 and commercial licensed model, be aware of license restrictions"
    )
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
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
    verbose: bool | None = None,
) -> Path:
    LOG.warning("YOLOv9 is a GPL-3.0 licensed model, be aware of license restrictions")
    _git_clone("https://github.com/WongKinYiu/yolov9", directory, verbose=verbose)
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        ["wget", "-nc", config["url"]],
        cwd=directory / "yolov9",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
    *,
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "YOLOv10 is a AGPL-3.0 licensed model, be aware of license restrictions"
    )
    _git_clone("https://github.com/THU-MIG/yolov10", directory, verbose=verbose)
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        ["wget", "-nc", config["url"]],
        cwd=directory / "yolov10",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
    *,
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "YOLOv12 is a AGPL-3.0 licensed model, be aware of license restrictions"
    )
    _git_clone("https://github.com/sunsmarterjie/yolov12", directory, verbose=verbose)
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    return directory / "yolov12" / (config["name"] + ".onnx")


def _export_yolov13(
    directory: Path,
    config: dict[str, str],
    python_path: Path,  # noqa: ARG001
    bin_path: Path,
    model: str,  # noqa: ARG001
    opset: int,
    imgsz: int,
    *,
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "YOLOv13 is a AGPL-3.0 licensed model, be aware of license restrictions"
    )
    _git_clone("https://github.com/iMoonLab/yolov13", directory, verbose=verbose)
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
            "onnxslim",
            "onnxruntime-gpu",
            "huggingface_hub",
        ],
        cwd=directory / "yolov13",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        ["wget", "-nc", config["url"]],
        cwd=directory / "yolov13",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        cwd=directory / "yolov13",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    return directory / "yolov13" / (config["name"] + ".onnx")


def _export_rtdetrv1(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int,
    *,
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "RT-DETRv1 is a Apache-2.0 licensed model, be aware of license restrictions"
    )
    _git_clone("https://github.com/lyuwenyu/RT-DETR", directory, verbose=verbose)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            "-r",
            "requirements.txt",
            "onnxsim>=0.4",
            "numpy==1.*",
        ],
        cwd=directory / "RT-DETR" / "rtdetr_pytorch",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        ["wget", "-nc", config["url"]],
        cwd=directory / "RT-DETR" / "rtdetr_pytorch",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        [
            "patch",
            "tools/export_onnx.py",
            "-i",
            str(
                (
                    Path(__file__).parent / "patches" / "rtdetrv1_export_onnx.patch"
                ).resolve()
            ),
        ],
        cwd=directory / "RT-DETR" / "rtdetr_pytorch",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        cwd=directory / "RT-DETR" / "rtdetr_pytorch",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = directory / "RT-DETR" / "rtdetr_pytorch" / "model.onnx"
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
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "RT-DETRv2 is a Apache-2.0 licensed model, be aware of license restrictions"
    )
    _git_clone("https://github.com/lyuwenyu/RT-DETR", directory, verbose=verbose)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            "-r",
            "requirements.txt",
            "onnxsim>=0.4",
            "numpy==1.*",
        ],
        cwd=directory / "RT-DETR" / "rtdetrv2_pytorch",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        ["wget", "-nc", config["url"]],
        cwd=directory / "RT-DETR" / "rtdetrv2_pytorch",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        [
            "patch",
            "tools/export_onnx.py",
            "-i",
            str(
                (
                    Path(__file__).parent / "patches" / "rtdetrv2_export_onnx.patch"
                ).resolve()
            ),
        ],
        cwd=directory / "RT-DETR" / "rtdetrv2_pytorch",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        ],
        cwd=directory / "RT-DETR" / "rtdetrv2_pytorch",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = directory / "RT-DETR" / "rtdetrv2_pytorch" / "model.onnx"
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
    imgsz: int,  # noqa: ARG001
    *,
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "RT-DETRv3 is a Apache-2.0 licensed model, be aware of license restrictions"
    )
    LOG.warning("RT-DETRv3 does not support setting alternative input sizes")
    paddle2onnx_max_opset = 16
    if opset > paddle2onnx_max_opset:
        LOG.warning(
            f"RT-DETRv3 only supports opset <{paddle2onnx_max_opset}, using opset {paddle2onnx_max_opset}"
        )
        opset = paddle2onnx_max_opset
    _git_clone("https://github.com/clxia12/RT-DETRv3", directory, verbose=verbose)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            "-r",
            "requirements.txt",
            "paddlepaddle==2.6.1",
            "paddle2onnx==1.0.5",
            "onnx==1.13.0",
            "onnxsim>=0.4",
            "scikit-learn",
            "gdown",
        ],
        cwd=directory / "RT-DETRv3",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        [
            python_path,
            "-m",
            "gdown",
            "--id",
            config["id"],
            "-O",
            config["name"] + ".pdparams",
        ],
        cwd=directory / "RT-DETRv3",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
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
        cwd=directory / "RT-DETRv3",
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
        cwd=directory / "RT-DETRv3",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = directory / "RT-DETRv3" / f"{config['name']}.onnx"
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
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "D-FINE is a Apache-2.0 licensed model, be aware of license restrictions"
    )
    _git_clone("https://github.com/Peterande/D-FINE", directory, verbose=verbose)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            "-r",
            "requirements.txt",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch==2.4.1",
            "torchvision==0.19.1",
            "onnx",
            "onnxsim",
            "onnxruntime",
        ],
        cwd=directory / "D-FINE",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        ["wget", "-nc", config["url"]],
        cwd=directory / "D-FINE",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        [
            "patch",
            "tools/deployment/export_onnx.py",
            "-i",
            str(
                (
                    Path(__file__).parent / "patches" / "dfine_export_onnx.patch"
                ).resolve()
            ),
        ],
        cwd=directory / "D-FINE",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        cwd=directory / "D-FINE",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = directory / "D-FINE" / f"{config['name']}.onnx"
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
    verbose: bool | None = None,
) -> Path:
    LOG.warning("DEIM is a Apache-2.0 licensed model, be aware of license restrictions")
    _git_clone("https://github.com/Intellindust-AI-Lab/DEIM", directory, verbose=verbose)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            "-r",
            "requirements.txt",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch==2.4.1",
            "torchvision==0.19.1",
            "onnx",
            "onnxsim",
            "onnxruntime",
            "gdown",
        ],
        cwd=directory / "DEIM",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        [
            python_path,
            "-m",
            "gdown",
            "--id",
            config["id"],
            "-O",
            config["name"] + ".pth",
        ],
        cwd=directory / "DEIM",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    subprocess.run(
        [
            "patch",
            "tools/deployment/export_onnx.py",
            "-i",
            str(
                (Path(__file__).parent / "patches" / "deim_export_onnx.patch").resolve()
            ),
        ],
        cwd=directory / "DEIM",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
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
        ],
        cwd=directory / "DEIM",
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = directory / "DEIM" / f"{config['name']}.onnx"
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
    imgsz: int,  # noqa: ARG001
    *,
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "RF-DETR is a Apache-2.0 licensed model, be aware of license restrictions"
    )
    LOG.warning("RF-DETR does not support setting alternative input sizes")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(bin_path.parent),
            "rfdetr[onnxexport]",
        ],
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    program = f"""
import rfdetr
model = rfdetr.{config["class"]}()
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
    model_path = directory / "output" / "inference_model.sim.onnx"
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def download_model(
    model: str,
    directory: Path,
    opset: int = 17,
    imgsz: int = 640,
    *,
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

    Returns
    -------
    Path
        The path to the exported model inside the directory.

    Raises
    ------
    ValueError
        If the model is not supported.

    """
    model_configs = _load_model_configs()
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

    python_path, bin_path = _make_venv(directory, verbose=verbose)
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
    if "deim" in model and "deimv2" not in model:
        model_path = _export_deim(*packet, verbose=verbose)
    elif "yolov7" in model:
        model_path = _export_yolov7(*packet, verbose=verbose)
    elif "yolov8" in model or "yolov11" in model:
        model_path = _export_ultralytics(*packet, verbose=verbose)
    elif "yolov9" in model:
        model_path = _export_yolov9(*packet, verbose=verbose)
    elif "yolov10" in model:
        model_path = _export_yolov10(*packet, verbose=verbose)
    elif "yolov12" in model:
        model_path = _export_yolov12(*packet, verbose=verbose)
    elif "yolov13" in model:
        model_path = _export_yolov13(*packet, verbose=verbose)
    elif "rtdetrv1" in model:
        model_path = _export_rtdetrv1(*packet, verbose=verbose)
    elif "rtdetrv2" in model:
        model_path = _export_rtdetrv2(*packet, verbose=verbose)
    elif "rtdetrv3" in model:
        model_path = _export_rtdetrv3(*packet, verbose=verbose)
    elif "dfine" in model:
        model_path = _export_dfine(*packet, verbose=verbose)
    elif "rfdetr" in model:
        model_path = _export_rfdetr(*packet, verbose=verbose)
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
        model_path = download_model(model, Path(temp_dir), opset, imgsz, verbose=verbose)
        shutil.copy(model_path, output)

    if verbose is not None:
        LOG.info(f"Model {model} downloaded and converted to ONNX.")
