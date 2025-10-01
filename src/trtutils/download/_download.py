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
from functools import lru_cache

from trtutils._log import LOG


@lru_cache(maxsize=1)
def _load_model_configs() -> dict[str, dict[str, dict[str, str]]]:
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


def _run_uv_pip_install(
    directory: Path,
    venv_path: Path,
    *packages: str,
    verbose: bool | None = None,
) -> None:
    cmd = ["uv", "pip", "install", "-p", str(venv_path)]
    cmd.extend(packages)
    subprocess.run(
        cmd,
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
    _run_uv_pip_install(
        directory,
        bin_path.parent,
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
        verbose=verbose,
    )
    return python_path, bin_path


def _run_download(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    *,
    verbose: bool | None = None,
) -> None:
    if "id" in config:
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


def _run_patch(
    directory: Path,
    patch_file: str,
    file_to_patch: str,
    *,
    verbose: bool | None = None,
) -> None:
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
    verbose: bool | None = None,
) -> Path:
    LOG.warning("YOLOv7 is a GPL-3.0 licensed model, be aware of license restrictions")
    _git_clone("https://github.com/WongKinYiu/yolov7", directory, verbose=verbose)
    yolov7_dir = directory / "yolov7"
    _run_uv_pip_install(
        yolov7_dir,
        bin_path.parent,
        "-r",
        "requirements.txt",
        "torch==2.4.*",
        "onnx",
        "onnxruntime",
        "onnxslim",
        "onnxsim",
        "onnx_graphsurgeon",
        verbose=verbose,
    )
    _run_download(yolov7_dir, config, python_path, verbose=verbose)
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
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "Ultralytics is a AGPL-3.0 and commercial licensed model, be aware of license restrictions"
    )
    _run_uv_pip_install(
        directory,
        bin_path.parent,
        "ultralytics",
        "onnx",
        "onnxruntime",
        "onnxslim",
        verbose=verbose,
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
    yolov9_dir = directory / "yolov9"
    _run_uv_pip_install(
        yolov9_dir,
        bin_path.parent,
        "-r",
        "requirements.txt",
        "torch==2.4.*",
        "onnx",
        "onnxruntime",
        "onnx-simplifier>=0.4.1",
        verbose=verbose,
    )
    _run_download(yolov9_dir, config, python_path, verbose=verbose)
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
        cwd=yolov9_dir,
        check=True,
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
    )
    model_path = yolov9_dir / (config["name"] + ".onnx")
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
    yolov10_dir = directory / "yolov10"
    _run_uv_pip_install(
        yolov10_dir,
        bin_path.parent,
        ".",
        "torch==2.4.*",
        "onnx",
        "onnxsim",
        "huggingface_hub",
        verbose=verbose,
    )
    _run_download(yolov10_dir, config, python_path, verbose=verbose)
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
    yolov12_dir = directory / "yolov12"
    _run_uv_pip_install(
        yolov12_dir,
        bin_path.parent,
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
        verbose=verbose,
    )
    _run_download(yolov12_dir, config, python_path, verbose=verbose)
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
    yolov13_dir = directory / "yolov13"
    _run_uv_pip_install(
        yolov13_dir,
        bin_path.parent,
        ".",
        "torch==2.4.*",
        "onnx",
        "onnxslim",
        "onnxruntime-gpu",
        "huggingface_hub",
        verbose=verbose,
    )
    _run_download(yolov13_dir, config, python_path, verbose=verbose)
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
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "RT-DETRv1 is a Apache-2.0 licensed model, be aware of license restrictions"
    )
    _git_clone("https://github.com/lyuwenyu/RT-DETR", directory, verbose=verbose)
    rtdetr_dir = directory / "RT-DETR" / "rtdetr_pytorch"
    _run_uv_pip_install(
        rtdetr_dir,
        bin_path.parent,
        "-r",
        "requirements.txt",
        "onnxsim>=0.4",
        "numpy==1.*",
        verbose=verbose,
    )
    _run_download(rtdetr_dir, config, python_path, verbose=verbose)
    _run_patch(
        rtdetr_dir,
        str(
            (
                Path(__file__).parent / "patches" / "rtdetrv1_export_onnx.patch"
            ).resolve()
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
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "RT-DETRv2 is a Apache-2.0 licensed model, be aware of license restrictions"
    )
    _git_clone("https://github.com/lyuwenyu/RT-DETR", directory, verbose=verbose)
    rtdetrv2_dir = directory / "RT-DETR" / "rtdetrv2_pytorch"
    _run_uv_pip_install(
        rtdetrv2_dir,
        bin_path.parent,
        "-r",
        "requirements.txt",
        "onnxsim>=0.4",
        "numpy==1.*",
        verbose=verbose,
    )
    _run_download(rtdetrv2_dir, config, python_path, verbose=verbose)
    _run_patch(
        rtdetrv2_dir,
        str(
            (
                Path(__file__).parent / "patches" / "rtdetrv2_export_onnx.patch"
            ).resolve()
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
    rtdetrv3_dir = directory / "RT-DETRv3"
    _run_uv_pip_install(
        rtdetrv3_dir,
        bin_path.parent,
        "-r",
        "requirements.txt",
        "paddlepaddle==2.6.1",
        "paddle2onnx==1.0.5",
        "onnx==1.13.0",
        "onnxsim>=0.4",
        "scikit-learn",
        "gdown",
        verbose=verbose,
    )
    _run_download(rtdetrv3_dir, config, python_path, verbose=verbose)
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
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "D-FINE is a Apache-2.0 licensed model, be aware of license restrictions"
    )
    _git_clone("https://github.com/Peterande/D-FINE", directory, verbose=verbose)
    dfine_dir = directory / "D-FINE"
    _run_uv_pip_install(
        dfine_dir,
        bin_path.parent,
        "-r",
        "requirements.txt",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "onnx",
        "onnxsim",
        "onnxruntime",
        verbose=verbose,
    )
    _run_download(dfine_dir, config, python_path, verbose=verbose)
    _run_patch(
        dfine_dir,
        str(
            (
                Path(__file__).parent / "patches" / "dfine_export_onnx.patch"
            ).resolve()
        ),
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
    verbose: bool | None = None,
) -> Path:
    LOG.warning("DEIM is a Apache-2.0 licensed model, be aware of license restrictions")
    _git_clone("https://github.com/Intellindust-AI-Lab/DEIM", directory, verbose=verbose)
    deim_dir = directory / "DEIM"
    _run_uv_pip_install(
        deim_dir,
        bin_path.parent,
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
        verbose=verbose,
    )
    _run_download(deim_dir, config, python_path, verbose=verbose)
    _run_patch(
        deim_dir,
        str(
            (Path(__file__).parent / "patches" / "deim_export_onnx.patch").resolve()
        ),
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
    imgsz: int,  # noqa: ARG001
    *,
    verbose: bool | None = None,
) -> Path:
    LOG.warning(
        "RF-DETR is a Apache-2.0 licensed model, be aware of license restrictions"
    )
    LOG.warning("RF-DETR does not support setting alternative input sizes")
    _run_uv_pip_install(
        directory,
        bin_path.parent,
        "rfdetr[onnxexport]",
        verbose=verbose,
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


def _export_deimv2(
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
    LOG.warning("DEIMv2 is a Apache-2.0 licensed model, be aware of license restrictions")
    LOG.warning("DEIMv2 does not support setting alternative input sizes")
    _git_clone("https://github.com/Intellindust-AI-Lab/DEIMv2", directory, verbose=verbose)
    deim_dir = directory / "DEIMv2"
    _run_uv_pip_install(
        deim_dir,
        bin_path.parent,
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
        verbose=verbose,
    )
    _run_download(deim_dir, config, python_path, verbose=verbose)
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
    verbose: bool | None = None,
) -> Path:
    LOG.warning("YOLOX is a Apache-2.0 licensed model, be aware of license restrictions")
    _git_clone("https://github.com/Megvii-BaseDetection/YOLOX", directory, verbose=verbose)
    yolox_dir = directory / "YOLOX"
    _run_uv_pip_install(
        yolox_dir,
        bin_path.parent,
        "-r",
        "requirements.txt",
        "onnxruntime",
        verbose=verbose,
    )
    _run_download(yolox_dir, config, python_path, verbose=verbose)
    subprocess.run(
        [
            python_path,
            "tools/export_onnx.py",
            "--output-name",
            f"{config['name']}.onnx",
            "--name",
            config["version"],
            "--ckpt",
            config["name"] + ".pth",
            "--opset",
            str(opset),
            "--decode_in_inference",
        ],
        cwd=yolox_dir,
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
    *,
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
    elif "deimv2" in model:
        model_path = _export_deimv2(*packet, verbose=verbose)
    elif "yolox" in model:
        model_path = _export_yolox(*packet, verbose=verbose)
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
    accept : bool, optional
        Whether to accept the license terms for the model. If None, will prompt the user.
        If False, will raise an error. If True, will proceed without prompting.
    verbose : bool, optional
        Whether to print verbose output.

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = download_model(model, Path(temp_dir), opset, imgsz, accept=accept, verbose=verbose)
        shutil.copy(model_path, output)

    if verbose is not None:
        LOG.info(f"Model {model} downloaded and converted to ONNX.")
