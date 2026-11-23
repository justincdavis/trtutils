# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from trtutils._log import LOG
from trtutils.download._tools import (
    CLASSES_SUFFIX,
    _640,
    get_weights_cache_dir,
    handle_imgsz,
    run_cmd,
    run_uv_pip_install,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# the vocabulary of a text-prompt YOLOE export when no classes are given
COCO80_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)  # fmt: skip


def export_ultralytics(
    directory: Path,
    config: dict[str, str],
    python_path: Path,  # noqa: ARG001
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None = None,
    *,
    batch: int = 1,
    dynamic: bool = False,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    imgsz = handle_imgsz(imgsz, int(config.get("imgsz", _640)), "Ultralytics")
    run_uv_pip_install(
        directory,
        bin_path.parent,
        "ultralytics",
        no_cache=no_uv_cache,
        verbose=verbose,
    )
    modelname = f"model={config['name']}"

    # Handle caching of ultralytics downloaded .pt weights
    weights_cache_dir = get_weights_cache_dir()
    pt_filename = config["name"] + ".pt"
    cache_key = f"ultralytics_{pt_filename}"
    cached_pt_file = weights_cache_dir / cache_key
    target_pt_file = directory / pt_filename

    # Check if .pt file exists in cache
    if cached_pt_file.exists() and not no_cache:
        LOG.info(f"Using cached ultralytics weights: {pt_filename}")
        shutil.copy(cached_pt_file, target_pt_file)
        modelname = f"model={pt_filename}"

    cmd = [
        str(bin_path / "yolo"),
        "export",
        modelname,
        "format=onnx",
        f"opset={opset}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"dynamic={dynamic}",
        "simplify=True",
    ]
    # yolo26 task heads export end-to-end by default; the task decoders want the raw head
    if "end2end" in config:
        cmd.append(f"end2end={config['end2end']}")
    run_cmd(cmd, cwd=directory, verbose=verbose)

    # Cache the .pt file if it was downloaded and not already cached
    if target_pt_file.exists() and not cached_pt_file.exists():
        shutil.copy(target_pt_file, cached_pt_file)
        LOG.info(f"Cached ultralytics weights: {pt_filename}")

    model_path = directory / (config["name"] + ".onnx")
    new_model_path = model_path.with_name(model + model_path.suffix)
    shutil.move(model_path, new_model_path)
    return new_model_path


def _restore_cached(directory: Path, filename: str, *, no_cache: bool | None) -> None:
    cached = get_weights_cache_dir() / f"ultralytics_{filename}"
    if cached.exists() and not no_cache:
        LOG.info(f"Using cached ultralytics file: {filename}")
        shutil.copy(cached, directory / filename)


def _cache(directory: Path, filename: str) -> None:
    cached = get_weights_cache_dir() / f"ultralytics_{filename}"
    if (directory / filename).exists() and not cached.exists():
        shutil.copy(directory / filename, cached)
        LOG.info(f"Cached ultralytics file: {filename}")


def _export_yoloe(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None,
    classes: Sequence[str] | None,
    *,
    batch: int,
    dynamic: bool,
    no_cache: bool | None,
    no_uv_cache: bool | None,
    verbose: bool | None,
) -> Path:
    imgsz = handle_imgsz(imgsz, int(config.get("imgsz", _640)), "YOLOE")
    run_uv_pip_install(directory, bin_path.parent, "yoloe", no_cache=no_uv_cache, verbose=verbose)
    # the weights and the text encoder are fetched into the working directory
    files = [config["name"] + ".pt"]
    if classes is not None:
        files.append(config["text_encoder"])
    for filename in files:
        _restore_cached(directory, filename, no_cache=no_cache)

    end2end = f", end2end={config['end2end']}" if "end2end" in config else ""
    script_content = f"""\
from pathlib import Path

import torch
from ultralytics import YOLOE

# exports never need gradients
torch.set_grad_enabled(False)
model = YOLOE("{files[0]}")
names = {None if classes is None else list(classes)!r}
# text prompts are embedded once and baked into the head as its classes
if names is not None:
    model.set_classes(names, model.get_text_pe(names))
model.export(
    format="onnx",
    opset={opset},
    imgsz={imgsz},
    batch={batch},
    dynamic={dynamic},
    simplify=True{end2end},
)
names = [model.names[i] for i in range(len(model.names))]
Path("{config["name"]}{CLASSES_SUFFIX}").write_text("".join(f"{{n}}\\n" for n in names))
"""
    script_path = directory / "_export_yoloe.py"
    script_path.write_text(script_content)
    run_cmd([python_path, str(script_path)], cwd=directory, verbose=verbose)
    for filename in files:
        _cache(directory, filename)

    model_path = directory / f"{model}.onnx"
    shutil.move(directory / f"{config['name']}.onnx", model_path)
    shutil.move(
        directory / f"{config['name']}{CLASSES_SUFFIX}", model_path.with_suffix(CLASSES_SUFFIX)
    )
    return model_path


def export_yoloe(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None = None,
    *,
    classes: Sequence[str] | None = None,
    batch: int = 1,
    dynamic: bool = False,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    """Export a text-prompt YOLOE model with ``classes`` (default: the 80 COCO names) baked in."""
    return _export_yoloe(
        directory,
        config,
        python_path,
        bin_path,
        model,
        opset,
        imgsz,
        list(classes) if classes is not None else list(COCO80_NAMES),
        batch=batch,
        dynamic=dynamic,
        no_cache=no_cache,
        no_uv_cache=no_uv_cache,
        verbose=verbose,
    )


def export_yoloe_pf(
    directory: Path,
    config: dict[str, str],
    python_path: Path,
    bin_path: Path,
    model: str,
    opset: int,
    imgsz: int | None = None,
    *,
    batch: int = 1,
    dynamic: bool = False,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,  # noqa: ARG001
    verbose: bool | None = None,
) -> Path:
    """Export a prompt-free YOLOE model with its built-in vocabulary."""
    return _export_yoloe(
        directory,
        config,
        python_path,
        bin_path,
        model,
        opset,
        imgsz,
        None,
        batch=batch,
        dynamic=dynamic,
        no_cache=no_cache,
        no_uv_cache=no_uv_cache,
        verbose=verbose,
    )
