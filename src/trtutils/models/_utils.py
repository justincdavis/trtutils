# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

with contextlib.suppress(Exception):
    import tensorrt as trt

from trtutils.download._download import load_model_configs
from trtutils.download._download import download as _download
from trtutils.builder._build import build_engine

if TYPE_CHECKING:
    from collections.abc import Callable


def get_valid_models(model_type: str) -> list[str]:
    """
    Get the valid models for a given model type.

    Parameters
    ----------
    model_type : str
        The model type to get the valid models for.

    Returns
    -------
    list[str]
        The valid models for the given model type.

    """
    # model family -> specific models -> config data (ignore here)
    model_configs: dict[str, dict[str, dict[str, str]]] = load_model_configs()
    
    return list(model_configs[model_type].keys())


def download_model_internal(
    model_type: str,
    friendly_name: str,
    model: str,
    output: Path | str,
    imgsz: int = 640,
    opset: int = 17,
    *,
    no_cache: bool | None = None,
    verbose: bool | None = None,
) -> None:
    """
    Validate model name for a family and delegate download to the core downloader.

    Parameters
    ----------
    model_type : str
        Key of the model family as defined in download configs (e.g., 'yolov8').
    friendly_name : str
        Human-friendly family name for error messaging (e.g., 'YOLOv8').
    model : str
        Specific model identifier within the family.
    output : Path | str
        Output file path for the resulting ONNX model.
    imgsz : int, optional
        Image size to use during export.
    opset : int, optional
        ONNX opset to use during export.
    no_cache : bool, optional
        Disable caching of downloaded weights and repos.
    verbose : bool, optional
        Print verbose output.
    """
    if model not in get_valid_models(model_type):
        err_msg = f"Model {model} is not a valid {friendly_name} model."
        raise ValueError(err_msg)
    _download(
        model=model,
        output=Path(output),
        imgsz=imgsz,
        opset=opset,
        no_cache=no_cache,
        verbose=verbose,
    )


def build_internal(
    onnx: Path | str,
    output: Path | str,
    shapes: list[tuple[str, tuple[int, ...]]] | None = None,
    hooks: list[Callable[[trt.INetworkDefinition], trt.INetworkDefinition]] | None = None,
) -> None:
    """
    Build a TensorRT engine from an ONNX model.

    Parameters
    ----------
    onnx : Path | str
        Path to the ONNX model to compile.
    output : Path | str
        Path to save the compiled engine to.
    shapes : list[tuple[str, tuple[int, ...]]] | None = None,
        A list of (input_name, shape) pairs to specify the shapes of the input layers.
    hooks : list[Callable[[trt.INetworkDefinition], trt.INetworkDefinition]] | None = None,
        A list of hooks to apply to the network definition.

    """
    onnx_path = Path(onnx)
    output_path = Path(output)

    build_engine(
        onnx=onnx_path,
        output=output_path,
        shapes=shapes,
        hooks=hooks,
        fp16=True,
    )
