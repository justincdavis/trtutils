# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from trtutils._log import LOG

from . import _simplify
from ._models import (
    export_deim,
    export_deimv2,
    export_depth_anything_v2,
    export_dfine,
    export_rfdetr,
    export_rtdetrv1,
    export_rtdetrv2,
    export_rtdetrv3,
    export_torchvision_classifier,
    export_ultralytics,
    export_yolov7,
    export_yolov9,
    export_yolov10,
    export_yolov12,
    export_yolov13,
    export_yolox,
)
from ._tools import (
    check_uv_version,
    export_requirements,
    make_venv,
)


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


def download_model(
    model: str,
    directory: Path,
    opset: int = 17,
    imgsz: int | None = None,
    requirements_export: Path | None = None,
    *,
    simplify: bool | None = None,
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
    simplify : bool, optional
        Whether to simplify the model after exporting.
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

    python_path, bin_path = make_venv(directory, no_cache=no_uv_cache, verbose=verbose)
    requirements_export_path = Path(requirements_export) if requirements_export is not None else None

    # Determine which export function to use
    export_func = None
    if config.get("url") == "torchvision_classifier":
        export_func = export_torchvision_classifier
    elif config.get("url") == "ultralytics":
        export_func = export_ultralytics
    elif "deim" in model and "deimv2" not in model:
        export_func = export_deim
    elif "deimv2" in model:
        export_func = export_deimv2
    elif "yolox" in model:
        export_func = export_yolox
    elif "yolov7" in model:
        export_func = export_yolov7
    elif "yolov9" in model:
        export_func = export_yolov9
    elif "yolov10" in model:
        export_func = export_yolov10
    elif "yolov12" in model:
        export_func = export_yolov12
    elif "yolov13" in model:
        export_func = export_yolov13
    elif "rtdetrv1" in model:
        export_func = export_rtdetrv1
    elif "rtdetrv2" in model:
        export_func = export_rtdetrv2
    elif "rtdetrv3" in model:
        export_func = export_rtdetrv3
    elif "dfine" in model:
        export_func = export_dfine
    elif "rfdetr" in model:
        export_func = export_rfdetr
    elif "depth_anything_v2" in model:
        export_func = export_depth_anything_v2

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
        export_requirements(bin_path.parent, requirements_export_path, verbose=verbose)

    if simplify:
        _simplify.simplify(
            model_path,
            directory=directory,
            bin_path=bin_path,
            no_uv_cache=no_uv_cache,
            verbose=verbose,
        )

    return model_path.with_name(model + model_path.suffix)


def download(
    model: str,
    output: Path,
    opset: int = 17,
    imgsz: int | None = None,
    requirements_export: Path | None = None,
    *,
    simplify: bool | None = None,
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
    simplify : bool, optional
        Whether to simplify the model after exporting.
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
    check_uv_version()

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = download_model(
            model,
            Path(temp_dir),
            opset,
            imgsz,
            requirements_export=requirements_export,
            simplify=simplify,
            no_cache=no_cache,
            no_uv_cache=no_uv_cache,
            no_warn=no_warn,
            accept=accept,
            verbose=verbose,
        )
        shutil.copy(model_path, output)

    if verbose is not None:
        LOG.info(f"Model {model} downloaded and converted to ONNX.")
