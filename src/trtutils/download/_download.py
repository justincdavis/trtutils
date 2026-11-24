# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from trtutils._log import LOG

from . import _make_static, _simplify
from ._models import EXPORTERS
from ._tools import (
    check_uv_version,
    export_requirements,
    handle_batch,
    make_venv,
    normalize_imgsz,
)

# copyleft or non-commercial licenses get a louder notice
_RESTRICTIVE_LICENSES = ("GPL", "-NC", "non-commercial")


@lru_cache(maxsize=1)
def load_model_configs() -> dict[str, dict[str, dict[str, str]]]:
    """
    Load every model family's configuration.

    A config file may hold a ``"_defaults"`` object; its keys are merged into
    each entry of that file (entry keys win) and it is not returned as a model.

    Returns
    -------
    dict[str, dict[str, dict[str, str]]]
        Family name -> model name -> config fields.

    """
    configs_dir = Path(__file__).parent / "configs"
    model_configs: dict[str, dict[str, dict[str, str]]] = {}

    for config_path in configs_dir.glob("*.json"):
        model_type = config_path.stem
        try:
            with config_path.open() as f:
                family = json.load(f)
        except (FileNotFoundError, KeyError) as e:
            LOG.warning(f"Failed to load configuration file {config_path.name}: {e}")
            continue
        defaults = family.pop("_defaults", {})
        model_configs[model_type] = {name: {**defaults, **entry} for name, entry in family.items()}

    return model_configs


def _license_notice(model: str, license_name: str) -> str:
    if license_name == "unknown":
        return (
            f"The license of the {model} weights is not stated upstream, "
            "check the source repository before using them"
        )
    notice = f"{model} weights are {license_name} licensed"
    if any(tag in license_name for tag in _RESTRICTIVE_LICENSES):
        return (
            f"{notice}: a RESTRICTIVE license (copyleft or non-commercial), "
            "review its terms before using or distributing the model"
        )
    return f"{notice}, be aware of license restrictions"


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


def download_model(  # noqa: C901, PLR0912
    model: str,
    directory: Path,
    opset: int = 17,
    imgsz: int | tuple[int, int] | None = None,
    requirements_export: Path | None = None,
    *,
    batch: int | None = None,
    dynamic: bool | None = None,
    simplify: Sequence[str] | bool | None = None,
    make_static: bool | None = None,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
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
    imgsz : int | tuple[int, int], optional
        The image size to use for the model: one int for a square image, or
        ``(height, width)``. Only exporters registered with ``hw=True`` accept a
        non-square size. By default, the model will use the default image size for the model.
    requirements_export : Path, optional
        Export the created virtual environment's requirements to this path using uv pip freeze.
    batch : int, optional
        The batch size to export at, by default 1. Only the ultralytics,
        YOLOv10/12/13, torchvision, and timm exporters accept a batch above 1.
    dynamic : bool, optional
        Leave the batch dimension symbolic (ultralytics and YOLOv10/12/13 also
        leave height and width symbolic). Same exporters as batch. By default
        only torchvision classifiers export dynamic, unless a batch is given.
    simplify : Sequence[str] or bool, optional
        Whether and how to simplify the model after exporting.
        If True, uses default tools (polygraphy, onnxslim).
        If a sequence of tool names, runs those tools in the given order.
        Valid tool names: "polygraphy", "onnxslim", "onnxsim".
        If False or None, no simplification is performed.
    make_static : bool, optional
        Set any dynamic dimensions in the ONNX model to fixed values (e.g. batch size to 1).
        If True, all dynamic or symbolic dimensions are replaced with a static value of 1.
        If False or None, no modification is performed.
    no_cache : bool, optional
        Whether to disable caching of downloaded weights and repos.
    no_uv_cache : bool, optional
        Whether to disable caching of uv packages.
    no_warn : bool, optional
        Whether to disable warnings for the model, including its license notice.
    verbose : bool, optional
        Whether to print verbose output.

    Returns
    -------
    Path
        The path to the exported model inside the directory.

    Raises
    ------
    ValueError
        If the model is not supported, the batch is not positive, dynamic is
        combined with a batch above 1 or make_static, or imgsz is non-square and the
        model's exporter only takes square sizes.
    NotImplementedError
        If batch > 1 or dynamic is requested from an exporter that cannot honor it.

    """
    if dynamic and make_static:
        err_msg = "dynamic and make_static are mutually exclusive"
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

    exporter = EXPORTERS[config["exporter"]]
    if imgsz is not None:
        imgsz = normalize_imgsz(imgsz)
        if isinstance(imgsz, tuple) and not exporter.hw:
            err_msg = (
                f"{model} exports only square images, got imgsz (height, width) of {imgsz}; "
                "pass a single size"
            )
            raise ValueError(err_msg)

    if dynamic is None:
        dynamic = exporter.dynamic_default and batch is None
    batch, dynamic = handle_batch(batch, model, dynamic=dynamic, supported=exporter.batch)
    batch_kwargs: dict[str, Any] = {"batch": batch, "dynamic": dynamic} if exporter.batch else {}

    if not no_warn:
        LOG.warning(_license_notice(model, config["license"]))

    python_path, bin_path = make_venv(directory, no_cache=no_uv_cache, verbose=verbose)
    requirements_export_path = Path(requirements_export) if requirements_export is not None else None

    model_path = exporter.fn(
        directory,
        config,
        python_path,
        bin_path,
        model,
        opset,
        imgsz,
        **batch_kwargs,
        no_cache=no_cache,
        no_uv_cache=no_uv_cache,
        no_warn=no_warn,
        verbose=verbose,
    )

    if requirements_export_path is not None:
        export_requirements(bin_path.parent, requirements_export_path, verbose=verbose)

    if simplify:
        if isinstance(simplify, bool):
            tools = None
        elif isinstance(simplify, str):
            tools = [simplify]
        else:
            tools = list(simplify)
        _simplify.simplify(
            model_path,
            tools=tools,
            directory=directory,
            bin_path=bin_path,
            no_uv_cache=no_uv_cache,
            verbose=verbose,
        )

    if make_static:
        _make_static.make_static(
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
    imgsz: int | tuple[int, int] | None = None,
    requirements_export: Path | None = None,
    *,
    batch: int | None = None,
    dynamic: bool | None = None,
    simplify: Sequence[str] | bool | None = None,
    make_static: bool | None = None,
    no_cache: bool | None = None,
    no_uv_cache: bool | None = None,
    no_warn: bool | None = None,
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
    imgsz : int | tuple[int, int], optional
        The image size to use for the model: one int for a square image, or
        ``(height, width)``. Only exporters registered with ``hw=True`` accept a
        non-square size. By default, the model will use the default image size for the model.
    requirements_export : Path, optional
        Export the created virtual environment's requirements to this path using uv pip freeze.
    batch : int, optional
        The batch size to export at, by default 1. Only the ultralytics,
        YOLOv10/12/13, torchvision, and timm exporters accept a batch above 1.
    dynamic : bool, optional
        Leave the batch dimension symbolic (ultralytics and YOLOv10/12/13 also
        leave height and width symbolic). Same exporters as batch. By default
        only torchvision classifiers export dynamic, unless a batch is given.
    simplify : Sequence[str] or bool, optional
        Whether and how to simplify the model after exporting.
        If True, uses default tools (polygraphy, onnxslim).
        If a sequence of tool names, runs those tools in the given order.
        Valid tool names: "polygraphy", "onnxslim", "onnxsim".
        If False or None, no simplification is performed.
    make_static : bool, optional
        Set any dynamic dimensions in the ONNX model to fixed values (e.g. batch size to 1).
        If True, all dynamic or symbolic dimensions are replaced with a static value of 1.
        If False or None, no modification is performed.
    no_cache : bool, optional
        Whether to disable caching of downloaded weights and repos.
    no_uv_cache : bool, optional
        Whether to disable caching of uv packages.
    no_warn : bool, optional
        Whether to disable warnings for the model, including its license notice.
    verbose : bool, optional
        Whether to print verbose output.

    """
    check_uv_version()
    output = Path(output)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = download_model(
            model,
            Path(temp_dir),
            opset,
            imgsz,
            requirements_export=requirements_export,
            batch=batch,
            dynamic=dynamic,
            simplify=simplify,
            make_static=make_static,
            no_cache=no_cache,
            no_uv_cache=no_uv_cache,
            no_warn=no_warn,
            verbose=verbose,
        )
        shutil.copy(model_path, output)
        # onnx external weights (>2GB models) live in a sidecar referenced by filename
        sidecar = model_path.with_name(model_path.name + ".data")
        if sidecar.exists():
            shutil.copy(sidecar, output.parent / sidecar.name)

    if verbose is not None:
        LOG.info(f"Model {model} downloaded and converted to ONNX.")
