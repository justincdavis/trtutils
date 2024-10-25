# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ._run import run_trtexec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_log = logging.getLogger(__name__)


def build_engine(
    weights: Path,
    output: Path,
    use_dla_core: int | None = None,
    shapes: Sequence[tuple[str, tuple[int, ...]]] | None = None,
    workspace: int | None = None,
    add_args: Sequence[str] | None = None,
    *,
    fp16: bool | None = None,
    int8: bool | None = None,
    fp8: bool | None = None,
    best: bool | None = None,
    allow_gpu_fallback: bool | None = None,
) -> bool:
    """
    Build an engine from a weight file using trtexec.

    Parameters
    ----------
    weights : Path
        The path to the weight file to build the engine from.
        Examples are: .onnx, .prototxt
        If a .onnx file is provided, the engine will be built from the ONNX model.
        If a .prototxt file is provided, the engine will be built with random
        weights based on the model architecture.
    output : Path
        The path to save the built engine to.
    use_dla_core : int, optional
        The DLA core to use for building the engine, by default None.
        The DLA core should be either 0 or 1 if specified.
    shapes : tuple[tuple[int, ...], ...], optional
        The input shapes to use for the engine, by default None.
        If provided, the engine will be built with these input shapes.
        The name of the input must also be defined.
        An example could be: (("images", (1, 3, 640, 640)), ...)
    workspace : int, optional
        The workspace size to use for the engine, by default None.
        Expressed in MiB.
    fp16 : bool, optional
        Whether to use FP16 precision for the engine, by default None.
    int8 : bool, optional
        Whether to use INT8 precision for the engine, by default None.
    fp8 : bool, optional
        Whether to use FP8 precision for the engine, by default None.
    best : bool, optional
        Whether to use the best precision available for the engine, by default None.
    allow_gpu_fallback : bool, optional
        Whether to allow GPU fallback when a layer is not supported on DLA.
        By default, this is None.
    add_args : Sequence[str], optional
        Additional arguments to pass to trtexec, by default None.

    Returns
    -------
    bool
        Whether the engine was built successfully.

    Raises
    ------
    FileNotFoundError
        If the weight file is not found.
    IsADirectoryError
        If the weight file is a directory.
    ValueError
        If the weight file does not have a valid extension.
    ValueError
        If the DLA core is not 0 or 1.
    RuntimeError
        If the command generation failed.
    TypeError
        If the input shapes are not integers.

    """
    if not weights.exists():
        err_msg = f"Weight file not found at {weights}"
        raise FileNotFoundError(err_msg)
    if weights.is_dir():
        err_msg = "Weight file should not be a directory"
        raise IsADirectoryError(err_msg)
    # ensure a valid suffix is present
    valid_weights = [
        ".onnx",
        ".prototxt",
    ]
    if weights.suffix not in valid_weights:
        err_msg = "Weights file has invalid extension."
        err_msg += f" Supported extensions are: {valid_weights}."
        err_msg += f" Found: {weights.suffix}"
        raise ValueError(err_msg)

    if output.exists():
        _log.warning(f"Overwriting existing file at {output}")

    if use_dla_core is not None and use_dla_core not in [0, 1]:
        err_msg = "DLA core must be either 0 or 1"
        raise ValueError(err_msg)

    if allow_gpu_fallback and use_dla_core is None:
        _log.warning("GPU fallback enabled without specifying DLA core")

    if best and (fp16 or int8 or fp8):
        _log.warning("Best precision enabled with other precisions also being enabled.")
        _log.warning("Best precision level ENABLES ALL precisions")
    if fp16 and int8:
        _log.warning(
            "FP16 and INT8 precision cannot be used together. Using lower precision level.",
        )
        fp16 = False
    if fp16 and fp8:
        _log.warning(
            "FP16 and FP8 precision cannot be used together. Using lower precision level.",
        )
        fp16 = False
    if int8 and fp8:
        _log.warning(
            "INT8 and FP8 precision cannot be used together. Using lower precision level.",
        )
        int8 = False

    # resolve the model and output paths
    weights_path_str = str(weights.resolve())
    output_path_str = str(output.resolve())

    # parse any shapes input if it exists
    shapes_str = ""
    if shapes:
        for name, shape in shapes:
            dim_str = ""
            for dim in shape:
                if not isinstance(dim, int):
                    err_msg = "Input shapes must be integers"
                    raise TypeError(err_msg)
                dim_str += f"{dim}x"
            dim_str = dim_str[:-1]
            shape_str = f"{name}:{dim_str}"
            shapes_str += f"{shape_str},"
        if len(shapes_str) > 0:
            shapes_str = shapes_str[:-1]

    # generate initial command with weight input
    command = ""
    if weights.suffix == ".onnx":
        command += f" --onnx={weights_path_str}"
    elif weights.suffix == ".prototxt":
        command += f" --deploy={weights_path_str}"

    # check length, if zero something went wrong
    if len(command) == 0:
        err_msg = "After generating command, no weight input was found."
        err_msg += " This is an internal error, please report."
        raise RuntimeError(err_msg)

    command += f" --saveEngine={output_path_str} --skipInference"
    if isinstance(use_dla_core, int):
        command += f" --useDLACore={use_dla_core}"
        command += " --memPoolSize=dlaSRAM:1"
    if fp16:
        command += " --fp16"
    if int8:
        command += " --int8"
    if fp8:
        command += " --fp8"
    if best:
        command += " --best"
    if allow_gpu_fallback:
        command += " --allowGPUFallback"
    if shapes_str:
        command += f" --shapes={shapes_str}"
    if workspace:
        command += f" --workspace={workspace}"

    # handle additional arguments
    if add_args:
        for arg in add_args:
            command += f" {arg}"

    # debug print
    _log.debug(f"TRTEXEC Command: {command}")

    success, _, stderr = run_trtexec(command)

    if not success:
        _log.error(f"Error building engine from ONNX: {stderr}")

    return success
