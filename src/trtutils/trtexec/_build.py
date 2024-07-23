# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ._run import run_trtexec

if TYPE_CHECKING:
    from pathlib import Path

_log = logging.getLogger(__name__)


def build_from_onnx(
    onnx_path: Path,
    output_path: Path,
    use_dla_core: int | None = None,
    *,
    fp16: bool | None = None,
    int8: bool | None = None,
    best: bool | None = None,
    allow_gpu_fallback: bool | None = None,
) -> bool:
    """
    Build an engine from an ONNX file using trtexec.

    Parameters
    ----------
    onnx_path : Path
        The path to the ONNX file to build the engine from.
    output_path : Path
        The path to save the built engine to.
    use_dla_core : int, optional
        The DLA core to use for building the engine, by default None.
        The DLA core should be either 0 or 1 if specified.
    fp16 : bool, optional
        Whether to use FP16 precision for the engine, by default None.
    int8 : bool, optional
        Whether to use INT8 precision for the engine, by default None.
    best : bool, optional
        Whether to use the best precision available for the engine, by default None.
    allow_gpu_fallback : bool, optional
        Whether to allow GPU fallback when a layer is not supported on DLA.
        By default, this is None.

    Returns
    -------
    bool
        Whether the engine was built successfully.

    Raises
    ------
    FileNotFoundError
        If the ONNX file is not found.
    IsADirectoryError
        If the ONNX file is a directory.
    ValueError
        If the ONNX file does not have a .onnx extension.
    ValueError
        If the DLA core is not 0 or 1.

    """
    if not onnx_path.exists():
        err_msg = f"ONNX file not found at {onnx_path}"
        raise FileNotFoundError(err_msg)
    if onnx_path.is_dir():
        err_msg = "ONNX file should not be a directory"
        raise IsADirectoryError(err_msg)
    if onnx_path.suffix != ".onnx":
        err_msg = "ONNX file should have a .onnx extension"
        raise ValueError(err_msg)

    if output_path.exists():
        _log.warning(f"Overwriting existing file at {output_path}")

    if use_dla_core is not None and use_dla_core not in [0, 1]:
        err_msg = "DLA core must be either 0 or 1"
        raise ValueError(err_msg)

    if allow_gpu_fallback and use_dla_core is None:
        _log.warning("GPU fallback enabled without specifying DLA core")

    if best and (fp16 or int8):
        _log.warning("Best precision cannot be used with FP16 or INT8")
        _log.warning("Using best precision level")
        fp16 = False
        int8 = False
    if fp16 and int8:
        _log.warning("FP16 and INT8 precision cannot be used together")
        _log.warning("Using lower precision level")
        fp16 = False

    onnx_path_str = str(onnx_path.resolve())
    output_path_str = str(output_path.resolve())

    command = f"--onnx={onnx_path_str} --saveEngine={output_path_str} --skipInference"
    if use_dla_core:
        command += f" --useDLACore={use_dla_core}"
    if fp16:
        command += " --fp16"
    if int8:
        command += " --int8"
    if allow_gpu_fallback:
        command += " --allowGPUFallback"

    success, _, stderr = run_trtexec(command)

    if not success:
        _log.error(f"Error building engine from ONNX: {stderr}")

    return success
