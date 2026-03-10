# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import numpy as np
from modelopt.onnx.quantization import quantize as _modelopt_quantize

from trtutils._log import LOG


def quantize_onnx(
    onnx_path: Path | str,
    output_path: Path | str,
    calibration_data: Path | str,
    quantize_mode: str = "int8",
    calibration_method: str = "max",
    *,
    calibrate_per_node: bool = False,
    verbose: bool | None = None,
) -> Path:
    """
    Quantize an ONNX model using NVIDIA modelopt.

    Parameters
    ----------
    onnx_path : Path, str
        Path to the input ONNX model.
    output_path : Path, str
        Path to save the quantized ONNX model.
    calibration_data : Path, str
        Path to the calibration data .npy file.
    quantize_mode : str, optional
        Quantization mode, by default "int8".
    calibration_method : str, optional
        Calibration method, by default "max".
    calibrate_per_node : bool, optional
        Whether to calibrate per node, by default False.
    verbose : bool, optional
        Whether to print verbose output, by default None.

    Returns
    -------
    Path
        The resolved path to the quantized ONNX model.

    Raises
    ------
    FileNotFoundError
        If the ONNX model or calibration data file does not exist.
    ValueError
        If the quantize_mode or calibration_method is not valid.

    """
    valid_quantize_modes = ("int4", "int8", "fp8")
    valid_calibration_methods = ("max", "entropy", "percentile", "mse")

    if quantize_mode not in valid_quantize_modes:
        err_msg = f"Invalid quantize_mode: {quantize_mode!r}, options are: {valid_quantize_modes}"
        raise ValueError(err_msg)

    if calibration_method not in valid_calibration_methods:
        err_msg = (
            f"Invalid calibration_method: {calibration_method!r}, "
            f"options are: {valid_calibration_methods}"
        )
        raise ValueError(err_msg)

    onnx_file = Path(onnx_path).resolve()
    if not onnx_file.exists():
        err_msg = f"ONNX model not found: {onnx_file}"
        raise FileNotFoundError(err_msg)

    calib_file = Path(calibration_data).resolve()
    if not calib_file.exists():
        err_msg = f"Calibration data not found: {calib_file}"
        raise FileNotFoundError(err_msg)

    if quantize_mode == "fp8":
        LOG.warning(
            "FP8 quantization requires compute capability >= 8.9 (Ada Lovelace / Hopper or newer)."
        )

    output = Path(output_path).resolve()

    if verbose:
        LOG.debug(
            f"Quantizing {onnx_file} with mode={quantize_mode!r}, method={calibration_method!r}"
        )

    calib_array = np.load(str(calib_file))

    _modelopt_quantize(
        onnx_path=str(onnx_file),
        quantize_mode=quantize_mode,
        calibration_data=calib_array,
        calibration_method=calibration_method,
        output_path=str(output),
        calibrate_per_node=calibrate_per_node,
    )

    LOG.info(f"Quantized model saved to {output}")

    return output
