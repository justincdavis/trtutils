# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path


def eval_dla(onnx_path: Path | str) -> bool:
    """
    Whether or not the entire model can be run on a DLA.
    
    Parameters
    ----------
    onnx_path : Path, str
        The path to the onnx file.
    
    Returns
    -------
    bool
        Whether or not the model will all run on DLA.
    
    """
    return False
