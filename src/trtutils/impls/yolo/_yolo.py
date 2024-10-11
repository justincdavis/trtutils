# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from trtutils._model import TRTModel
from ._process import preprocess, postprocess, postprocess_v10

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self


class YOLO:
    """Implementation of YOLO object detectors."""

    def __init__(
            self: Self,
            engine_path: Path | str,
            version: int,
            warmup_iterations: int = 10,
            *,
            warmup: bool | None = None,
        ) -> None:
        """
        Create a YOLO object.
        
        Parameters
        ----------
        engine_path : Path, str
            The Path or str to the compiled TensorRT engine.
        verison : int
            What version of YOLO the compiled engine is.
            Options are: [7, 8, 9, 10]
        warmup_iterations : int
            The number of warmup iterations to perform.
            The default is 10.
        warmup : bool, optional
            Whether or not to perform warmup iterations.
        
        Raises
        ------
        ValueError
            If the version number given is not valid
            
        """
        valid_versions = [7, 8, 9, 10]
        if version not in valid_versions:
            err_msg = f"Invalid version of YOLO given. Received {version}, valid options: {valid_versions}"
            raise ValueError(err_msg)
        postprocessor: Callable[[list[np.ndarray]], list[np.ndarray]] = postprocess_v10 if version >= 10 else postprocess
        self._model = TRTModel(
            engine_path=engine_path,
            postprocess=postprocessor,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
        )
        input_spec = self._model._engine.input_spec[0]
        input_size: tuple[int, int] = tuple(input_spec[0])
        dtype = input_spec[1]
        preprocessor: Callable[[list[np.ndarray]], list[np.ndarray]] = partial(
            preprocess,
            input_shape=input_size,
            dtype=dtype,
        )
        self._model.preprocessor = preprocessor
