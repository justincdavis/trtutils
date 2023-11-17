from __future__ import annotations

from typing import Any

import numpy as np

from _engine import TRTEngine


class TRTModel:
    def __init__(
        self, 
        engine_path: str,
        preprocess: callable[[list[np.ndarray]], list[np.ndarray]],
        postprocess: callable[[list[np.ndarray]], Any],
        warmup: bool = False,
        warmup_iterations: int = 5,
        dtype: np.number = np.float32,
        device: int = 0,
    ) -> None:
        self._engine = TRTEngine(engine_path, warmup, warmup_iterations, dtype, device)
        self._preprocess = preprocess
        self._postprocess = postprocess
        