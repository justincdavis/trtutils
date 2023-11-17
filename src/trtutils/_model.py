from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from _engine import TRTEngine

if TYPE_CHECKING:
    from typing_extensions import Self


class TRTModel:
    def __init__(
        self: Self,
        engine_path: str,
        preprocess: callable[[list[np.ndarray]], list[np.ndarray]],
        postprocess: callable[[list[np.ndarray]], Any],
        warmup: bool | None = None,
        warmup_iterations: int = 5,
        dtype: np.number = np.float32,
        device: int = 0,
    ) -> None:
        self._engine = TRTEngine(engine_path, warmup, warmup_iterations, dtype, device)
        self._preprocess = preprocess
        self._postprocess = postprocess
