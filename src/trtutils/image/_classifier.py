# # Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
# #
# # MIT License
# from __future__ import annotations

# import contextlib
# from typing import TYPE_CHECKING

# from trtutils._engine import TRTEngine

# if TYPE_CHECKING:
#     from pathlib import Path

#     from typing_extensions import Self

#     with contextlib.suppress(Exception):
#         try:
#             import cuda.bindings.runtime as cudart
#         except (ImportError, ModuleNotFoundError):
#             from cuda import cudart


# class Classifier:
#     """A wrapper around classification models."""

#     def __init__(
#         self: Self,
#         engine: TRTEngine | Path | str,
#         input_range: tuple[float, float] = (0.0, 1.0),
#         preprocessor: str = "trt",
#         resize_method: str = "letterbox",
#         conf_thres: float = 0.1,
#         warmup_iterations: int = 5,
#         backend: str = "auto",
#         stream: cudart.cudaStream_t | None = None,
#         dla_core: int | None = None,
#         *,
#         warmup: bool | None = None,
#         pagelocked_mem: bool | None = None,
#         unified_mem: bool | None = None,
#         no_warn: bool | None = None,
#         verbose: bool | None = None,
#     ) -> None:
#         """
#         Initialize the classifier.

#         Parameters
#         ----------
#         engine : TRTEngine | Path | str
#             The engine to use for the classifier.

#         """
#         self._engine = engine if isinstance(engine, TRTEngine) else TRTEngine(engine)
