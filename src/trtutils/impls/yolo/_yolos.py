# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from ._yolo import YOLO

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class YOLOX(YOLO):
    """Implementation of YOLOX."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        *,
        warmup: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
            input_range=(0, 255),
        )


class YOLO7(YOLO):
    """Implementation of YOLO7."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        *,
        warmup: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
            input_range=(0, 1),
        )


class YOLO8(YOLO):
    """Implementation of YOLO8."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        *,
        warmup: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
            input_range=(0, 1),
        )


class YOLO9(YOLO):
    """Implementation of YOLO9."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        *,
        warmup: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
            input_range=(0, 1),
        )


class YOLO10(YOLO):
    """Implementation of YOLO10."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 10,
        *,
        warmup: bool | None = None,
    ) -> None:
        super().__init__(
            engine_path=engine_path,
            warmup_iterations=warmup_iterations,
            warmup=warmup,
            input_range=(0, 1),
        )
