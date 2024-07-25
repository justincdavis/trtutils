# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self


def TRTEngineInterface(ABC):
    @abstractmethod
    def __init__(self, engine_path: str | Path) -> None:
        """
        Initialize the TensorRT engine.

        Parameters
        ----------
        engine_path : str
            The path to the serialized engine file.

        """
        pass

    @abstractmethod
    def __del__(self) -> None:
        """
        Free the engine resources.

        """
        pass

    @abstractmethod
    def __call__(self: Self, data: list[np.ndarray]) -> list[np.ndarray]:
        """
        Run inference on the engine.

        """
        pass

    @abstractmethod
    def execute(self: Self, data: list[np.ndarray]) -> list[np.ndarray]:
        """
        Execute the engine on the given data.

        Parameters
        ----------
        data : list[np.ndarray]
            The input data.

        Returns
        -------
        list[np.ndarray]
            The output data.

        """
        pass

    @abstractmethod
    def mock_execute(self: Self, data: list[np.ndarray] | None = None) -> list[np.ndarray]:
        """
        Execute the engine with random data.

        Parameters
        ----------
        data : list[np.ndarray], optional
            The input data, by default None
            If None, random data will be generated

        Returns
        -------
        list[np.ndarray]
            The output data.

        """
        pass