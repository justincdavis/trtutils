# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
from functools import cached_property
from typing import TYPE_CHECKING

from .core import (
    TRTEngineInterface,
    allocate_bindings,
    memcpy_device_to_host,
    memcpy_host_to_device,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from typing_extensions import Self


class TRTEngine(TRTEngineInterface):
    """Implements a generic interface for TensorRT engines."""

    def __init__(
        self: Self,
        engine_path: Path | str,
        warmup_iterations: int = 5,
        *,
        warmup: bool | None = None,
    ) -> None:
        """
        Load the TensorRT engine from a file.

        Parameters
        ----------
        engine_path : Path | str
            The path to the serialized engine file.
        warmup : bool, optional
            Whether to do warmup iterations, by default None
            If None, warmup will be set to False
        warmup_iterations : int, optional
            The number of warmup iterations to do, by default 5

        """
        super().__init__(engine_path)

        # allocate memory for inputs and outputs
        self._inputs, self._outputs, self._allocations, self._batch_size = (
            allocate_bindings(
                self._engine,
                self._context,
                self._logger,
            )
        )

        if warmup:
            for _ in range(warmup_iterations):
                self.mock_execute()

    def __del__(self: Self) -> None:
        def _del(obj: object, attr: str) -> None:
            with contextlib.suppress(AttributeError):
                delattr(obj, attr)

        with contextlib.suppress(AttributeError):
            for binding in self._inputs:
                with contextlib.suppress(RuntimeError):
                    binding.free()
        with contextlib.suppress(AttributeError):
            for binding in self._outputs:
                with contextlib.suppress(RuntimeError):
                    binding.free()

        attrs = ["_context", "_engine"]
        for attr in attrs:
            _del(self, attr)

    @cached_property
    def input_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each input tensor.

        """
        return [(i.shape, i.dtype) for i in self._inputs]

    @cached_property
    def input_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the input tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each input tensor.

        """
        return [tuple(i.shape) for i in self._inputs]

    @cached_property
    def input_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the input tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each input tensor.

        """
        return [i.dtype for i in self._inputs]

    @cached_property
    def output_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each output tensor.

        """
        return [(o.shape, o.dtype) for o in self._outputs]

    @cached_property
    def output_shapes(self: Self) -> list[tuple[int, ...]]:
        """
        Get the shapes for the output tensors of the network.

        Returns
        -------
        list[tuple[int, ...]]
            A list with the shape of each output tensor.

        """
        return [tuple(o.shape) for o in self._outputs]

    @cached_property
    def output_dtypes(self: Self) -> list[np.dtype]:
        """
        Get the datatypes for the output tensors of the network.

        Returns
        -------
        list[np.dtype]
            A list with the datatype of each output tensor.

        """
        return [o.dtype for o in self._outputs]

    def execute(self: Self, data: list[np.ndarray]) -> list[np.ndarray]:
        """
        Execute the network with the given inputs.

        Parameters
        ----------
        data : list[np.ndarray]
            The inputs to the network.

        Returns
        -------
        list[np.ndarray]
            The outputs of the network.

        """
        # Copy inputs
        for i_idx in range(len(self._inputs)):
            memcpy_host_to_device(
                self._inputs[i_idx].allocation,
                data[i_idx],
            )
        # execute
        self._context.execute_v2(self._allocations)
        # Copy outputs
        for o_idx in range(len(self._outputs)):
            memcpy_device_to_host(
                self._outputs[o_idx].host_allocation,
                self._outputs[o_idx].allocation,
            )
        # return
        return [o.host_allocation for o in self._outputs]
