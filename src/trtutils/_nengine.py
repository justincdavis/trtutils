from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING
from functools import cached_property

from .core import (
    allocate_bindings,
    create_engine,
    memcpy_device_to_host,
    memcpy_host_to_device,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    import numpy as np


class TRTEngine:
    """Implements a generic interface for TensorRT engines."""

    def __init__(self: Self, engine_path: Path | str):
        """
        Load the TensorRT engine from a file.

        Parameters
        ----------
        engine_path : Path | str
            The path to the serialized engine file.
        
        """
        # Load TRT engine
        self._engine, self._context, self._logger = create_engine(engine_path)

        # allocate memory for inputs and outputs
        self._inputs, self._outputs, self._allocations, self._batch_size = allocate_bindings(
            self._engine,
            self._context,
            self._logger,
        )

    def __del__(self: Self) -> None:
        def _del(obj: object, attr: str) -> None:
            with contextlib.suppress(AttributeError):
                delattr(obj, attr)

        for binding in self._inputs + self._outputs:
            # attempt to free each allocation
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
    def output_spec(self: Self) -> list[tuple[list[int], np.dtype]]:
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.

        Returns
        -------
        list[tuple[list[int], np.dtype]]
            A list with two items per element, the shape and (numpy) datatype of each output tensor.
        
        """
        return [(o.shape, o.dtype) for o in self._outputs]    

    def __call__(self: Self, data: list[np.ndarray]) -> list[np.ndarray]:
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
        return self.execute(data)

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
