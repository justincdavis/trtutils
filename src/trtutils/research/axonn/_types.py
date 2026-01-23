# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Data structures for AxoNN optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self


class ProcessorType(Enum):
    """Enum representing available processor types."""

    GPU = "GPU"
    DLA = "DLA"


@dataclass
class Layer:
    """
    Represents a neural network layer with its properties.

    Attributes
    ----------
    index : int
        The layer index in the network.
    name : str
        The name of the layer.
    layer_type : str
        The type of the layer (e.g., "Convolution", "Pooling").
    output_tensor_size : int
        Size of the output tensor in bytes.
    can_run_on_dla : bool
        Whether the layer can run on DLA.
    input_tensor_size : int
        Size of the input tensor in bytes.

    """

    index: int
    name: str
    layer_type: str
    output_tensor_size: int
    can_run_on_dla: bool
    input_tensor_size: int = 0

    def __str__(self: Self) -> str:
        dla_str = "DLA-compatible" if self.can_run_on_dla else "GPU-only"
        return f"Layer({self.index}: {self.name}, {self.layer_type}, {dla_str})"

    def __repr__(self: Self) -> str:
        return (
            f"Layer(index={self.index}, name={self.name!r}, "
            f"layer_type={self.layer_type!r}, output_tensor_size={self.output_tensor_size}, "
            f"can_run_on_dla={self.can_run_on_dla}, input_tensor_size={self.input_tensor_size})"
        )


@dataclass
class LayerCost:
    """
    Stores execution costs for a layer on different processors.

    Attributes
    ----------
    layer_idx : int
        The index of the layer.
    layer_name : str
        The name of the layer.
    gpu_time_ms : float
        Execution time on GPU in milliseconds.
    gpu_energy_mj : float
        Energy consumption on GPU in millijoules.
    dla_time_ms : float | None
        Execution time on DLA in milliseconds, None if not DLA-compatible.
    dla_energy_mj : float | None
        Energy consumption on DLA in millijoules, None if not DLA-compatible.

    """

    layer_idx: int
    layer_name: str
    gpu_time_ms: float
    gpu_energy_mj: float
    dla_time_ms: float | None = None
    dla_energy_mj: float | None = None

    def __str__(self: Self) -> str:
        dla_str = (
            f"DLA: {self.dla_time_ms:.3f}ms/{self.dla_energy_mj:.3f}mJ"
            if self.dla_time_ms is not None
            else "DLA: N/A"
        )
        return (
            f"LayerCost({self.layer_idx}: GPU: {self.gpu_time_ms:.3f}ms/"
            f"{self.gpu_energy_mj:.3f}mJ, {dla_str})"
        )

    def __repr__(self: Self) -> str:
        return (
            f"LayerCost(layer_idx={self.layer_idx}, layer_name={self.layer_name!r}, "
            f"gpu_time_ms={self.gpu_time_ms}, gpu_energy_mj={self.gpu_energy_mj}, "
            f"dla_time_ms={self.dla_time_ms}, dla_energy_mj={self.dla_energy_mj})"
        )


@dataclass
class TransitionCost:
    """
    Stores the cost of transitioning between processors.

    Attributes
    ----------
    from_processor : ProcessorType
        The source processor.
    to_processor : ProcessorType
        The destination processor.
    time_ms : float
        Transition time in milliseconds.
    energy_mj : float
        Transition energy in millijoules.

    """

    from_processor: ProcessorType
    to_processor: ProcessorType
    time_ms: float
    energy_mj: float

    def __str__(self: Self) -> str:
        return (
            f"TransitionCost({self.from_processor.value} -> {self.to_processor.value}: "
            f"{self.time_ms:.3f}ms, {self.energy_mj:.3f}mJ)"
        )


@dataclass
class Schedule:
    """
    Maps each layer to a processor assignment.

    Attributes
    ----------
    assignments : dict[int, ProcessorType]
        Maps layer index to processor assignment.
    total_time_ms : float
        Estimated total execution time in milliseconds.
    total_energy_mj : float
        Estimated total energy consumption in millijoules.
    num_transitions : int
        Number of GPU<->DLA transitions.

    """

    assignments: dict[int, ProcessorType] = field(default_factory=dict)
    total_time_ms: float = 0.0
    total_energy_mj: float = 0.0
    num_transitions: int = 0

    def get_processor(self: Self, layer_idx: int) -> ProcessorType:
        """
        Get the processor assignment for a layer.

        Parameters
        ----------
        layer_idx : int
            The layer index.

        Returns
        -------
        ProcessorType
            The assigned processor.

        Raises
        ------
        KeyError
            If the layer is not in the schedule.

        """
        return self.assignments[layer_idx]

    def set_processor(self: Self, layer_idx: int, processor: ProcessorType) -> None:
        """
        Set the processor assignment for a layer.

        Parameters
        ----------
        layer_idx : int
            The layer index.
        processor : ProcessorType
            The processor to assign.

        """
        self.assignments[layer_idx] = processor

    def count_transitions(self: Self) -> int:
        """
        Count the number of processor transitions.

        Returns
        -------
        int
            The number of transitions between GPU and DLA.

        """
        if len(self.assignments) <= 1:
            return 0

        sorted_indices = sorted(self.assignments.keys())
        transitions = 0
        for i in range(len(sorted_indices) - 1):
            curr_proc = self.assignments[sorted_indices[i]]
            next_proc = self.assignments[sorted_indices[i + 1]]
            if curr_proc != next_proc:
                transitions += 1

        self.num_transitions = transitions
        return transitions

    def get_dla_layers(self: Self) -> list[int]:
        """
        Get indices of layers assigned to DLA.

        Returns
        -------
        list[int]
            Layer indices assigned to DLA.

        """
        return [idx for idx, proc in self.assignments.items() if proc == ProcessorType.DLA]

    def get_gpu_layers(self: Self) -> list[int]:
        """
        Get indices of layers assigned to GPU.

        Returns
        -------
        list[int]
            Layer indices assigned to GPU.

        """
        return [idx for idx, proc in self.assignments.items() if proc == ProcessorType.GPU]

    def __str__(self: Self) -> str:
        dla_count = len(self.get_dla_layers())
        gpu_count = len(self.get_gpu_layers())
        return (
            f"Schedule(GPU: {gpu_count} layers, DLA: {dla_count} layers, "
            f"transitions: {self.num_transitions}, "
            f"time: {self.total_time_ms:.2f}ms, energy: {self.total_energy_mj:.2f}mJ)"
        )

    def __repr__(self: Self) -> str:
        return (
            f"Schedule(assignments={self.assignments}, "
            f"total_time_ms={self.total_time_ms}, "
            f"total_energy_mj={self.total_energy_mj}, "
            f"num_transitions={self.num_transitions})"
        )


@dataclass
class AxoNNConfig:
    """
    Configuration for AxoNN optimization.

    Attributes
    ----------
    energy_target_mj : float | None
        Target energy consumption in millijoules per inference.
        If None, will be computed as a fraction of GPU-only energy.
    energy_target_ratio : float
        Ratio of GPU energy to use as target if energy_target_mj is None.
        Default is 0.8 (80% of GPU energy).
    max_transitions : int
        Maximum allowed number of GPU<->DLA transitions.
    transition_time_base_ms : float
        Base transition time overhead in milliseconds.
    transition_time_per_mb : float
        Additional transition time per megabyte of tensor data.
    transition_energy_base_mj : float
        Base transition energy overhead in millijoules.
    transition_energy_per_mb : float
        Additional transition energy per megabyte of tensor data.
    dla_core : int
        DLA core to use (0 or 1).
    profile_iterations : int
        Number of iterations for profiling.
    warmup_iterations : int
        Number of warmup iterations before profiling.

    """

    energy_target_mj: float | None = None
    energy_target_ratio: float = 0.8
    max_transitions: int = 3
    transition_time_base_ms: float = 0.1
    transition_time_per_mb: float = 0.05
    transition_energy_base_mj: float = 0.01
    transition_energy_per_mb: float = 0.005
    dla_core: int = 0
    profile_iterations: int = 1000
    warmup_iterations: int = 50

    def __str__(self: Self) -> str:
        return (
            f"AxoNNConfig(energy_target={self.energy_target_mj}mJ, "
            f"max_transitions={self.max_transitions}, dla_core={self.dla_core})"
        )
