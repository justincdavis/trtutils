# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Data structures for HaX-CoNN optimization."""

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


class Objective(Enum):
    """
    Optimization objective for HaX-CoNN scheduling.

    Attributes
    ----------
    MAX_THROUGHPUT : str
        Minimize the sum of all DNN latencies (Eq. 10).
    MIN_MAX_LATENCY : str
        Minimize the maximum DNN latency (Eq. 11).

    """

    MAX_THROUGHPUT = "MAX_THROUGHPUT"
    MIN_MAX_LATENCY = "MIN_MAX_LATENCY"


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
    group_id : int | None
        The layer group this layer belongs to.

    """

    index: int
    name: str
    layer_type: str
    output_tensor_size: int
    can_run_on_dla: bool
    input_tensor_size: int = 0
    group_id: int | None = None

    def __str__(self: Self) -> str:
        dla_str = "DLA-compatible" if self.can_run_on_dla else "GPU-only"
        group_str = f", group={self.group_id}" if self.group_id is not None else ""
        return f"Layer({self.index}: {self.name}, {self.layer_type}, {dla_str}{group_str})"

    def __repr__(self: Self) -> str:
        return (
            f"Layer(index={self.index}, name={self.name!r}, "
            f"layer_type={self.layer_type!r}, output_tensor_size={self.output_tensor_size}, "
            f"can_run_on_dla={self.can_run_on_dla}, input_tensor_size={self.input_tensor_size}, "
            f"group_id={self.group_id})"
        )


@dataclass
class LayerGroup:
    """
    Atomic scheduling unit: a contiguous group of layers with the same DLA compatibility.

    Attributes
    ----------
    group_id : int
        Unique identifier for this group.
    dnn_id : int
        The DNN this group belongs to.
    layer_indices : list[int]
        Indices of the layers in this group.
    can_run_on_dla : bool
        Whether all layers in this group can run on DLA.
    total_output_tensor_size : int
        Aggregate output tensor size in bytes.
    total_input_tensor_size : int
        Aggregate input tensor size in bytes.

    """

    group_id: int
    dnn_id: int
    layer_indices: list[int] = field(default_factory=list)
    can_run_on_dla: bool = False
    total_output_tensor_size: int = 0
    total_input_tensor_size: int = 0

    @property
    def num_layers(self: Self) -> int:
        """Number of layers in this group."""
        return len(self.layer_indices)

    def __str__(self: Self) -> str:
        dla_str = "DLA-compatible" if self.can_run_on_dla else "GPU-only"
        return f"LayerGroup({self.group_id}: dnn={self.dnn_id}, {self.num_layers} layers, {dla_str})"

    def __repr__(self: Self) -> str:
        return (
            f"LayerGroup(group_id={self.group_id}, dnn_id={self.dnn_id}, "
            f"layer_indices={self.layer_indices}, can_run_on_dla={self.can_run_on_dla}, "
            f"total_output_tensor_size={self.total_output_tensor_size}, "
            f"total_input_tensor_size={self.total_input_tensor_size})"
        )


@dataclass
class LayerGroupCost:
    """
    Standalone execution costs for a layer group on each processor.

    Attributes
    ----------
    gpu_time_ms : float
        Execution time on GPU in milliseconds.
    gpu_energy_mj : float
        Energy consumption on GPU in millijoules.
    gpu_mem_throughput_mbps : float
        Memory throughput on GPU in MB/s (from EMC profiling).
    dla_time_ms : float | None
        Execution time on DLA in milliseconds, None if not DLA-compatible.
    dla_energy_mj : float | None
        Energy consumption on DLA in millijoules, None if not DLA-compatible.
    dla_mem_throughput_mbps : float | None
        Memory throughput on DLA in MB/s (from EMC profiling).

    """

    gpu_time_ms: float
    gpu_energy_mj: float
    gpu_mem_throughput_mbps: float = 0.0
    dla_time_ms: float | None = None
    dla_energy_mj: float | None = None
    dla_mem_throughput_mbps: float | None = None

    def __str__(self: Self) -> str:
        dla_str = (
            f"DLA: {self.dla_time_ms:.3f}ms/{self.dla_energy_mj:.3f}mJ"
            if self.dla_time_ms is not None
            else "DLA: N/A"
        )
        return (
            f"LayerGroupCost(GPU: {self.gpu_time_ms:.3f}ms/{self.gpu_energy_mj:.3f}mJ "
            f"@{self.gpu_mem_throughput_mbps:.0f}MB/s, {dla_str})"
        )

    def __repr__(self: Self) -> str:
        return (
            f"LayerGroupCost(gpu_time_ms={self.gpu_time_ms}, "
            f"gpu_energy_mj={self.gpu_energy_mj}, "
            f"gpu_mem_throughput_mbps={self.gpu_mem_throughput_mbps}, "
            f"dla_time_ms={self.dla_time_ms}, "
            f"dla_energy_mj={self.dla_energy_mj}, "
            f"dla_mem_throughput_mbps={self.dla_mem_throughput_mbps})"
        )


@dataclass
class DNNSchedule:
    """
    Per-DNN schedule: maps each group to a processor and records timing.

    Attributes
    ----------
    dnn_id : int
        The DNN identifier.
    assignments : dict[int, ProcessorType]
        Maps group_id to processor assignment.
    group_start_times : dict[int, float]
        Start time in ms for each group.
    group_end_times : dict[int, float]
        End time in ms for each group.
    total_time_ms : float
        Total DNN latency in milliseconds.

    """

    dnn_id: int
    assignments: dict[int, ProcessorType] = field(default_factory=dict)
    group_start_times: dict[int, float] = field(default_factory=dict)
    group_end_times: dict[int, float] = field(default_factory=dict)
    total_time_ms: float = 0.0

    def get_processor(self: Self, group_id: int) -> ProcessorType:
        """Get the processor assignment for a group."""
        return self.assignments[group_id]

    def set_processor(self: Self, group_id: int, processor: ProcessorType) -> None:
        """Set the processor assignment for a group."""
        self.assignments[group_id] = processor

    def get_dla_groups(self: Self) -> list[int]:
        """Get group IDs assigned to DLA."""
        return [gid for gid, proc in self.assignments.items() if proc == ProcessorType.DLA]

    def get_gpu_groups(self: Self) -> list[int]:
        """Get group IDs assigned to GPU."""
        return [gid for gid, proc in self.assignments.items() if proc == ProcessorType.GPU]

    def __str__(self: Self) -> str:
        dla_count = len(self.get_dla_groups())
        gpu_count = len(self.get_gpu_groups())
        return (
            f"DNNSchedule(dnn={self.dnn_id}, GPU: {gpu_count} groups, "
            f"DLA: {dla_count} groups, time: {self.total_time_ms:.2f}ms)"
        )

    def __repr__(self: Self) -> str:
        return (
            f"DNNSchedule(dnn_id={self.dnn_id}, assignments={self.assignments}, "
            f"total_time_ms={self.total_time_ms})"
        )


@dataclass
class MultiSchedule:
    """
    Combined schedule for all concurrent DNNs.

    Attributes
    ----------
    dnn_schedules : list[DNNSchedule]
        Per-DNN schedules.
    throughput_objective : float
        Sum of all DNN latencies (Eq. 10).
    max_latency_objective : float
        Maximum DNN latency (Eq. 11).

    """

    dnn_schedules: list[DNNSchedule] = field(default_factory=list)
    throughput_objective: float = 0.0
    max_latency_objective: float = 0.0

    @property
    def num_dnns(self: Self) -> int:
        """Number of DNNs in the schedule."""
        return len(self.dnn_schedules)

    def get_schedule(self: Self, dnn_id: int) -> DNNSchedule:
        """Get the schedule for a specific DNN."""
        for sched in self.dnn_schedules:
            if sched.dnn_id == dnn_id:
                return sched
        err_msg = f"No schedule found for DNN {dnn_id}"
        raise KeyError(err_msg)

    def __str__(self: Self) -> str:
        return (
            f"MultiSchedule({self.num_dnns} DNNs, "
            f"throughput={self.throughput_objective:.2f}ms, "
            f"max_latency={self.max_latency_objective:.2f}ms)"
        )

    def __repr__(self: Self) -> str:
        return (
            f"MultiSchedule(dnn_schedules={self.dnn_schedules}, "
            f"throughput_objective={self.throughput_objective}, "
            f"max_latency_objective={self.max_latency_objective})"
        )


@dataclass
class HaxconnConfig:
    """
    Configuration for HaX-CoNN optimization.

    Attributes
    ----------
    objective : Objective
        Optimization objective: MAX_THROUGHPUT or MIN_MAX_LATENCY.
    dla_core : int
        DLA core to use (0 or 1).
    profile_iterations : int
        Number of iterations for profiling.
    warmup_iterations : int
        Number of warmup iterations before profiling.
    transition_time_base_ms : float
        Base transition time overhead in milliseconds.
    transition_time_per_mb : float
        Additional transition time per megabyte of tensor data.
    pccs_alpha : float
        PCCS model alpha parameter (contention sensitivity).
    pccs_beta : float
        PCCS model beta parameter (nonlinearity exponent).
    max_bandwidth_mbps : float
        Maximum platform memory bandwidth in MB/s.
    contention_epsilon : float
        Convergence threshold for contention iteration.
    contention_max_iters : int
        Maximum iterations for contention convergence.
    z3_timeout_ms : int
        Z3 solver timeout in milliseconds.
    dynamic_budget_s : float
        Time budget for D-HaX-CoNN dynamic improvement in seconds.
    dynamic_max_rounds : int
        Maximum number of improvement rounds for D-HaX-CoNN.
    dynamic_improvement_threshold : float
        Minimum improvement ratio to continue D-HaX-CoNN rounds.

    """

    objective: Objective = Objective.MAX_THROUGHPUT
    dla_core: int = 0
    profile_iterations: int = 1000
    warmup_iterations: int = 50
    transition_time_base_ms: float = 0.1
    transition_time_per_mb: float = 0.05
    pccs_alpha: float = 0.5
    pccs_beta: float = 1.5
    max_bandwidth_mbps: float = 25600.0
    contention_epsilon: float = 0.01
    contention_max_iters: int = 20
    z3_timeout_ms: int = 60000
    dynamic_budget_s: float = 30.0
    dynamic_max_rounds: int = 10
    dynamic_improvement_threshold: float = 0.01

    def __str__(self: Self) -> str:
        return (
            f"HaxconnConfig(objective={self.objective.value}, "
            f"dla_core={self.dla_core}, "
            f"alpha={self.pccs_alpha}, beta={self.pccs_beta})"
        )
