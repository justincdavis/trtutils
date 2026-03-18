# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Data structures for AxoNN optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self


class Processor(Enum):
    """Available processor types."""

    GPU = "GPU"
    DLA = "DLA"


@dataclass
class TransitionCosts:
    """Cost parameters for GPU<->DLA transitions."""

    time_base_ms: float = 0.1
    time_per_mb: float = 0.05
    energy_base_mj: float = 0.01
    energy_per_mb: float = 0.005


@dataclass
class LayerCost:
    """Execution costs for a layer on different processors."""

    gpu_time_ms: float
    gpu_energy_mj: float
    dla_time_ms: float | None = None
    dla_energy_mj: float | None = None


@dataclass
class Schedule:
    """Maps each layer to a processor assignment."""

    assignments: dict[int, Processor] = field(default_factory=dict)
    total_time_ms: float = 0.0
    total_energy_mj: float = 0.0

    @property
    def num_transitions(self: Self) -> int:
        """Count the number of processor transitions."""
        if len(self.assignments) <= 1:
            return 0
        sorted_indices = sorted(self.assignments.keys())
        transitions = 0
        for i in range(len(sorted_indices) - 1):
            if self.assignments[sorted_indices[i]] != self.assignments[sorted_indices[i + 1]]:
                transitions += 1
        return transitions
