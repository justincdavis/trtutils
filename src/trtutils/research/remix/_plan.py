# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Data structures for partition plans and blocks.

Classes
-------
PartitionBlock
    Represents an image region with metadata.
PartitionPlan
    Contains blocks, detector assignments, and performance estimates.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass
class PartitionBlock:
    """Represents an image region for partitioned detection."""

    x1: int
    y1: int
    x2: int
    y2: int
    detector_name: str | None = None
    skip_window: int = 0

    @property
    def width(self: Self) -> int:
        """Get block width."""
        return self.x2 - self.x1

    @property
    def height(self: Self) -> int:
        """Get block height."""
        return self.y2 - self.y1

    @property
    def area(self: Self) -> int:
        """Get block area."""
        return self.width * self.height

    @property
    def coords(self: Self) -> tuple[int, int, int, int]:
        """Get block coordinates as tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self: Self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls: type[Self], data: dict) -> Self:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PartitionPlan:
    """Partition plan with blocks and detector assignments."""

    blocks: list[PartitionBlock]
    est_ap: float
    est_lat: float
    plan_id: int = 0

    @property
    def num_blocks(self: Self) -> int:
        """Get number of blocks in plan."""
        return len(self.blocks)

    @property
    def total_area(self: Self) -> int:
        """Get total area covered by all blocks."""
        return sum(block.area for block in self.blocks)

    def get_detector_names(self: Self) -> list[str]:
        """Get list of unique detector names used in plan."""
        names = set()
        for block in self.blocks:
            if block.detector_name is not None:
                names.add(block.detector_name)
        return sorted(names)

    def to_dict(self: Self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "est_ap": self.est_ap,
            "est_lat": self.est_lat,
            "plan_id": self.plan_id,
        }

    @classmethod
    def from_dict(cls: type[Self], data: dict) -> Self:
        """Create from dictionary."""
        blocks = [PartitionBlock.from_dict(b) for b in data["blocks"]]
        return cls(
            blocks=blocks,
            est_ap=data["est_ap"],
            est_lat=data["est_lat"],
            plan_id=data.get("plan_id", 0),
        )

