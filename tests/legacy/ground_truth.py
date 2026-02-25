# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageGroundTruth:
    """Ground-truth metadata for a test image."""

    image_key: str
    min_detections: int
    expected_detections: int
    max_detections: int
    required_class_ids: list[int] = field(default_factory=list)


# COCO class IDs
HORSE_CLASS_ID = 17
PERSON_CLASS_ID = 0

HORSE_GT = ImageGroundTruth("horse", 1, 1, 2, [HORSE_CLASS_ID])
PEOPLE_GT = ImageGroundTruth("people", 3, 4, 5, [PERSON_CLASS_ID])
