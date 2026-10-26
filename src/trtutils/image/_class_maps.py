# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

_COCO_CATEGORY_IDS = (
    *range(1, 12),
    *range(13, 26),
    27,
    28,
    *range(31, 45),
    *range(46, 66),
    67,
    70,
    *range(72, 83),
    *range(84, 91),
)

COCO91_TO_COCO80: dict[int, int] = {cid: idx for idx, cid in enumerate(_COCO_CATEGORY_IDS)}
"""
COCO category ids (1-90, with gaps) to the contiguous 80-class COCO indices (0-79).

Pass as ``class_map`` to a detector whose model emits COCO category ids (stock RF-DETR),
for example ``RFDETR(engine, class_map=COCO91_TO_COCO80)`` makes horse 17 instead of 19.
"""
