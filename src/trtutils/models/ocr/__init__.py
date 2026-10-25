# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""OCR model implementations."""

from __future__ import annotations

from ._ppocr import PPOCR, PPOCR_KEYS_V1, PPOCRDetector, PPOCRRecognizer

__all__ = [
    "PPOCR",
    "PPOCR_KEYS_V1",
    "PPOCRDetector",
    "PPOCRRecognizer",
]
