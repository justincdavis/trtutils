# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Implementation of the FlexPatch paper for efficient object detection.

FlexPatch enables real-time object detection on high-resolution video
by combining tracking and selective patch-based detection.

Paper: https://juheonyi.github.io/files/FlexPatch.pdf

Classes
-------
ObjectTracker
    Optical flow-based object tracker.
TrackingFailureRecommender
    Recommends patches for tracking failure detection.
NewObjectRecommender
    Recommends patches for new object detection.
PatchAggregator
    Packs patches into compact clusters using bin packing.
FlexPatch
    Main FlexPatch system integrating all components.

"""

from __future__ import annotations

from ._aggregator import PatchAggregator
from ._flexpatch import FlexPatch
from ._no_recommender import NewObjectRecommender
from ._tf_recommender import TrackingFailureRecommender
from ._tracker import ObjectTracker

__all__ = [
    "FlexPatch",
    "NewObjectRecommender",
    "ObjectTracker",
    "PatchAggregator",
    "TrackingFailureRecommender",
]

