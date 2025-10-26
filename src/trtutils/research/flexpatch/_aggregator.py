# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Patch aggregator using bin packing.

Classes
-------
PatchAggregator
    Packs patches into compact clusters using Guillotine bin packing.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self


class PatchInfo:
    """Information about a patch to be packed."""

    def __init__(
        self: Self,
        bbox: tuple[int, int, int, int],
        priority: int,
        patch_type: str,
    ) -> None:
        """
        Initialize patch information.

        Parameters
        ----------
        bbox : tuple[int, int, int, int]
            Original bbox as (x, y, w, h).
        priority : int
            Priority score (higher = more important).
        patch_type : str
            Type of patch ("tracking-failure" or "new-object").

        """
        self.bbox = bbox
        self.priority = priority
        self.patch_type = patch_type
        self.cluster_pos: tuple[int, int] | None = None


class PatchAggregator:
    """Packs patches into compact clusters using bin packing."""

    def __init__(
        self: Self,
        cluster_size: tuple[int, int] = (640, 360),
    ) -> None:
        """
        Initialize the patch aggregator.

        Parameters
        ----------
        cluster_size : tuple[int, int], optional
            Size of patch cluster as (width, height), by default (640, 360).

        """
        self.cluster_size = cluster_size

    def pack(
        self: Self,
        patches: list[tuple[tuple[int, int, int, int], int, str]],
    ) -> list[PatchInfo]:
        """
        Pack patches into a cluster using Guillotine bin packing.

        Parameters
        ----------
        patches : list[tuple[tuple[int, int, int, int], int, str]]
            List of (bbox, priority, type) tuples to pack.

        Returns
        -------
        list[PatchInfo]
            List of patches that were successfully packed with cluster positions.

        """
        if not patches:
            return []
        
        # Convert to PatchInfo objects
        patch_infos = [
            PatchInfo(bbox, priority, patch_type)
            for bbox, priority, patch_type in patches
        ]
        
        # Sort by priority (highest first)
        patch_infos.sort(key=lambda p: p.priority, reverse=True)
        
        # Initialize free rectangles with full cluster space
        free_rects = [(0, 0, self.cluster_size[0], self.cluster_size[1])]
        placed_patches = []
        
        for patch in patch_infos:
            _, _, w, h = patch.bbox
            
            # Try to fit in existing free rectangles
            placed = False
            for i, (fx, fy, fw, fh) in enumerate(free_rects):
                if fw >= w and fh >= h:
                    # Place patch at this position
                    patch.cluster_pos = (fx, fy)
                    placed_patches.append(patch)
                    placed = True
                    
                    # Split remaining space (Guillotine algorithm)
                    # Remove used rectangle
                    free_rects.pop(i)
                    
                    # Create two new rectangles from remaining space
                    # Split along shorter remaining dimension
                    remaining_width = fw - w
                    remaining_height = fh - h
                    
                    if remaining_width > 0:
                        # Right remainder
                        free_rects.append((fx + w, fy, remaining_width, h))
                    
                    if remaining_height > 0:
                        # Bottom remainder
                        free_rects.append((fx, fy + h, fw, remaining_height))
                    
                    break
            
            if not placed:
                # Patch doesn't fit, skip it
                continue
        
        return placed_patches

    def pack_with_ratio(
        self: Self,
        tf_patches: list[tuple[tuple[int, int, int, int], int, str]],
        no_patches: list[tuple[tuple[int, int, int, int], int, str]],
        tf_ratio: float = 0.75,
    ) -> list[PatchInfo]:
        """
        Pack patches with a specific ratio between tracking-failure and new-object.

        Parameters
        ----------
        tf_patches : list[tuple[tuple[int, int, int, int], int, str]]
            Tracking-failure patches.
        no_patches : list[tuple[tuple[int, int, int, int], int, str]]
            New-object patches.
        tf_ratio : float, optional
            Ratio of cluster to dedicate to tracking-failure patches,
            by default 0.75 (3:1 ratio).

        Returns
        -------
        list[PatchInfo]
            List of packed patches.

        """
        # Calculate approximate capacity for each type
        total_area = self.cluster_size[0] * self.cluster_size[1]
        tf_area = int(total_area * tf_ratio)
        no_area = total_area - tf_area
        
        # Sort patches by priority
        tf_sorted = sorted(tf_patches, key=lambda p: p[1], reverse=True)
        no_sorted = sorted(no_patches, key=lambda p: p[1], reverse=True)
        
        # Select patches to fit approximate area budget
        selected_tf = []
        accumulated_area = 0
        for patch in tf_sorted:
            _, _, w, h = patch[0]
            patch_area = w * h
            if accumulated_area + patch_area <= tf_area:
                selected_tf.append(patch)
                accumulated_area += patch_area
        
        selected_no = []
        accumulated_area = 0
        for patch in no_sorted:
            _, _, w, h = patch[0]
            patch_area = w * h
            if accumulated_area + patch_area <= no_area:
                selected_no.append(patch)
                accumulated_area += patch_area
        
        # Pack all selected patches together
        all_patches = selected_tf + selected_no
        return self.pack(all_patches)

