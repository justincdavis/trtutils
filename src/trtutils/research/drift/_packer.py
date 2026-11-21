# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""DRIFT frame packing: adaptive grid-cell sampling and shelf bin packing (Section 3.2)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from trtutils.image.types import Bbox

_DENSE_GROUPS = ("rectangles", "blocks")
_COUNTER_DECREMENTS = ("all", "empty")
_ANNEAL_COUNTERS = ("frames", "packs")
# the reference implementation's block search: 2x2 squares, then horizontal, then vertical pairs
_BLOCK_OFFSETS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, -1), (-1, 0), (-1, -1)),
    ((0, 1), (-1, 0), (-1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, -1), (1, 0), (1, -1)),
    ((0, -1),),
    ((0, 1),),
    ((-1, 0),),
    ((1, 0),),
)


def cell_probabilities(
    counters: np.ndarray,
    frame: int,
    annealing_rate: float = 0.01,
    annealing_floor: float = 0.1,
    fps: float = 30.0,
) -> np.ndarray:
    """
    Compute the sampling probability of every grid cell (Equation 1).

    ``P = min(1, max(p_min, exp(-alpha * c) + min(1, d / min(fps, c + 1))))``

    Parameters
    ----------
    counters : np.ndarray
        The detection counter ``d`` of every cell.
    frame : int
        The annealing counter ``c``.
    annealing_rate : float, optional
        The decay rate ``alpha``. Default is 0.01.
    annealing_floor : float, optional
        The minimum probability ``p_min``. Default is 0.1.
    fps : float, optional
        The frame rate bounding the detection factor. Default is 30.0.

    Returns
    -------
    np.ndarray
        The probabilities, same shape as ``counters``.

    Raises
    ------
    ValueError
        If ``frame`` is negative.

    """
    if frame < 0:
        err_msg = f"frame must be non-negative, got {frame}"
        raise ValueError(err_msg)
    time_factor = math.exp(-annealing_rate * frame)
    detection_factor = np.minimum(1.0, np.asarray(counters, dtype=float) / min(fps, frame + 1))
    return np.minimum(1.0, np.maximum(annealing_floor, time_factor + detection_factor))


def dense_groups(mask: np.ndarray, mode: str = "rectangles") -> list[tuple[int, int, int, int]]:
    """
    Group sampled grid cells into rectangles that pack together.

    Parameters
    ----------
    mask : np.ndarray
        A 2D boolean array, True for every sampled cell.
    mode : str, optional
        "rectangles" keeps each 4-connected component whole when it fills its
        bounding rectangle and splits it into single cells otherwise (paper).
        "blocks" greedily forms 2x2, 1x2 and 2x1 blocks in row-major order
        (reference implementation). Default is "rectangles".

    Returns
    -------
    list[tuple[int, int, int, int]]
        The groups as (row, col, height, width), in cells. Every sampled cell is in
        exactly one group.

    Raises
    ------
    ValueError
        If ``mode`` is unknown or ``mask`` is not 2D.

    """
    if mode not in _DENSE_GROUPS:
        err_msg = f"Unknown dense_groups mode {mode!r}, expected one of {_DENSE_GROUPS}"
        raise ValueError(err_msg)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:  # noqa: PLR2004
        err_msg = f"mask must be 2D, got shape {mask.shape}"
        raise ValueError(err_msg)
    if mode == "blocks":
        return _block_groups(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    groups: list[tuple[int, int, int, int]] = []
    for label in range(1, count):
        col, row, width, height, area = (int(v) for v in stats[label])
        if area == width * height:
            groups.append((row, col, height, width))
            continue
        rows, cols = np.nonzero(labels == label)
        groups.extend((int(r), int(c), 1, 1) for r, c in zip(rows, cols))
    return groups


def _block_groups(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    unmatched = {(int(r), int(c)) for r, c in zip(*np.nonzero(mask))}
    groups: list[tuple[int, int, int, int]] = []
    for row, col in sorted(unmatched):
        if (row, col) not in unmatched:
            continue
        cells = [(row, col)]
        for offsets in _BLOCK_OFFSETS:
            neighbors = [(row + dr, col + dc) for dr, dc in offsets]
            if all(n in unmatched for n in neighbors):
                cells.extend(neighbors)
                break
        unmatched.difference_update(cells)
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        groups.append((min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1))
    return groups


def shelf_pack(
    groups: Sequence[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """
    Pack groups of cells toward a square with shelf bin packing (Algorithm 1).

    Groups are placed largest first, each on the first shelf of equal width that
    stays within the target side ``ceil(sqrt(total cells))``, else on a new shelf.

    Parameters
    ----------
    groups : Sequence[tuple[int, int, int, int]]
        The groups as (row, col, height, width), in cells.

    Returns
    -------
    tuple[list[tuple[int, int]], tuple[int, int]]
        The (row, col) of every group in the packed grid, in input order, and the
        packed grid's (rows, cols).

    """
    if len(groups) == 0:
        return [], (0, 0)
    target = math.ceil(math.sqrt(sum(h * w for _, _, h, w in groups)))
    order = sorted(range(len(groups)), key=lambda i: -groups[i][2] * groups[i][3])
    # each shelf is [x, width, height]
    shelves: list[list[int]] = []
    placements: list[tuple[int, int]] = [(0, 0)] * len(groups)
    for index in order:
        _, _, height, width = groups[index]
        for shelf in shelves:
            if shelf[1] == width and shelf[2] + height <= target:
                placements[index] = (shelf[2], shelf[0])
                shelf[2] += height
                break
        else:
            x = sum(shelf[1] for shelf in shelves)
            shelves.append([x, width, height])
            placements[index] = (0, x)
    return placements, (max(shelf[2] for shelf in shelves), sum(shelf[1] for shelf in shelves))


class FramePacker:
    """
    Packs the likely-relevant regions of a frame into a smaller image.

    The frame is covered by a fixed grid of square cells (cells overlap slightly so
    each is a full square). Each cell counts recent detections and is sampled with
    the probability of :func:`cell_probabilities`; sampled cells are grouped by
    :func:`dense_groups`, placed by :func:`shelf_pack`, and copied into a new image.
    Boxes detected on the packed image map back with :meth:`unpack`.
    """

    def __init__(
        self: Self,
        image_size: tuple[int, int],
        grid_divisor: int = 10,
        annealing_rate: float = 0.01,
        annealing_floor: float = 0.1,
        fps: float = 30.0,
        counter_decrement: str = "all",
        dense_groups: str = "rectangles",
        anneal_counter: str = "frames",
        seed: int | None = None,
    ) -> None:
        """
        Create a frame packer.

        Parameters
        ----------
        image_size : tuple[int, int]
            The frame size as (width, height).
        grid_divisor : int, optional
            The cell side is ``max(width, height) // grid_divisor``. Default is 10.
        annealing_rate : float, optional
            The decay rate of the sampling probability. Default is 0.01.
        annealing_floor : float, optional
            The minimum sampling probability. Default is 0.1.
        fps : float, optional
            The frame rate bounding the detection factor. Default is 30.0.
        counter_decrement : str, optional
            "all" decrements every counter after adding hits, "empty" only the
            counters of cells without hits. Default is "all".
        dense_groups : str, optional
            The grouping mode of :func:`dense_groups`. Default is "rectangles".
        anneal_counter : str, optional
            Whether the annealing counter counts every frame passed to
            :meth:`update` ("frames") or every :meth:`pack` call ("packs").
            Default is "frames".
        seed : int, optional
            Seed of the sampler. Default is None (random).

        Raises
        ------
        ValueError
            If a size or mode is invalid.

        """
        width, height = image_size
        if width < 1 or height < 1 or grid_divisor < 1:
            err_msg = f"invalid image_size {image_size} or grid_divisor {grid_divisor}"
            raise ValueError(err_msg)
        cell = max(width, height) // grid_divisor
        if cell < 1:
            err_msg = f"grid_divisor {grid_divisor} leaves cells smaller than one pixel"
            raise ValueError(err_msg)
        if counter_decrement not in _COUNTER_DECREMENTS:
            err_msg = f"Unknown counter_decrement {counter_decrement!r}"
            raise ValueError(err_msg)
        if anneal_counter not in _ANNEAL_COUNTERS:
            err_msg = f"Unknown anneal_counter {anneal_counter!r}"
            raise ValueError(err_msg)
        if dense_groups not in _DENSE_GROUPS:
            err_msg = f"Unknown dense_groups mode {dense_groups!r}"
            raise ValueError(err_msg)
        self._size = (width, height)
        self._cell = cell
        self._cols = math.ceil(width / cell)
        self._rows = math.ceil(height / cell)
        self._col_step = (width - cell) // (self._cols - 1) if self._cols > 1 else width
        self._row_step = (height - cell) // (self._rows - 1) if self._rows > 1 else height
        self._rate = annealing_rate
        self._floor = annealing_floor
        self._fps = fps
        self._decrement = counter_decrement
        self._grouping = dense_groups
        self._anneal = anneal_counter
        self._seed = seed
        self.reset()

    @property
    def image_size(self: Self) -> tuple[int, int]:
        """The frame size as (width, height)."""
        return self._size

    @property
    def cell_size(self: Self) -> int:
        """The side of a grid cell in pixels."""
        return self._cell

    @property
    def grid_shape(self: Self) -> tuple[int, int]:
        """The grid as (rows, cols)."""
        return self._rows, self._cols

    @property
    def counters(self: Self) -> np.ndarray:
        """A copy of the per-cell detection counters."""
        return self._counters.copy()

    @property
    def frames(self: Self) -> int:
        """The number of :meth:`update` calls since the last reset."""
        return self._frames

    @property
    def packs(self: Self) -> int:
        """The number of :meth:`pack` calls since the last reset."""
        return self._packs

    def cell_bounds(self: Self, row: int, col: int) -> Bbox:
        """
        Get the pixel box of a grid cell.

        Parameters
        ----------
        row : int
            The cell row.
        col : int
            The cell column.

        Returns
        -------
        Bbox
            The cell as (x1, y1, x2, y2), clipped to the frame.

        """
        x1, y1 = col * self._col_step, row * self._row_step
        return x1, y1, min(x1 + self._cell, self._size[0]), min(y1 + self._cell, self._size[1])

    def cells_of(self: Self, box: Bbox | Sequence[float]) -> tuple[slice, slice]:
        """
        Get the grid cells a box touches.

        Parameters
        ----------
        box : Bbox | Sequence[float]
            The box as (x1, y1, x2, y2) pixels.

        Returns
        -------
        tuple[slice, slice]
            The touched (rows, cols), for indexing a grid-shaped array.

        """
        x1, y1, x2, y2 = box
        r0 = min(self._rows - 1, max(0, math.floor(y1 / self._row_step)))
        r1 = max(r0 + 1, min(self._rows, math.ceil(y2 / self._row_step)))
        c0 = min(self._cols - 1, max(0, math.floor(x1 / self._col_step)))
        c1 = max(c0 + 1, min(self._cols, math.ceil(x2 / self._col_step)))
        return slice(r0, r1), slice(c0, c1)

    def probabilities(self: Self) -> np.ndarray:
        """
        Get the current sampling probability of every cell.

        Returns
        -------
        np.ndarray
            The (rows, cols) probabilities.

        """
        counter = self._frames if self._anneal == "frames" else self._packs
        return cell_probabilities(self._counters, counter, self._rate, self._floor, self._fps)

    def update(self: Self, boxes: Sequence[Bbox] | np.ndarray) -> None:
        """
        Count a frame's detections into the cells they touch.

        Parameters
        ----------
        boxes : Sequence[Bbox] | np.ndarray
            The frame's detection boxes as (x1, y1, x2, y2) pixels.

        """
        hits = np.zeros_like(self._counters)
        for box in boxes:
            rows, cols = self.cells_of(box)
            hits[rows, cols] += 1
        if self._decrement == "all":
            self._counters = np.maximum(self._counters + hits - 1, 0)
        else:
            self._counters = np.where(
                hits > 0, self._counters + hits, np.maximum(self._counters - 1, 0)
            )
        self._frames += 1

    def sample(self: Self, exclude: Bbox | None = None) -> np.ndarray:
        """
        Sample the cells to pack.

        Parameters
        ----------
        exclude : Bbox, optional
            A region whose cells are never sampled, e.g. the high-priority box.

        Returns
        -------
        np.ndarray
            The (rows, cols) boolean mask of sampled cells.

        """
        probabilities = self.probabilities()
        sampled = self._rng.random(probabilities.shape) < probabilities
        if exclude is not None:
            sampled[self.cells_of(exclude)] = False
        return sampled

    def pack(
        self: Self,
        image: np.ndarray,
        exclude: Bbox | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Sample cells and pack them into a new image.

        Each group of cells is copied as one contiguous region, so objects spanning
        cells of a group stay whole.

        Parameters
        ----------
        image : np.ndarray
            The frame.
        exclude : Bbox, optional
            A region whose cells are never packed, e.g. the high-priority box.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            The packed image and the offset table for :meth:`unpack`, or None when no
            cell was sampled.

        Raises
        ------
        ValueError
            If the image size does not match the packer.

        """
        height, width = image.shape[:2]
        if (width, height) != self._size:
            err_msg = f"image size {(width, height)} does not match the packer's {self._size}"
            raise ValueError(err_msg)
        sampled = self.sample(exclude)
        self._packs += 1
        if not sampled.any():
            return None
        groups = dense_groups(sampled, self._grouping)
        placements, (rows, cols) = shelf_pack(groups)
        g = self._cell
        packed = np.zeros((rows * g, cols * g, *image.shape[2:]), dtype=image.dtype)
        offsets = np.full((rows, cols, 2), -1, dtype=np.int64)
        for (row, col, h, w), (prow, pcol) in zip(groups, placements):
            x0, y0 = col * self._col_step, row * self._row_step
            x1 = min(x0 + (w - 1) * self._col_step + g, width)
            y1 = min(y0 + (h - 1) * self._row_step + g, height)
            packed[prow * g : prow * g + y1 - y0, pcol * g : pcol * g + x1 - x0] = image[
                y0:y1, x0:x1
            ]
            for i in range(h):
                for j in range(w):
                    offsets[prow + i, pcol + j] = (x0 + j * g, y0 + i * g)
        return packed, offsets

    def unpack(
        self: Self,
        boxes: Sequence[Bbox] | np.ndarray,
        offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map boxes detected on a packed image back to frame coordinates.

        A box moves with the packed cell holding its centre.

        Parameters
        ----------
        boxes : Sequence[Bbox] | np.ndarray
            The boxes on the packed image as (x1, y1, x2, y2) pixels.
        offsets : np.ndarray
            The offset table returned by :meth:`pack`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The (N, 4) integer boxes in frame coordinates, clipped to the frame, and an
            (N,) mask that is False for boxes centred on an empty packed slot or
            clipped to nothing.

        """
        arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if arr.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.int64), np.zeros((0,), dtype=bool)
        g = self._cell
        rows, cols = offsets.shape[:2]
        pcol = np.clip(((arr[:, 0] + arr[:, 2]) / 2.0 // g).astype(np.int64), 0, cols - 1)
        prow = np.clip(((arr[:, 1] + arr[:, 3]) / 2.0 // g).astype(np.int64), 0, rows - 1)
        origin = offsets[prow, pcol]
        out = np.empty_like(arr)
        out[:, [0, 2]] = arr[:, [0, 2]] - (pcol * g)[:, None] + origin[:, 0:1]
        out[:, [1, 3]] = arr[:, [1, 3]] - (prow * g)[:, None] + origin[:, 1:2]
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, self._size[0])
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, self._size[1])
        result = np.rint(out).astype(np.int64)
        valid = (origin[:, 0] >= 0) & (result[:, 2] > result[:, 0]) & (result[:, 3] > result[:, 1])
        return result, valid

    def reset(self: Self) -> None:
        """Clear the counters and reseed the sampler."""
        self._counters = np.zeros((self._rows, self._cols), dtype=np.int64)
        self._frames = 0
        self._packs = 0
        self._rng = np.random.default_rng(self._seed)
