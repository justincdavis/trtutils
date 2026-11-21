# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/research/drift/_packer.py -- cell sampling, grouping, shelf packing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trtutils.research.drift import FramePacker, cell_probabilities, dense_groups, shelf_pack

pytestmark = pytest.mark.cpu


def _coordinate_image(width, height):
    # every pixel holds its own (x + 1, y + 1); 0 marks pixels no cell copied
    ys, xs = np.mgrid[0:height, 0:width]
    return np.stack([xs + 1, ys + 1], axis=-1).astype(np.int32)


def _silence(packer, frames):
    # advance the annealing counter with empty frames so probabilities fall to the floor
    for _ in range(frames):
        packer.update([])


@pytest.mark.parametrize(
    ("size", "cell", "grid", "steps"),
    [
        pytest.param((1920, 1080), 192, (6, 10), (177, 192), id="mot17"),
        pytest.param((1392, 512), 139, (4, 11), (124, 125), id="kitti"),
        pytest.param((640, 480), 64, (8, 10), (59, 64), id="mot17-05"),
    ],
)
def test_grid_layout(size, cell, grid, steps) -> None:
    """Cells are full squares that cover the frame, as in the reference implementation."""
    packer = FramePacker(size)
    assert packer.cell_size == cell
    assert packer.grid_shape == grid
    rows, cols = grid
    width, height = size
    for row in range(rows):
        for col in range(cols):
            x1, y1, x2, y2 = packer.cell_bounds(row, col)
            assert (x2 - x1, y2 - y1) == (cell, cell)
            assert (x1, y1) == (col * steps[1], row * steps[0])
    assert packer.cell_bounds(rows - 1, cols - 1)[2] <= width
    assert packer.cell_bounds(rows - 1, cols - 1)[3] <= height


@pytest.mark.parametrize(
    ("counters", "frame", "expected"),
    [
        pytest.param([0, 1, 5], 0, [1.0, 1.0, 1.0], id="start-explores-everything"),
        pytest.param([0, 1, 5], 4, [math.exp(-0.04), 1.0, 1.0], id="early-frames"),
        pytest.param([0, 1, 5], 1000, [0.1, 0.1, 5 / 30 + math.exp(-10)], id="annealed-to-floor"),
        pytest.param([60], 1000, [1.0], id="busy-cell-saturates"),
    ],
)
def test_cell_probabilities(counters, frame, expected) -> None:
    """Equation 1 at hand-computed points (alpha 0.01, floor 0.1, 30 FPS)."""
    actual = cell_probabilities(np.array(counters), frame)
    np.testing.assert_allclose(actual, expected)


def test_cell_probabilities_rejects_negative_frame() -> None:
    """A negative frame counter raises ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        cell_probabilities(np.zeros(3), -1)


@pytest.mark.parametrize(
    ("mode", "after_hits", "after_empty"),
    [
        pytest.param("all", 1, 0, id="decrement-all"),
        pytest.param("empty", 2, 1, id="decrement-empty"),
    ],
)
def test_counter_update(mode, after_hits, after_empty) -> None:
    """Counters add one per touching box, then decay by the configured rule."""
    packer = FramePacker((640, 480), counter_decrement=mode)
    box = (1, 1, 10, 10)
    packer.update([box, box])
    expected = np.zeros(packer.grid_shape, dtype=int)
    expected[0, 0] = after_hits
    np.testing.assert_array_equal(packer.counters, expected)
    packer.update(np.zeros((0, 4)))
    expected[0, 0] = after_empty
    np.testing.assert_array_equal(packer.counters, expected)
    assert packer.frames == 2


def test_cells_of_maps_by_step_and_clamps() -> None:
    """A box maps to the cells its extent steps over; boxes past the last cell start still count."""
    packer = FramePacker((1920, 1080))
    assert packer.cells_of((100, 100, 200, 400)) == (slice(0, 3), slice(0, 2))
    # y 1078 is below the last cell (885..1077), so it maps to the last row
    assert packer.cells_of((10, 1078, 20, 1080)) == (slice(5, 6), slice(0, 1))


def _mask(cells, shape=(4, 4)):
    mask = np.zeros(shape, dtype=bool)
    for cell in cells:
        mask[cell] = True
    return mask


SQUARE_3X3 = [(r, c) for r in range(3) for c in range(3)]
RECT_2X3 = [(r, c) for r in range(1, 3) for c in range(1, 4)]
L_SHAPE = [(0, 0), (1, 0), (1, 1)]


@pytest.mark.parametrize(
    ("mode", "cells", "expected"),
    [
        pytest.param("rectangles", RECT_2X3, [(1, 1, 2, 3)], id="rectangles-filled"),
        pytest.param(
            "rectangles", L_SHAPE, [(0, 0, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)], id="rectangles-l"
        ),
        pytest.param("rectangles", SQUARE_3X3, [(0, 0, 3, 3)], id="rectangles-square"),
        pytest.param(
            "blocks",
            SQUARE_3X3,
            [(0, 0, 2, 2), (0, 2, 2, 1), (2, 0, 1, 2), (2, 2, 1, 1)],
            id="blocks-square",
        ),
        pytest.param("blocks", RECT_2X3, [(1, 1, 2, 2), (1, 3, 2, 1)], id="blocks-rectangle"),
        pytest.param("blocks", L_SHAPE, [(0, 0, 2, 1), (1, 1, 1, 1)], id="blocks-l"),
    ],
)
def test_dense_groups(mode, cells, expected) -> None:
    """Both grouping modes cover every sampled cell exactly once with the expected groups."""
    groups = dense_groups(_mask(cells), mode)
    assert sorted(groups) == sorted(expected)
    covered = sorted((r + i, c + j) for r, c, h, w in groups for i in range(h) for j in range(w))
    assert covered == sorted(cells)


def test_dense_groups_invalid() -> None:
    """An unknown mode or a non-2D mask raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dense_groups"):
        dense_groups(np.ones((2, 2), dtype=bool), "triangles")
    with pytest.raises(ValueError, match="2D"):
        dense_groups(np.ones((2, 2, 2), dtype=bool))


def test_shelf_pack_places_toward_square() -> None:
    """Algorithm 1: largest first, first equal-width shelf within the target side, else a new shelf."""
    groups = [(0, 0, 2, 2), (0, 2, 2, 1), (2, 0, 1, 2), (2, 2, 1, 1)]
    placements, grid = shelf_pack(groups)
    assert placements == [(0, 0), (0, 2), (2, 0), (2, 2)]
    assert grid == (3, 3)
    assert shelf_pack([]) == ([], (0, 0))


@pytest.mark.parametrize("seed", range(5))
def test_shelf_pack_never_overlaps(seed) -> None:
    """Random groupings pack inside the grid with no two groups sharing a slot."""
    mask = np.random.default_rng(seed).random((6, 10)) < 0.5
    groups = dense_groups(mask)
    placements, (rows, cols) = shelf_pack(groups)
    used = np.zeros((rows, cols), dtype=int)
    for (_, _, h, w), (pr, pc) in zip(groups, placements):
        used[pr : pr + h, pc : pc + w] += 1
    assert used.max() == 1
    assert used.sum() == mask.sum()


def test_pack_unpack_whole_frame_is_exact() -> None:
    """With every cell sampled the frame is one group, so boxes spanning cells unpack exactly."""
    image = _coordinate_image(1920, 1080)
    packer = FramePacker((1920, 1080), seed=0)
    packed, offsets = packer.pack(image)
    assert offsets.shape[:2] == packer.grid_shape
    # a box across four cells; the reference implementation shifts it by the cell overlap
    box = (150, 150, 250, 400)
    x1, y1 = packed[box[1], box[0]]
    boxes, valid = packer.unpack([box], offsets)
    assert valid.tolist() == [True]
    assert boxes[0].tolist() == [x1 - 1, y1 - 1, x1 - 1 + 100, y1 - 1 + 250]
    assert boxes[0].tolist() == list(box)


@pytest.mark.parametrize("mode", ["rectangles", "blocks"])
def test_pack_unpack_roundtrip(mode) -> None:
    """Every copied pixel of a packed image unpacks to the frame pixel it came from."""
    image = _coordinate_image(1920, 1080)
    packer = FramePacker((1920, 1080), dense_groups=mode, seed=0)
    exclude = (700, 300, 1100, 700)
    packed, offsets = packer.pack(image, exclude=exclude)
    ys, xs = np.nonzero(packed[..., 0])
    picks = np.random.default_rng(1).choice(len(xs), size=500, replace=False)
    boxes = [(int(xs[i]), int(ys[i]), int(xs[i]) + 1, int(ys[i]) + 1) for i in picks]
    unpacked, valid = packer.unpack(boxes, offsets)
    assert valid.all()
    for (px, py, _, _), (x1, y1, x2, y2) in zip(boxes, unpacked.tolist()):
        source_x, source_y = packed[py, px] - 1
        assert (x1, y1, x2, y2) == (source_x, source_y, source_x + 1, source_y + 1)
    # nothing from the excluded cells was packed
    rows, cols = packer.cells_of(exclude)
    for row in range(rows.start, rows.stop):
        for col in range(cols.start, cols.stop):
            cx1, cy1, _, _ = packer.cell_bounds(row, col)
            assert not np.any(np.all(offsets == (cx1, cy1), axis=-1))


def test_unpack_rejects_empty_slots_and_clips() -> None:
    """A box centred on an empty packed slot is invalid; others are clipped to the frame."""
    packer = FramePacker((640, 480))
    offsets = np.full((1, 2, 2), -1)
    offsets[0, 0] = (600, 450)
    boxes, valid = packer.unpack([(10, 10, 60, 50), (70, 10, 80, 20)], offsets)
    assert valid.tolist() == [True, False]
    assert boxes[0].tolist() == [610, 460, 640, 480]
    empty, empty_valid = packer.unpack([], offsets)
    assert empty.shape == (0, 4)
    assert empty_valid.shape == (0,)


def test_pack_returns_none_when_nothing_sampled() -> None:
    """No sampled cell means no packed image."""
    packer = FramePacker((640, 480), annealing_rate=1000.0, annealing_floor=0.0)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # frame 0 samples everything, so exclude the whole frame
    assert packer.pack(image, exclude=(0, 0, 640, 480)) is None
    _silence(packer, 1)
    assert packer.pack(image) is None
    assert packer.packs == 2


def test_pack_seed_and_reset() -> None:
    """The same seed gives the same packing; reset reseeds and clears the counters."""
    image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    outputs = []
    for _ in range(2):
        packer = FramePacker((640, 480), annealing_rate=1.0, seed=7)
        _silence(packer, 10)
        outputs.append(packer.pack(image))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    packer.update([(0, 0, 100, 100)])
    packer.reset()
    assert (packer.frames, packer.packs) == (0, 0)
    assert not packer.counters.any()
    _silence(packer, 10)
    np.testing.assert_array_equal(packer.pack(image)[1], outputs[0][1])


@pytest.mark.parametrize(
    ("anneal_counter", "expected"),
    [
        pytest.param("frames", math.exp(-5.0), id="frames"),
        pytest.param("packs", 1.0, id="packs"),
    ],
)
def test_anneal_counter(anneal_counter, expected) -> None:
    """The annealing counter advances with every frame or only with every pack."""
    packer = FramePacker(
        (640, 480), annealing_rate=1.0, annealing_floor=0.0, anneal_counter=anneal_counter
    )
    _silence(packer, 5)
    np.testing.assert_allclose(packer.probabilities(), expected)


def test_pack_rejects_wrong_size_and_bad_config() -> None:
    """A frame of another size, or invalid packer settings, raise ValueError."""
    packer = FramePacker((640, 480))
    with pytest.raises(ValueError, match="does not match"):
        packer.pack(np.zeros((100, 100, 3), dtype=np.uint8))
    for kwargs in (
        {"image_size": (0, 480)},
        {"image_size": (8, 8), "grid_divisor": 10},
        {"image_size": (640, 480), "counter_decrement": "some"},
        {"image_size": (640, 480), "anneal_counter": "seconds"},
        {"image_size": (640, 480), "dense_groups": "triangles"},
    ):
        with pytest.raises(ValueError):  # noqa: PT011
            FramePacker(**kwargs)
