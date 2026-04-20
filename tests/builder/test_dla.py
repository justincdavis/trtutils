# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/builder/_dla.py -- can_run_on_dla, build_dla_engine, get_check_dla."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trtutils.builder._dla import build_dla_engine, can_run_on_dla
from trtutils.builder._onnx import read_onnx
from trtutils.builder._utils import get_check_dla
from trtutils.compat._libs import trt

pytestmark = pytest.mark.jetson


@pytest.mark.parametrize(
    "use_network",
    [
        pytest.param(False, id="onnx-path"),
        pytest.param(True, id="network-with-config"),
    ],
)
def test_can_run_on_dla(onnx_path, use_network) -> None:
    """can_run_on_dla returns correct structure from both path and network inputs."""
    if use_network:
        network, _, config, _ = read_onnx(onnx_path)
        full_dla, chunks = can_run_on_dla(network, config=config)
    else:
        full_dla, chunks = can_run_on_dla(onnx_path)

    assert isinstance(full_dla, bool)
    assert isinstance(chunks, list)
    for chunk in chunks:
        assert len(chunk) == 4
        layers, start, end, on_dla = chunk
        assert isinstance(layers, list)
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(on_dla, bool)


def test_network_input_requires_config(onnx_path) -> None:
    """can_run_on_dla raises ValueError when passing a network without a config."""
    network, _, _, _ = read_onnx(onnx_path)
    with pytest.raises(ValueError, match="Config must be provided"):
        can_run_on_dla(network, config=None)


def test_get_check_dla(onnx_path) -> None:
    """get_check_dla returns a callable that accepts a layer and returns bool."""
    network, _, config, _ = read_onnx(onnx_path)
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = 0
    check_fn = get_check_dla(config)
    assert callable(check_fn)
    assert network.num_layers > 0
    assert isinstance(check_fn(network.get_layer(0)), bool)


@pytest.mark.parametrize(
    ("max_chunks", "min_layers"),
    [
        pytest.param(0, 0, id="full-dla-all-chunks"),
        pytest.param(1, 0, id="largest-chunk-only"),
        pytest.param(1, 20, id="default"),
        pytest.param(1, 10_000, id="no-dla-chunks"),
    ],
)
def test_build_dla_engine(
    onnx_path,
    output_engine_path,
    synthetic_batcher,
    max_chunks: int,
    min_layers: int,
) -> None:
    """build_dla_engine produces a valid engine across full-DLA, no-DLA, and mixed branches."""
    build_dla_engine(
        onnx_path,
        output_engine_path,
        data_batcher=synthetic_batcher,
        dla_core=0,
        max_chunks=max_chunks,
        min_layers=min_layers,
        optimization_level=1,
    )
    assert output_engine_path.exists()
    assert output_engine_path.stat().st_size > 0


def test_bad_onnx_raises(invalid_onnx_file, output_engine_path, synthetic_batcher) -> None:
    """Invalid ONNX content raises RuntimeError when passed to build_dla_engine."""
    with pytest.raises(RuntimeError):
        build_dla_engine(
            invalid_onnx_file,
            output_engine_path,
            data_batcher=synthetic_batcher,
            dla_core=0,
            optimization_level=1,
        )


def _make_mock_network(
    num_layers: int,
    *,
    constant_indices: set[int] | None = None,
    shuffle_indices: set[int] | None = None,
    tile_name_indices: set[int] | None = None,
):
    """Build a MagicMock network that reports num_layers and per-layer types/names."""
    mock_network = MagicMock()
    mock_network.num_layers = num_layers
    layers = {}
    for i in range(num_layers):
        layer = MagicMock()
        layer.name = f"layer_{i}"
        if constant_indices and i in constant_indices:
            layer.type = trt.LayerType.CONSTANT
        elif shuffle_indices and i in shuffle_indices:
            layer.type = trt.LayerType.SHUFFLE
        elif tile_name_indices and i in tile_name_indices:
            layer.type = trt.LayerType.CONVOLUTION
            layer.name = f"tile_op_{i}"
        else:
            layer.type = trt.LayerType.CONVOLUTION
        layers[i] = layer
    mock_network.get_layer.side_effect = lambda idx: layers[idx]
    return mock_network


@pytest.mark.parametrize(
    (
        "num_layers",
        "constant_idx",
        "shuffle_idx",
        "tile_idx",
        "chunks",
        "max_chunks",
        "min_layers",
        "expected_dla",
        "expected_none",
    ),
    [
        pytest.param(
            10,
            set(),
            set(),
            set(),
            [(0, 4, False), (5, 9, True)],
            1,
            0,
            range(5, 10),
            set(),
            id="mixed-gpu-dla",
        ),
        pytest.param(
            4,
            {0},
            {1},
            {2},
            [(0, 2, False), (3, 3, True)],
            1,
            0,
            range(3, 4),
            {0, 1, 2},
            id="constant-shuffle-tile-skipped",
        ),
        pytest.param(
            20,
            set(),
            set(),
            set(),
            [(0, 4, True), (5, 9, False), (10, 19, True)],
            1,
            0,
            range(10, 20),
            set(),
            id="max-chunks-picks-largest",
        ),
        pytest.param(
            10,
            set(),
            set(),
            set(),
            [(0, 6, False), (7, 9, True)],
            1,
            5,
            range(0),
            set(),
            id="min-layers-excludes-all",
        ),
        pytest.param(
            15,
            set(),
            set(),
            set(),
            [(0, 4, True), (5, 9, False), (10, 14, True)],
            0,
            0,
            None,
            set(),
            id="max-chunks-zero-all-dla",
        ),
    ],
)
@patch("trtutils.builder._dla.build_engine")
@patch("trtutils.builder._dla.can_run_on_dla")
@patch("trtutils.builder._dla.read_onnx")
def test_layer_assignment_mapping(
    mock_read_onnx,
    mock_can_run,
    mock_build,
    synthetic_batcher,
    num_layers: int,
    constant_idx: set[int],
    shuffle_idx: set[int],
    tile_idx: set[int],
    chunks: list[tuple[int, int, bool]],
    max_chunks: int,
    min_layers: int,
    expected_dla,
    expected_none: set[int],
) -> None:
    """build_dla_engine maps Constant/Shuffle/Tile to None, DLA chunks to INT8+DLA, GPU chunks to HALF+GPU."""
    network = _make_mock_network(
        num_layers,
        constant_indices=constant_idx,
        shuffle_indices=shuffle_idx,
        tile_name_indices=tile_idx,
    )
    mock_read_onnx.return_value = (network, MagicMock(), MagicMock(), MagicMock())

    chunk_payload = [
        ([MagicMock() for _ in range(end - start + 1)], start, end, on_dla)
        for start, end, on_dla in chunks
    ]
    mock_can_run.return_value = (False, chunk_payload)

    build_dla_engine(
        "fake.onnx",
        "out.engine",
        synthetic_batcher,
        dla_core=0,
        max_chunks=max_chunks,
        min_layers=min_layers,
    )

    mock_build.assert_called_once()
    call_kwargs = mock_build.call_args.kwargs
    layer_precision = call_kwargs["layer_precision"]
    layer_device = call_kwargs["layer_device"]

    dla_indices = set()
    if expected_dla is None:
        for start, end, on_dla in chunks:
            if on_dla:
                dla_indices.update(range(start, end + 1))
    else:
        dla_indices = set(expected_dla)

    for i in range(num_layers):
        if i in expected_none:
            assert layer_precision[i] == (i, None)
        elif i in dla_indices:
            assert layer_precision[i] == (i, trt.DataType.INT8)
            assert layer_device[i] == (i, trt.DeviceType.DLA)
        else:
            assert layer_precision[i] == (i, trt.DataType.HALF)
            assert layer_device[i] == (i, trt.DeviceType.GPU)
