# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for DLA analysis -- can_run_on_dla(), build_dla_engine(), get_check_dla()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trtutils.builder._batcher import SyntheticBatcher
from trtutils.builder._dla import build_dla_engine, can_run_on_dla
from trtutils.builder._onnx import read_onnx
from trtutils.builder._utils import get_check_dla
from trtutils.compat._libs import trt


def _make_batcher():
    return SyntheticBatcher(
        shape=(3, 8, 8),
        dtype=np.dtype(np.float32),
        batch_size=1,
        num_batches=2,
        order="NCHW",
    )


def _make_mock_network(
    num_layers, *, constant_indices=None, shuffle_indices=None, tile_name_indices=None
):
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


def test_returns_tuple(onnx_path) -> None:
    """can_run_on_dla returns (bool, list) tuple."""
    result = can_run_on_dla(onnx_path)
    assert isinstance(result, tuple)
    assert len(result) == 2
    full_dla, chunks = result
    assert isinstance(full_dla, bool)
    assert isinstance(chunks, list)


def test_chunks_have_correct_structure(onnx_path) -> None:
    """Each chunk is (layers, start, end, on_dla) tuple."""
    _, chunks = can_run_on_dla(onnx_path)
    for chunk in chunks:
        assert len(chunk) == 4
        layers, start, end, on_dla = chunk
        assert isinstance(layers, list)
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(on_dla, bool)


def test_network_input_requires_config(onnx_path) -> None:
    """ValueError when passing network without config."""
    network, _, _, _ = read_onnx(onnx_path)
    with pytest.raises(ValueError, match="Config must be provided"):
        can_run_on_dla(network, config=None)


def test_network_input_with_config(onnx_path) -> None:
    """can_run_on_dla accepts a pre-made network with config."""
    network, _, config, _ = read_onnx(onnx_path)
    full_dla, chunks = can_run_on_dla(network, config=config)
    assert isinstance(full_dla, bool)
    assert isinstance(chunks, list)


def test_build_basic(onnx_path, output_engine_path) -> None:
    """build_dla_engine runs without error."""
    batcher = _make_batcher()
    build_dla_engine(
        onnx_path,
        output_engine_path,
        data_batcher=batcher,
        dla_core=0,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_full_dla_path(onnx_path, output_engine_path) -> None:
    """When full_dla is True, build_engine is called with DLA default_device."""
    batcher = _make_batcher()
    with patch("trtutils.builder._dla.can_run_on_dla") as mock_check:
        mock_check.return_value = (True, [])
        with patch("trtutils.builder._dla.build_engine") as mock_build:
            build_dla_engine(
                onnx_path,
                output_engine_path,
                data_batcher=batcher,
                dla_core=0,
            )
            mock_build.assert_called_once()
            call_kwargs = mock_build.call_args

            assert call_kwargs.kwargs.get("fp16") is True
            assert call_kwargs.kwargs.get("int8") is True


def test_no_dla_chunks(onnx_path, output_engine_path) -> None:
    """No DLA-compatible layers → GPU-only build with warning."""
    batcher = _make_batcher()
    mock_layers = [MagicMock() for _ in range(5)]
    chunks = [(mock_layers, 0, 4, False)]
    with patch("trtutils.builder._dla.can_run_on_dla") as mock_check:
        mock_check.return_value = (False, chunks)
        with patch("trtutils.builder._dla.build_engine") as mock_build:
            build_dla_engine(
                onnx_path,
                output_engine_path,
                data_batcher=batcher,
                dla_core=0,
            )
            mock_build.assert_called_once()


def test_mixed_dla_gpu(onnx_path, output_engine_path) -> None:
    """Partial DLA → layer assignments with mixed devices."""
    batcher = _make_batcher()
    build_dla_engine(
        onnx_path,
        output_engine_path,
        data_batcher=batcher,
        dla_core=0,
        max_chunks=1,
        min_layers=0,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_max_chunks_limit(onnx_path, output_engine_path) -> None:
    """max_chunks parameter limits DLA chunk assignment."""
    batcher = _make_batcher()
    build_dla_engine(
        onnx_path,
        output_engine_path,
        data_batcher=batcher,
        dla_core=0,
        max_chunks=0,
        min_layers=0,
        optimization_level=1,
    )
    assert output_engine_path.exists()


def test_min_layers_filter(onnx_path, output_engine_path) -> None:
    """min_layers parameter filters small chunks."""
    batcher = _make_batcher()
    build_dla_engine(
        onnx_path,
        output_engine_path,
        data_batcher=batcher,
        dla_core=0,
        min_layers=99999,
        optimization_level=1,
    )
    assert output_engine_path.exists()


@patch("trtutils.builder._dla.build_engine")
@patch("trtutils.builder._dla.can_run_on_dla")
@patch("trtutils.builder._dla.read_onnx")
def test_mixed_layer_precision_assignment(
    mock_read_onnx,
    mock_can_run,
    mock_build,
) -> None:
    """Mixed path assigns FP16 to GPU layers, INT8 to DLA layers."""
    mock_network = _make_mock_network(10)
    mock_config = MagicMock()
    mock_read_onnx.return_value = (mock_network, MagicMock(), mock_config, MagicMock())

    gpu_layers = [MagicMock() for _ in range(5)]
    dla_layers = [MagicMock() for _ in range(5)]
    chunks = [
        (gpu_layers, 0, 4, False),
        (dla_layers, 5, 9, True),
    ]
    mock_can_run.return_value = (False, chunks)

    batcher = _make_batcher()
    build_dla_engine("fake.onnx", "out.engine", batcher, dla_core=0, min_layers=0)

    mock_build.assert_called_once()
    call_kwargs = mock_build.call_args.kwargs
    layer_precision = call_kwargs["layer_precision"]
    layer_device = call_kwargs["layer_device"]

    for i in range(5):
        assert layer_precision[i] == (i, trt.DataType.HALF)
        assert layer_device[i] == (i, trt.DeviceType.GPU)

    for i in range(5, 10):
        assert layer_precision[i] == (i, trt.DataType.INT8)
        assert layer_device[i] == (i, trt.DeviceType.DLA)


@patch("trtutils.builder._dla.build_engine")
@patch("trtutils.builder._dla.can_run_on_dla")
@patch("trtutils.builder._dla.read_onnx")
def test_constant_shuffle_tile_skip_precision(
    mock_read_onnx,
    mock_can_run,
    mock_build,
) -> None:
    """Constant, Shuffle, and Tile layers get None precision (skipped)."""
    mock_network = _make_mock_network(
        4,
        constant_indices={0},
        shuffle_indices={1},
        tile_name_indices={2},
    )
    mock_config = MagicMock()
    mock_read_onnx.return_value = (mock_network, MagicMock(), mock_config, MagicMock())

    gpu_layers = [MagicMock() for _ in range(3)]
    dla_layers = [MagicMock()]
    chunks = [
        (gpu_layers, 0, 2, False),
        (dla_layers, 3, 3, True),
    ]
    mock_can_run.return_value = (False, chunks)

    batcher = _make_batcher()
    build_dla_engine(
        "fake.onnx",
        "out.engine",
        batcher,
        dla_core=0,
        min_layers=0,
        max_chunks=0,
    )

    call_kwargs = mock_build.call_args.kwargs
    layer_precision = call_kwargs["layer_precision"]

    assert layer_precision[0] == (0, None)
    assert layer_precision[1] == (1, None)
    assert layer_precision[2] == (2, None)
    assert layer_precision[3] == (3, trt.DataType.INT8)


@patch("trtutils.builder._dla.build_engine")
@patch("trtutils.builder._dla.can_run_on_dla")
@patch("trtutils.builder._dla.read_onnx")
def test_max_chunks_limits_dla_assignment(
    mock_read_onnx,
    mock_can_run,
    mock_build,
) -> None:
    """max_chunks=1 only assigns the largest DLA chunk."""
    mock_network = _make_mock_network(20)
    mock_config = MagicMock()
    mock_read_onnx.return_value = (mock_network, MagicMock(), mock_config, MagicMock())

    dla_small = [MagicMock() for _ in range(5)]
    gpu_mid = [MagicMock() for _ in range(5)]
    dla_large = [MagicMock() for _ in range(10)]
    chunks = [
        (dla_small, 0, 4, True),
        (gpu_mid, 5, 9, False),
        (dla_large, 10, 19, True),
    ]
    mock_can_run.return_value = (False, chunks)

    batcher = _make_batcher()
    build_dla_engine(
        "fake.onnx",
        "out.engine",
        batcher,
        dla_core=0,
        max_chunks=1,
        min_layers=0,
    )

    call_kwargs = mock_build.call_args.kwargs
    layer_device = call_kwargs["layer_device"]

    for i in range(5):
        assert layer_device[i] == (i, trt.DeviceType.GPU), f"Layer {i} should be GPU"
    for i in range(5, 10):
        assert layer_device[i] == (i, trt.DeviceType.GPU), f"Layer {i} should be GPU"
    for i in range(10, 20):
        assert layer_device[i] == (i, trt.DeviceType.DLA), f"Layer {i} should be DLA"


@patch("trtutils.builder._dla.build_engine")
@patch("trtutils.builder._dla.can_run_on_dla")
@patch("trtutils.builder._dla.read_onnx")
def test_min_layers_filters_small_chunks(
    mock_read_onnx,
    mock_can_run,
    mock_build,
) -> None:
    """min_layers filters out DLA chunks smaller than the threshold."""
    mock_network = _make_mock_network(10)
    mock_config = MagicMock()
    mock_read_onnx.return_value = (mock_network, MagicMock(), mock_config, MagicMock())

    gpu_layers = [MagicMock() for _ in range(7)]
    dla_small = [MagicMock() for _ in range(3)]
    chunks = [
        (gpu_layers, 0, 6, False),
        (dla_small, 7, 9, True),
    ]
    mock_can_run.return_value = (False, chunks)

    batcher = _make_batcher()
    build_dla_engine(
        "fake.onnx",
        "out.engine",
        batcher,
        dla_core=0,
        min_layers=5,
    )

    call_kwargs = mock_build.call_args.kwargs
    layer_device = call_kwargs["layer_device"]

    for i in range(10):
        assert layer_device[i] == (i, trt.DeviceType.GPU), f"Layer {i} should be GPU"


@patch("trtutils.builder._dla.build_engine")
@patch("trtutils.builder._dla.can_run_on_dla")
@patch("trtutils.builder._dla.read_onnx")
def test_max_chunks_zero_assigns_all(
    mock_read_onnx,
    mock_can_run,
    mock_build,
) -> None:
    """max_chunks=0 assigns ALL qualifying DLA chunks."""
    mock_network = _make_mock_network(15)
    mock_config = MagicMock()
    mock_read_onnx.return_value = (mock_network, MagicMock(), mock_config, MagicMock())

    dla1 = [MagicMock() for _ in range(5)]
    gpu = [MagicMock() for _ in range(5)]
    dla2 = [MagicMock() for _ in range(5)]
    chunks = [
        (dla1, 0, 4, True),
        (gpu, 5, 9, False),
        (dla2, 10, 14, True),
    ]
    mock_can_run.return_value = (False, chunks)

    batcher = _make_batcher()
    build_dla_engine(
        "fake.onnx",
        "out.engine",
        batcher,
        dla_core=0,
        max_chunks=0,
        min_layers=0,
    )

    call_kwargs = mock_build.call_args.kwargs
    layer_device = call_kwargs["layer_device"]

    for i in range(5):
        assert layer_device[i] == (i, trt.DeviceType.DLA), f"Layer {i} should be DLA"
    for i in range(5, 10):
        assert layer_device[i] == (i, trt.DeviceType.GPU), f"Layer {i} should be GPU"
    for i in range(10, 15):
        assert layer_device[i] == (i, trt.DeviceType.DLA), f"Layer {i} should be DLA"


def test_get_check_dla_returns_callable(onnx_path) -> None:
    """get_check_dla returns a callable."""
    _, _, config, _ = read_onnx(onnx_path)
    check_fn = get_check_dla(config)
    assert callable(check_fn)


def test_get_check_dla_function_accepts_layer(onnx_path) -> None:
    """Returned function can be called with a layer."""
    network, _, config, _ = read_onnx(onnx_path)
    check_fn = get_check_dla(config)
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = 0
    if network.num_layers > 0:
        layer = network.get_layer(0)
        result = check_fn(layer)
        assert isinstance(result, bool)
