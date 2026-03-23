# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for build hooks -- YOLO NMS hook and common utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from trtutils.builder.hooks._common import make_plugin_field
from trtutils.builder.hooks._yolo import (
    _get_yolo_output_tensor,
    _slice_dynamic,
    _transpose_nc_to_nlc,
    yolo_efficient_nms_hook,
)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("test_int", 42, id="int"),
        pytest.param("test_float", 0.5, id="float"),
        pytest.param("test_list", [1, 2, 3], id="list_int"),
        pytest.param("test_list_f", [1.0, 2.0], id="list_float"),
    ],
)
def test_make_plugin_field(name, value) -> None:
    """Plugin field is created with correct name for each value type."""
    field = make_plugin_field(name, value)
    assert field.name == name


@pytest.mark.cpu
def test_yolov10_passthrough() -> None:
    """YOLOv10-style (1,300,6) output passes through unmodified."""
    hook = yolo_efficient_nms_hook(num_classes=80)
    mock_network = MagicMock()
    mock_output = MagicMock()
    mock_output.shape = (1, 300, 6)
    mock_network.get_output.return_value = mock_output
    result = hook(mock_network)
    assert result is mock_network


@pytest.mark.cpu
def test_get_yolo_output_tensor_finds_output_matching_dim1() -> None:
    """Finds output tensor where dims[1] matches num_classes."""
    mock_network = MagicMock()
    mock_output = MagicMock()
    mock_output.shape = (1, 84, 8400)
    type(mock_network).num_outputs = PropertyMock(return_value=1)
    mock_network.get_output.return_value = mock_output

    result = _get_yolo_output_tensor(mock_network, 84)
    assert result is mock_output


@pytest.mark.cpu
def test_get_yolo_output_tensor_finds_output_matching_dim2() -> None:
    """Finds output tensor where dims[2] matches num_classes."""
    mock_network = MagicMock()
    mock_output = MagicMock()
    mock_output.shape = (1, 8400, 84)
    type(mock_network).num_outputs = PropertyMock(return_value=1)
    mock_network.get_output.return_value = mock_output

    result = _get_yolo_output_tensor(mock_network, 84)
    assert result is mock_output


@pytest.mark.cpu
def test_get_yolo_output_tensor_skips_non_3d_outputs() -> None:
    """Skips outputs that are not 3-dimensional."""
    mock_network = MagicMock()
    mock_output_2d = MagicMock()
    mock_output_2d.shape = (1, 84)
    mock_output_3d = MagicMock()
    mock_output_3d.shape = (1, 84, 8400)
    type(mock_network).num_outputs = PropertyMock(return_value=2)
    mock_network.get_output.side_effect = [mock_output_2d, mock_output_3d]

    result = _get_yolo_output_tensor(mock_network, 84)
    assert result is mock_output_3d


@pytest.mark.cpu
def test_get_yolo_output_tensor_raises_when_no_match() -> None:
    """Raises RuntimeError when no matching output found."""
    mock_network = MagicMock()
    mock_output = MagicMock()
    mock_output.shape = (1, 100, 200)
    type(mock_network).num_outputs = PropertyMock(return_value=1)
    mock_network.get_output.return_value = mock_output

    with pytest.raises(RuntimeError, match="Could not find YOLO-like output"):
        _get_yolo_output_tensor(mock_network, 84)


@pytest.mark.cpu
def test_get_yolo_output_tensor_raises_when_no_outputs() -> None:
    """Raises RuntimeError when network has zero outputs."""
    mock_network = MagicMock()
    type(mock_network).num_outputs = PropertyMock(return_value=0)

    with pytest.raises(RuntimeError, match="Could not find YOLO-like output"):
        _get_yolo_output_tensor(mock_network, 84)


@pytest.mark.cpu
def test_transpose_nc_to_nlc_transposes_when_dim1_matches() -> None:
    """Adds shuffle layer when dims[1] == num_classes and dims[2] != num_classes."""
    mock_network = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.shape = (1, 84, 8400)
    mock_shuffle = MagicMock()
    mock_transposed = MagicMock()
    mock_network.add_shuffle.return_value = mock_shuffle
    mock_shuffle.get_output.return_value = mock_transposed

    result = _transpose_nc_to_nlc(mock_network, mock_tensor, 84)
    assert result is mock_transposed
    mock_network.add_shuffle.assert_called_once_with(mock_tensor)
    assert mock_shuffle.first_transpose == (0, 2, 1)


@pytest.mark.cpu
def test_transpose_nc_to_nlc_passthrough_when_dim2_matches() -> None:
    """Returns tensor unchanged when dims[2] == num_classes (already NLC)."""
    mock_network = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.shape = (1, 8400, 84)

    result = _transpose_nc_to_nlc(mock_network, mock_tensor, 84)
    assert result is mock_tensor
    mock_network.add_shuffle.assert_not_called()


@pytest.mark.cpu
def test_transpose_nc_to_nlc_passthrough_when_both_dims_match() -> None:
    """Returns tensor unchanged when both dims match (dims[1] == dims[2])."""
    mock_network = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.shape = (1, 84, 84)

    result = _transpose_nc_to_nlc(mock_network, mock_tensor, 84)
    assert result is mock_tensor


@pytest.mark.cpu
def test_slice_dynamic_creates_slice_layer() -> None:
    """_slice_dynamic creates shape/constant/gather/concat/slice layers."""
    mock_network = MagicMock()
    mock_tensor = MagicMock()

    mock_shape_layer = MagicMock()
    mock_shape_tensor = MagicMock()
    mock_shape_tensor.dtype = MagicMock()
    mock_shape_layer.get_output.return_value = mock_shape_tensor
    mock_network.add_shape.return_value = mock_shape_layer

    mock_const_layer = MagicMock()
    mock_const_tensor = MagicMock()
    mock_const_layer.get_output.return_value = mock_const_tensor
    mock_network.add_constant.return_value = mock_const_layer

    mock_gather_layer = MagicMock()
    mock_gather_tensor = MagicMock()
    mock_gather_layer.get_output.return_value = mock_gather_tensor
    mock_network.add_gather.return_value = mock_gather_layer

    mock_concat_layer = MagicMock()
    mock_concat_tensor = MagicMock()
    mock_concat_layer.get_output.return_value = mock_concat_tensor
    mock_network.add_concatenation.return_value = mock_concat_layer

    mock_slice_layer = MagicMock()
    mock_slice_output = MagicMock()
    mock_slice_layer.get_output.return_value = mock_slice_output
    mock_network.add_slice.return_value = mock_slice_layer

    with patch("trtutils.builder.hooks._yolo.FLAGS") as mock_flags:
        mock_flags.TRT_HAS_INT64 = False
        result = _slice_dynamic(mock_network, mock_tensor, [0, 0, 0], 4)

    assert result is mock_slice_output
    mock_network.add_shape.assert_called_once_with(mock_tensor)
    mock_network.add_slice.assert_called_once()


@pytest.mark.cpu
def test_slice_dynamic_invalid_start_vals_length_raises() -> None:
    """_slice_dynamic raises ValueError for wrong start_vals length."""
    mock_network = MagicMock()
    mock_tensor = MagicMock()

    with pytest.raises(ValueError, match="expected start_vals to be a list of length 3"):
        _slice_dynamic(mock_network, mock_tensor, [0, 0], 4)


def _setup_yolo_network(*, output_shape, num_outputs=1):
    """Create a mock network for YOLO hook testing."""
    mock_network = MagicMock()
    mock_output = MagicMock()
    mock_output.shape = output_shape

    type(mock_network).num_outputs = PropertyMock(return_value=num_outputs)
    mock_network.get_output.return_value = mock_output

    for method_name in [
        "add_shuffle",
        "add_shape",
        "add_constant",
        "add_gather",
        "add_concatenation",
        "add_slice",
        "add_elementwise",
    ]:
        mock_layer = MagicMock()
        mock_layer.get_output.return_value = MagicMock()
        getattr(mock_network, method_name).return_value = mock_layer

    mock_plugin_layer = MagicMock()
    mock_nms_outputs = [MagicMock() for _ in range(4)]
    mock_plugin_layer.get_output.side_effect = mock_nms_outputs
    mock_network.add_plugin_v2.return_value = mock_plugin_layer

    return mock_network


def _patch_trt_for_hook():
    """Create common TRT mock patches for hook tests."""
    mock_shape_layer = MagicMock()
    mock_shape_tensor = MagicMock()
    mock_shape_tensor.dtype = MagicMock()
    mock_shape_layer.get_output.return_value = mock_shape_tensor

    mock_creator = MagicMock()
    mock_plugin = MagicMock()
    mock_creator.create_plugin.return_value = mock_plugin

    mock_registry = MagicMock()
    mock_registry.get_plugin_creator.return_value = mock_creator

    return mock_shape_layer, mock_registry


@pytest.mark.cpu
def test_full_hook_yolov8_style() -> None:
    """Full hook flow with YOLOv8-style output (N, channels, boxes)."""
    hook = yolo_efficient_nms_hook(num_classes=80)
    mock_network = _setup_yolo_network(output_shape=(1, 84, 8400))
    mock_shape_layer, mock_registry = _patch_trt_for_hook()

    with patch("trtutils.builder.hooks._yolo.trt") as mock_trt, patch(
        "trtutils.builder.hooks._yolo.FLAGS"
    ) as mock_flags:
        mock_flags.TRT_HAS_INT64 = False
        mock_trt.TensorIOMode = MagicMock()
        mock_network.add_shape.return_value = mock_shape_layer
        mock_trt.get_plugin_registry.return_value = mock_registry
        mock_trt.PluginFieldCollection = MagicMock()
        mock_trt.PluginFieldType.INT32 = 0
        mock_trt.PluginFieldType.FLOAT32 = 1
        mock_trt.PluginField = MagicMock()
        mock_trt.ElementWiseOperation.PROD = MagicMock()
        mock_trt.DataType.INT64 = MagicMock()

        result = hook(mock_network)

    assert result is mock_network
    mock_network.unmark_output.assert_called()
    assert mock_network.mark_output.call_count == 4


@pytest.mark.cpu
def test_full_hook_yolox_objectness() -> None:
    """Hook with YOLOX-style output including objectness (N, boxes, classes+5)."""
    hook = yolo_efficient_nms_hook(num_classes=80)
    mock_network = _setup_yolo_network(output_shape=(1, 8400, 85))
    mock_shape_layer, mock_registry = _patch_trt_for_hook()

    with patch("trtutils.builder.hooks._yolo.trt") as mock_trt, patch(
        "trtutils.builder.hooks._yolo.FLAGS"
    ) as mock_flags:
        mock_flags.TRT_HAS_INT64 = False
        mock_network.add_shape.return_value = mock_shape_layer
        mock_trt.get_plugin_registry.return_value = mock_registry
        mock_trt.PluginFieldCollection = MagicMock()
        mock_trt.PluginFieldType.INT32 = 0
        mock_trt.PluginFieldType.FLOAT32 = 1
        mock_trt.PluginField = MagicMock()
        mock_trt.ElementWiseOperation.PROD = MagicMock()
        mock_trt.DataType.INT64 = MagicMock()

        result = hook(mock_network)

    assert result is mock_network
    mock_network.add_elementwise.assert_called_once()
    assert mock_network.mark_output.call_count == 4


@pytest.mark.cpu
def test_plugin_not_found_raises() -> None:
    """Raises RuntimeError when EfficientNMS_TRT plugin is not available."""
    hook = yolo_efficient_nms_hook(num_classes=80)
    mock_network = _setup_yolo_network(output_shape=(1, 84, 8400))

    mock_shape_layer, mock_registry = _patch_trt_for_hook()
    mock_registry.get_plugin_creator.return_value = None

    with patch("trtutils.builder.hooks._yolo.trt") as mock_trt, patch(
        "trtutils.builder.hooks._yolo.FLAGS"
    ) as mock_flags:
        mock_flags.TRT_HAS_INT64 = False
        mock_network.add_shape.return_value = mock_shape_layer
        mock_trt.get_plugin_registry.return_value = mock_registry
        mock_trt.PluginFieldCollection = MagicMock()
        mock_trt.PluginFieldType.INT32 = 0
        mock_trt.PluginFieldType.FLOAT32 = 1
        mock_trt.PluginField = MagicMock()
        mock_trt.DataType.INT64 = MagicMock()

        with pytest.raises(RuntimeError, match="EfficientNMS_TRT plugin not found"):
            hook(mock_network)


@pytest.mark.cpu
def test_no_matching_output_raises() -> None:
    """Raises RuntimeError when no YOLO-like output tensor found."""
    hook = yolo_efficient_nms_hook(num_classes=80)
    mock_network = MagicMock()
    mock_output = MagicMock()
    mock_output.shape = (1, 100, 200)
    type(mock_network).num_outputs = PropertyMock(return_value=1)
    mock_network.get_output.return_value = mock_output

    with pytest.raises(RuntimeError, match="Could not find YOLO-like output"):
        hook(mock_network)
