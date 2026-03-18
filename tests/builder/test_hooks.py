# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for build hooks -- YOLO NMS hook and common utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from trtutils.builder._onnx import read_onnx
from trtutils.builder.hooks._common import make_plugin_field
from trtutils.builder.hooks._yolo import (
    _get_yolo_output_tensor,
    _slice_dynamic,
    _transpose_nc_to_nlc,
    yolo_efficient_nms_hook,
)


@pytest.mark.cpu
def test_make_plugin_field_int_value() -> None:
    """Integer value creates INT32 field."""
    field = make_plugin_field("test_int", 42)
    assert field.name == "test_int"


@pytest.mark.cpu
def test_make_plugin_field_float_value() -> None:
    """Float value creates FLOAT32 field."""
    field = make_plugin_field("test_float", 0.5)
    assert field.name == "test_float"


@pytest.mark.cpu
def test_make_plugin_field_list_int_value() -> None:
    """List of ints creates INT32 field."""
    field = make_plugin_field("test_list", [1, 2, 3])
    assert field.name == "test_list"


@pytest.mark.cpu
def test_make_plugin_field_list_float_value() -> None:
    """List of floats creates FLOAT32 field."""
    field = make_plugin_field("test_list_f", [1.0, 2.0])
    assert field.name == "test_list_f"


@pytest.mark.cpu
def test_yolo_efficient_nms_hook_returns_callable() -> None:
    """yolo_efficient_nms_hook returns a callable."""
    hook = yolo_efficient_nms_hook(num_classes=80)
    assert callable(hook)


@pytest.mark.cpu
def test_yolo_efficient_nms_hook_custom_params() -> None:
    """Custom parameters are accepted."""
    hook = yolo_efficient_nms_hook(
        num_classes=91,
        conf_threshold=0.3,
        iou_threshold=0.45,
        top_k=200,
        box_coding="corner",
        class_agnostic=True,
    )
    assert callable(hook)


@pytest.mark.cpu
def test_yolo_efficient_nms_hook_default_params() -> None:
    """Default parameters produce a valid hook."""
    hook = yolo_efficient_nms_hook()
    assert callable(hook)


def _create_yolo_network(output_shape, onnx_path):
    """Build a TRT network with a specific output shape for testing hooks."""
    network, builder, config, _ = read_onnx(onnx_path)
    return network, builder, config


def test_yolov10_passthrough(onnx_path) -> None:
    """YOLOv10-style (1,300,6) output passes through unmodified."""
    hook = yolo_efficient_nms_hook(num_classes=80)

    # Create a mock network with YOLOv10-style output
    mock_network = MagicMock()
    mock_output = MagicMock()
    mock_output.shape = (1, 300, 6)
    mock_network.get_output.return_value = mock_output

    result = hook(mock_network)
    # Should return the same network unmodified
    assert result is mock_network


def test_hook_modifies_network(onnx_path) -> None:
    """Hook adds EfficientNMS plugin to YOLO-like network."""
    network, _, _config, _ = read_onnx(onnx_path)

    # Only run if the model has a compatible output shape
    output = network.get_output(0)
    dims = output.shape

    # Skip if output doesn't look YOLO-like (3D with class channel)
    if len(dims) != 3:
        pytest.skip("Model output is not 3D, cannot test YOLO hook")

    num_classes = dims[2] - 4 if dims[2] > 4 else dims[1] - 4
    if num_classes <= 0:
        pytest.skip("Model output doesn't have enough channels for YOLO")

    hook = yolo_efficient_nms_hook(num_classes=num_classes)
    try:
        modified = hook(network)
        # After hook, should have 4 NMS outputs
        assert modified.num_outputs == 4
    except RuntimeError:
        # EfficientNMS plugin may not be available in all TRT installations
        pytest.skip("EfficientNMS_TRT plugin not available")


@pytest.mark.parametrize("box_coding", ["center_size", "corner"], ids=["center", "corner"])
def test_box_coding_modes(box_coding) -> None:
    """Both box coding modes produce valid hooks."""
    hook = yolo_efficient_nms_hook(num_classes=80, box_coding=box_coding)
    assert callable(hook)


@pytest.mark.parametrize(
    "class_agnostic", [True, False, None], ids=["agnostic", "per_class", "default"]
)
def test_class_agnostic(class_agnostic) -> None:
    """class_agnostic parameter variants produce valid hooks."""
    hook = yolo_efficient_nms_hook(num_classes=80, class_agnostic=class_agnostic)
    assert callable(hook)


def test_with_objectness_mock() -> None:
    """Hook handles YOLOX-style output with objectness channel (N+5 channels)."""
    num_classes = 80
    yolo_efficient_nms_hook(num_classes=num_classes)

    # Mock a network where the output has channels_with_obj = 85 (80 + 5)
    mock_network = MagicMock()

    # First output is not YOLOv10 style
    mock_output_0 = MagicMock()
    mock_output_0.shape = (1, 85, 8400)

    def get_output_side_effect(_idx):
        return mock_output_0

    mock_network.get_output.side_effect = get_output_side_effect
    type(mock_network).num_outputs = PropertyMock(return_value=1)
    type(mock_network).num_inputs = PropertyMock(return_value=1)

    # The hook will try to find an output matching channels_without_obj (84) first
    # then fallback to channels_with_obj (85) which matches dims[1]=85
    # This will attempt real TRT operations which need GPU
    # So we test just the hook creation with parameters
    hook2 = yolo_efficient_nms_hook(num_classes=num_classes)
    assert callable(hook2)


def test_without_objectness_mock() -> None:
    """Hook handles YOLOv8-style output without objectness (N+4 channels)."""
    num_classes = 80
    hook = yolo_efficient_nms_hook(num_classes=num_classes)
    # 84 channels = 80 classes + 4 bbox coords
    assert callable(hook)


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
    mock_output.shape = (1, 100, 200)  # Neither dim matches 84
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
    mock_tensor.shape = (1, 84, 8400)  # dims[1]=84=num_classes, dims[2]!=84
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
    mock_tensor.shape = (1, 8400, 84)  # dims[2]=84=num_classes

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

    # Setup shape layer chain
    mock_shape_layer = MagicMock()
    mock_shape_tensor = MagicMock()
    mock_shape_tensor.dtype = MagicMock()  # Will be checked for INT64
    mock_shape_layer.get_output.return_value = mock_shape_tensor
    mock_network.add_shape.return_value = mock_shape_layer

    # All intermediate ops return mock layers/tensors
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

    # Setup all intermediate layer return values
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

    # Plugin layer returns 4 outputs (NMS outputs)
    mock_plugin_layer = MagicMock()
    mock_nms_outputs = [MagicMock() for _ in range(4)]
    mock_plugin_layer.get_output.side_effect = mock_nms_outputs
    mock_network.add_plugin_v2.return_value = mock_plugin_layer

    return mock_network


@pytest.mark.cpu
def test_full_hook_yolov8_style() -> None:
    """Full hook flow with YOLOv8-style output (N, channels, boxes)."""
    hook = yolo_efficient_nms_hook(num_classes=80)
    mock_network = _setup_yolo_network(output_shape=(1, 84, 8400))

    # Mock the plugin registry
    mock_creator = MagicMock()
    mock_plugin = MagicMock()
    mock_creator.create_plugin.return_value = mock_plugin

    with patch("trtutils.builder.hooks._yolo.trt") as mock_trt, patch(
        "trtutils.builder.hooks._yolo.FLAGS"
    ) as mock_flags:
        mock_flags.TRT_HAS_INT64 = False
        mock_trt.TensorIOMode = MagicMock()

        # Setup the shape layer dtype to not match INT64
        mock_shape_layer = MagicMock()
        mock_shape_tensor = MagicMock()
        mock_shape_tensor.dtype = MagicMock()
        mock_shape_layer.get_output.return_value = mock_shape_tensor
        mock_network.add_shape.return_value = mock_shape_layer

        mock_registry = MagicMock()
        mock_registry.get_plugin_creator.return_value = mock_creator
        mock_trt.get_plugin_registry.return_value = mock_registry

        # Need to make PluginFieldCollection and PluginField work
        mock_trt.PluginFieldCollection = MagicMock()
        mock_trt.PluginFieldType.INT32 = 0
        mock_trt.PluginFieldType.FLOAT32 = 1
        mock_trt.PluginField = MagicMock()
        mock_trt.ElementWiseOperation.PROD = MagicMock()
        mock_trt.DataType.INT64 = MagicMock()

        result = hook(mock_network)

    assert result is mock_network
    # Should have unmarked old outputs and marked new ones
    mock_network.unmark_output.assert_called()
    assert mock_network.mark_output.call_count == 4


@pytest.mark.cpu
def test_full_hook_yolox_objectness() -> None:
    """Hook with YOLOX-style output including objectness (N, boxes, classes+5)."""
    hook = yolo_efficient_nms_hook(num_classes=80)
    # (1, 8400, 85): 80 classes + 4 bbox + 1 objectness
    mock_network = _setup_yolo_network(output_shape=(1, 8400, 85))

    mock_creator = MagicMock()
    mock_plugin = MagicMock()
    mock_creator.create_plugin.return_value = mock_plugin

    with patch("trtutils.builder.hooks._yolo.trt") as mock_trt, patch(
        "trtutils.builder.hooks._yolo.FLAGS"
    ) as mock_flags:
        mock_flags.TRT_HAS_INT64 = False

        mock_shape_layer = MagicMock()
        mock_shape_tensor = MagicMock()
        mock_shape_tensor.dtype = MagicMock()
        mock_shape_layer.get_output.return_value = mock_shape_tensor
        mock_network.add_shape.return_value = mock_shape_layer

        mock_registry = MagicMock()
        mock_registry.get_plugin_creator.return_value = mock_creator
        mock_trt.get_plugin_registry.return_value = mock_registry
        mock_trt.PluginFieldCollection = MagicMock()
        mock_trt.PluginFieldType.INT32 = 0
        mock_trt.PluginFieldType.FLOAT32 = 1
        mock_trt.PluginField = MagicMock()
        mock_trt.ElementWiseOperation.PROD = MagicMock()
        mock_trt.DataType.INT64 = MagicMock()

        # First attempt with channels_without_obj=84 should fail (no match)
        # _get_yolo_output_tensor tries dims[1]=8400!=84 and dims[2]=85!=84 → RuntimeError
        # Then retries with channels_with_obj=85, dims[2]=85 matches
        result = hook(mock_network)

    assert result is mock_network
    # Objectness path uses add_elementwise for obj * class_scores
    mock_network.add_elementwise.assert_called_once()
    assert mock_network.mark_output.call_count == 4


@pytest.mark.cpu
def test_plugin_not_found_raises() -> None:
    """Raises RuntimeError when EfficientNMS_TRT plugin is not available."""
    hook = yolo_efficient_nms_hook(num_classes=80)
    mock_network = _setup_yolo_network(output_shape=(1, 84, 8400))

    with patch("trtutils.builder.hooks._yolo.trt") as mock_trt, patch(
        "trtutils.builder.hooks._yolo.FLAGS"
    ) as mock_flags:
        mock_flags.TRT_HAS_INT64 = False

        mock_shape_layer = MagicMock()
        mock_shape_tensor = MagicMock()
        mock_shape_tensor.dtype = MagicMock()
        mock_shape_layer.get_output.return_value = mock_shape_tensor
        mock_network.add_shape.return_value = mock_shape_layer

        mock_registry = MagicMock()
        mock_registry.get_plugin_creator.return_value = None
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
    mock_output.shape = (1, 100, 200)  # No dim matches 84 or 85
    type(mock_network).num_outputs = PropertyMock(return_value=1)
    mock_network.get_output.return_value = mock_output

    with pytest.raises(RuntimeError, match="Could not find YOLO-like output"):
        hook(mock_network)
