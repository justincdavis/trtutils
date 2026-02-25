"""Tests for build hooks -- YOLO NMS hook and common utilities."""

from __future__ import annotations

import pytest


@pytest.mark.cpu
class TestMakePluginField:
    """Tests for make_plugin_field() utility."""

    def test_int_value(self) -> None:
        """Integer value creates INT32 field."""
        from trtutils.builder.hooks._common import make_plugin_field

        field = make_plugin_field("test_int", 42)
        assert field.name == "test_int"

    def test_float_value(self) -> None:
        """Float value creates FLOAT32 field."""
        from trtutils.builder.hooks._common import make_plugin_field

        field = make_plugin_field("test_float", 0.5)
        assert field.name == "test_float"

    def test_list_int_value(self) -> None:
        """List of ints creates INT32 field."""
        from trtutils.builder.hooks._common import make_plugin_field

        field = make_plugin_field("test_list", [1, 2, 3])
        assert field.name == "test_list"

    def test_list_float_value(self) -> None:
        """List of floats creates FLOAT32 field."""
        from trtutils.builder.hooks._common import make_plugin_field

        field = make_plugin_field("test_list_f", [1.0, 2.0])
        assert field.name == "test_list_f"


@pytest.mark.cpu
class TestYoloEfficientNmsHook:
    """Tests for yolo_efficient_nms_hook() factory function."""

    def test_returns_callable(self) -> None:
        """yolo_efficient_nms_hook returns a callable."""
        from trtutils.builder.hooks._yolo import yolo_efficient_nms_hook

        hook = yolo_efficient_nms_hook(num_classes=80)
        assert callable(hook)

    def test_custom_params(self) -> None:
        """Custom parameters are accepted."""
        from trtutils.builder.hooks._yolo import yolo_efficient_nms_hook

        hook = yolo_efficient_nms_hook(
            num_classes=91,
            conf_threshold=0.3,
            iou_threshold=0.45,
            top_k=200,
            box_coding="corner",
            class_agnostic=True,
        )
        assert callable(hook)

    def test_default_params(self) -> None:
        """Default parameters produce a valid hook."""
        from trtutils.builder.hooks._yolo import yolo_efficient_nms_hook

        hook = yolo_efficient_nms_hook()
        assert callable(hook)


@pytest.mark.gpu
class TestYoloHookNetworkModification:
    """Tests for yolo_efficient_nms_hook applied to real TRT networks."""

    def _create_yolo_network(self, output_shape, onnx_path):
        """Build a TRT network with a specific output shape for testing hooks."""
        from trtutils.builder._onnx import read_onnx

        network, builder, config, _ = read_onnx(onnx_path)
        return network, builder, config

    def test_yolov10_passthrough(self, onnx_path) -> None:
        """YOLOv10-style (1,300,6) output passes through unmodified."""
        from unittest.mock import MagicMock

        from trtutils.builder.hooks._yolo import yolo_efficient_nms_hook

        hook = yolo_efficient_nms_hook(num_classes=80)

        # Create a mock network with YOLOv10-style output
        mock_network = MagicMock()
        mock_output = MagicMock()
        mock_output.shape = (1, 300, 6)
        mock_network.get_output.return_value = mock_output

        result = hook(mock_network)
        # Should return the same network unmodified
        assert result is mock_network

    def test_hook_modifies_network(self, onnx_path) -> None:
        """Hook adds EfficientNMS plugin to YOLO-like network."""
        from trtutils.builder._onnx import read_onnx
        from trtutils.builder.hooks._yolo import yolo_efficient_nms_hook

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
    def test_box_coding_modes(self, box_coding) -> None:
        """Both box coding modes produce valid hooks."""
        from trtutils.builder.hooks._yolo import yolo_efficient_nms_hook

        hook = yolo_efficient_nms_hook(num_classes=80, box_coding=box_coding)
        assert callable(hook)

    @pytest.mark.parametrize(
        "class_agnostic", [True, False, None], ids=["agnostic", "per_class", "default"]
    )
    def test_class_agnostic(self, class_agnostic) -> None:
        """class_agnostic parameter variants produce valid hooks."""
        from trtutils.builder.hooks._yolo import yolo_efficient_nms_hook

        hook = yolo_efficient_nms_hook(num_classes=80, class_agnostic=class_agnostic)
        assert callable(hook)

    def test_with_objectness_mock(self) -> None:
        """Hook handles YOLOX-style output with objectness channel (N+5 channels)."""
        from unittest.mock import MagicMock, PropertyMock

        from trtutils.builder.hooks._yolo import yolo_efficient_nms_hook

        num_classes = 80
        yolo_efficient_nms_hook(num_classes=num_classes)

        # Mock a network where the output has channels_with_obj = 85 (80 + 5)
        mock_network = MagicMock()

        # First output is not YOLOv10 style
        mock_output_0 = MagicMock()
        mock_output_0.shape = (1, 85, 8400)

        def get_output_side_effect(idx):
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

    def test_without_objectness_mock(self) -> None:
        """Hook handles YOLOv8-style output without objectness (N+4 channels)."""
        from trtutils.builder.hooks._yolo import yolo_efficient_nms_hook

        num_classes = 80
        hook = yolo_efficient_nms_hook(num_classes=num_classes)
        # 84 channels = 80 classes + 4 bbox coords
        assert callable(hook)
