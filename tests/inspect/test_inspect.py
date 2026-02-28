# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the inspect module."""

from __future__ import annotations

from pathlib import Path

import pytest

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SIMPLE_ONNX = DATA_DIR / "simple.onnx"
YOLOV10_ONNX = DATA_DIR / "yolov10" / "yolov10n_640.onnx"


# ---------------------------------------------------------------------------
# inspect_engine -- GPU tests
# ---------------------------------------------------------------------------
class TestInspectEngine:
    """Tests for inspect_engine()."""

    def test_inspect_engine_from_path(self, build_test_engine) -> None:
        """inspect_engine should return a 4-tuple from a built engine."""
        engine_path = build_test_engine(SIMPLE_ONNX)

        from trtutils.inspect import inspect_engine

        result = inspect_engine(engine_path)

        assert isinstance(result, tuple)
        assert len(result) == 4

        mem_size, batch_size, inputs, outputs = result
        assert isinstance(mem_size, int)
        assert mem_size >= 0
        assert isinstance(batch_size, int)
        assert isinstance(inputs, list)
        assert isinstance(outputs, list)

    def test_inspect_engine_inputs_format(self, build_test_engine) -> None:
        """Each input tensor entry should be a 4-tuple."""
        engine_path = build_test_engine(SIMPLE_ONNX)

        from trtutils.inspect import inspect_engine

        _, _, inputs, _ = inspect_engine(engine_path)
        assert len(inputs) > 0
        for entry in inputs:
            assert isinstance(entry, tuple)
            assert len(entry) == 4
            name, _shape, _dtype, _fmt = entry
            assert isinstance(name, str)
            assert len(name) > 0

    def test_inspect_engine_outputs_format(self, build_test_engine) -> None:
        """Each output tensor entry should be a 4-tuple."""
        engine_path = build_test_engine(SIMPLE_ONNX)

        from trtutils.inspect import inspect_engine

        _, _, _, outputs = inspect_engine(engine_path)
        assert len(outputs) > 0
        for entry in outputs:
            assert isinstance(entry, tuple)
            assert len(entry) == 4
            name, _shape, _dtype, _fmt = entry
            assert isinstance(name, str)
            assert len(name) > 0

    def test_inspect_engine_verbose(self, build_test_engine) -> None:
        """verbose=True should not raise."""
        engine_path = build_test_engine(SIMPLE_ONNX)

        from trtutils.inspect import inspect_engine

        result = inspect_engine(engine_path, verbose=True)
        assert result is not None

    def test_inspect_engine_nonexistent_path_raises(self) -> None:
        """A nonexistent engine file should raise."""
        from trtutils.inspect import inspect_engine

        with pytest.raises(FileNotFoundError):
            inspect_engine("nonexistent_engine_abc123.engine")


# ---------------------------------------------------------------------------
# inspect_onnx_layers -- GPU tests (requires TensorRT for parsing)
# ---------------------------------------------------------------------------
class TestInspectOnnxLayers:
    """Tests for inspect_onnx_layers()."""

    def test_inspect_onnx_returns_list(self) -> None:
        """inspect_onnx_layers should return a list of layer tuples."""
        if not SIMPLE_ONNX.exists():
            pytest.skip("simple.onnx not found")

        from trtutils.inspect import inspect_onnx_layers

        result = inspect_onnx_layers(SIMPLE_ONNX)
        assert isinstance(result, list)

    def test_inspect_onnx_layer_format(self) -> None:
        """Each layer entry should be a 4-tuple (idx, name, type, prec)."""
        if not SIMPLE_ONNX.exists():
            pytest.skip("simple.onnx not found")

        from trtutils.inspect import inspect_onnx_layers

        layers = inspect_onnx_layers(SIMPLE_ONNX)
        assert len(layers) > 0
        for entry in layers:
            assert isinstance(entry, tuple)
            assert len(entry) == 4
            idx, name, _layer_type, _precision = entry
            assert isinstance(idx, int)
            assert isinstance(name, str)

    def test_inspect_onnx_verbose(self) -> None:
        """verbose=True should not raise."""
        if not SIMPLE_ONNX.exists():
            pytest.skip("simple.onnx not found")

        from trtutils.inspect import inspect_onnx_layers

        result = inspect_onnx_layers(SIMPLE_ONNX, verbose=True)
        assert result is not None

    def test_inspect_onnx_nonexistent_raises(self) -> None:
        """Nonexistent ONNX file should raise."""
        from trtutils.inspect import inspect_onnx_layers

        with pytest.raises(FileNotFoundError):
            inspect_onnx_layers("nonexistent_model_abc123.onnx")

    def test_inspect_onnx_yolov10(self) -> None:
        """YOLOv10 ONNX should parse into multiple layers."""
        if not YOLOV10_ONNX.exists():
            pytest.skip("yolov10n_640.onnx not found")

        from trtutils.inspect import inspect_onnx_layers

        layers = inspect_onnx_layers(YOLOV10_ONNX)
        assert len(layers) > 10  # YOLO models have many layers


# ---------------------------------------------------------------------------
# get_engine_names -- GPU tests
# ---------------------------------------------------------------------------
class TestGetEngineNames:
    """Tests for get_engine_names()."""

    def test_returns_two_lists(self, build_test_engine) -> None:
        """get_engine_names should return (input_names, output_names)."""
        engine_path = build_test_engine(SIMPLE_ONNX)

        from trtutils.inspect import get_engine_names

        result = get_engine_names(engine_path)
        assert isinstance(result, tuple)
        assert len(result) == 2
        input_names, output_names = result
        assert isinstance(input_names, list)
        assert isinstance(output_names, list)

    def test_names_are_strings(self, build_test_engine) -> None:
        """All returned names should be non-empty strings."""
        engine_path = build_test_engine(SIMPLE_ONNX)

        from trtutils.inspect import get_engine_names

        input_names, output_names = get_engine_names(engine_path)
        for name in input_names + output_names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_has_at_least_one_input_and_output(
        self,
        build_test_engine,
    ) -> None:
        """Engine should have at least one input and one output."""
        engine_path = build_test_engine(SIMPLE_ONNX)

        from trtutils.inspect import get_engine_names

        input_names, output_names = get_engine_names(engine_path)
        assert len(input_names) >= 1
        assert len(output_names) >= 1
