"""Tests for DLA analysis -- can_run_on_dla(), build_dla_engine(), get_check_dla()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.mark.gpu
class TestCanRunOnDla:
    """Tests for can_run_on_dla() analysis function."""

    def test_returns_tuple(self, onnx_path):
        """can_run_on_dla returns (bool, list) tuple."""
        from trtutils.builder._dla import can_run_on_dla

        result = can_run_on_dla(onnx_path)
        assert isinstance(result, tuple)
        assert len(result) == 2
        full_dla, chunks = result
        assert isinstance(full_dla, bool)
        assert isinstance(chunks, list)

    def test_chunks_have_correct_structure(self, onnx_path):
        """Each chunk is (layers, start, end, on_dla) tuple."""
        from trtutils.builder._dla import can_run_on_dla

        _, chunks = can_run_on_dla(onnx_path)
        for chunk in chunks:
            assert len(chunk) == 4
            layers, start, end, on_dla = chunk
            assert isinstance(layers, list)
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert isinstance(on_dla, bool)

    def test_network_input_requires_config(self, onnx_path):
        """ValueError when passing network without config."""
        from trtutils.builder._dla import can_run_on_dla
        from trtutils.builder._onnx import read_onnx

        network, _, _, _ = read_onnx(onnx_path)
        with pytest.raises(ValueError, match="Config must be provided"):
            can_run_on_dla(network, config=None)

    def test_network_input_with_config(self, onnx_path):
        """can_run_on_dla accepts a pre-made network with config."""
        from trtutils.builder._dla import can_run_on_dla
        from trtutils.builder._onnx import read_onnx

        network, _, config, _ = read_onnx(onnx_path)
        full_dla, chunks = can_run_on_dla(network, config=config)
        assert isinstance(full_dla, bool)
        assert isinstance(chunks, list)

    def test_verbose_layers(self, onnx_path):
        """verbose_layers=True does not raise."""
        from trtutils.builder._dla import can_run_on_dla

        can_run_on_dla(onnx_path, verbose_layers=True)

    def test_verbose_chunks(self, onnx_path):
        """verbose_chunks=True does not raise."""
        from trtutils.builder._dla import can_run_on_dla

        can_run_on_dla(onnx_path, verbose_chunks=True)


@pytest.mark.gpu
class TestBuildDlaEngine:
    """Tests for build_dla_engine() with mocked build_engine."""

    def _make_batcher(self):
        """Create a SyntheticBatcher for DLA tests."""
        from trtutils.builder._batcher import SyntheticBatcher

        return SyntheticBatcher(
            shape=(3, 8, 8),
            dtype=np.dtype(np.float32),
            batch_size=1,
            num_batches=2,
            order="NCHW",
        )

    def test_build_basic(self, onnx_path, output_engine_path):
        """build_dla_engine runs without error."""
        from trtutils.builder._dla import build_dla_engine

        batcher = self._make_batcher()
        build_dla_engine(
            onnx_path,
            output_engine_path,
            data_batcher=batcher,
            dla_core=0,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_full_dla_path(self, onnx_path, output_engine_path):
        """When full_dla is True, build_engine is called with DLA default_device."""
        from trtutils.builder._dla import build_dla_engine

        batcher = self._make_batcher()
        # Mock can_run_on_dla to return full_dla=True
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

    def test_no_dla_chunks(self, onnx_path, output_engine_path):
        """No DLA-compatible layers → GPU-only build with warning."""
        from trtutils.builder._dla import build_dla_engine

        batcher = self._make_batcher()
        # Mock: full_dla=False, all chunks GPU-only
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

    def test_mixed_dla_gpu(self, onnx_path, output_engine_path):
        """Partial DLA → layer assignments with mixed devices."""
        from trtutils.builder._dla import build_dla_engine

        batcher = self._make_batcher()
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

    def test_max_chunks_limit(self, onnx_path, output_engine_path):
        """max_chunks parameter limits DLA chunk assignment."""
        from trtutils.builder._dla import build_dla_engine

        batcher = self._make_batcher()
        build_dla_engine(
            onnx_path,
            output_engine_path,
            data_batcher=batcher,
            dla_core=0,
            max_chunks=0,  # 0 means assign all qualifying chunks
            min_layers=0,
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_min_layers_filter(self, onnx_path, output_engine_path):
        """min_layers parameter filters small chunks."""
        from trtutils.builder._dla import build_dla_engine

        batcher = self._make_batcher()
        build_dla_engine(
            onnx_path,
            output_engine_path,
            data_batcher=batcher,
            dla_core=0,
            min_layers=99999,  # No chunk will be large enough
            optimization_level=1,
        )
        assert output_engine_path.exists()

    def test_verbose_output(self, onnx_path, output_engine_path):
        """verbose=True produces log output without error."""
        from trtutils.builder._dla import build_dla_engine

        batcher = self._make_batcher()
        build_dla_engine(
            onnx_path,
            output_engine_path,
            data_batcher=batcher,
            dla_core=0,
            min_layers=0,
            verbose=True,
            optimization_level=1,
        )
        assert output_engine_path.exists()


@pytest.mark.gpu
class TestGetCheckDla:
    """Tests for get_check_dla() utility function."""

    def test_returns_callable(self, onnx_path):
        """get_check_dla returns a callable."""
        from trtutils.builder._onnx import read_onnx
        from trtutils.builder._utils import get_check_dla

        _, _, config, _ = read_onnx(onnx_path)
        check_fn = get_check_dla(config)
        assert callable(check_fn)

    def test_function_accepts_layer(self, onnx_path):
        """Returned function can be called with a layer."""
        from trtutils.builder._onnx import read_onnx
        from trtutils.builder._utils import get_check_dla
        from trtutils.compat._libs import trt

        network, _, config, _ = read_onnx(onnx_path)
        check_fn = get_check_dla(config)
        # Assign DLA device type first (required for can_run_on_DLA)
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = 0
        if network.num_layers > 0:
            layer = network.get_layer(0)
            result = check_fn(layer)
            assert isinstance(result, bool)
