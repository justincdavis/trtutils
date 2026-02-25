# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for the download module API."""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# CPU tests -- config loading, model listing, validation
# ---------------------------------------------------------------------------
class TestLoadModelConfigs:
    """Tests for load_model_configs()."""

    @pytest.mark.cpu
    def test_returns_dict(self, model_configs: dict) -> None:
        """Configs should be a non-empty dict of dicts."""
        assert isinstance(model_configs, dict)
        assert len(model_configs) > 0

    @pytest.mark.cpu
    def test_each_family_is_dict(self, model_configs: dict) -> None:
        """Each top-level value should itself be a dict of model entries."""
        for family_name, family in model_configs.items():
            assert isinstance(family, dict), f"Expected dict for family {family_name!r}"
            for model_name, config in family.items():
                assert isinstance(config, dict), f"Expected dict config for model {model_name!r}"

    @pytest.mark.cpu
    def test_config_files_match_known_families(
        self,
        model_configs: dict,
    ) -> None:
        """At least a few well-known config families should exist."""
        known = {"yolov8", "yolov10", "yolov11", "rfdetr", "dfine"}
        present = set(model_configs.keys())
        missing = known - present
        assert not missing, f"Missing expected config families: {missing}"


class TestGetSupportedModels:
    """Tests for get_supported_models()."""

    @pytest.mark.cpu
    def test_returns_list(self, supported_models: list) -> None:
        """Should return a non-empty list of strings."""
        assert isinstance(supported_models, list)
        assert len(supported_models) > 0

    @pytest.mark.cpu
    def test_all_entries_are_strings(self, supported_models: list) -> None:
        for name in supported_models:
            assert isinstance(name, str)

    @pytest.mark.cpu
    def test_known_models_present(self, supported_models: list) -> None:
        """A few well-known model names should be present."""
        expected_subset = {"yolov10n", "yolov11n"}
        present = set(supported_models)
        missing = expected_subset - present
        assert not missing, f"Missing expected models: {missing}"


# ---------------------------------------------------------------------------
# download_model -- error paths (no network access required)
# ---------------------------------------------------------------------------
class TestDownloadModelErrors:
    """Error-path tests for download_model (no real downloads)."""

    @pytest.mark.cpu
    def test_unsupported_model_raises(self, download_tmp_dir: Path) -> None:
        """Requesting an unknown model name should raise ValueError."""
        from trtutils.download import download_model

        with pytest.raises(ValueError, match="not supported"):
            download_model("totally_fake_model_xyz", download_tmp_dir)

    @pytest.mark.cpu
    def test_empty_model_name_raises(self, download_tmp_dir: Path) -> None:
        """An empty string is not a valid model name."""
        from trtutils.download import download_model

        with pytest.raises(ValueError, match="not supported"):
            download_model("", download_tmp_dir)


# ---------------------------------------------------------------------------
# download (high-level) -- error paths
# ---------------------------------------------------------------------------
class TestDownloadErrors:
    """Error-path tests for the high-level download() function."""

    @pytest.mark.cpu
    def test_unsupported_model_raises(self, download_tmp_dir: Path) -> None:
        """download() should propagate ValueError for unknown models."""
        from trtutils.download import download

        with pytest.raises((ValueError, RuntimeError)):
            download(
                "totally_fake_model_xyz",
                download_tmp_dir / "out.onnx",
            )


# ---------------------------------------------------------------------------
# download_model -- mock-based functional tests
# ---------------------------------------------------------------------------
class TestDownloadModelMocked:
    """Functional tests using mocks to avoid real network calls."""

    @pytest.mark.cpu
    def test_valid_model_calls_make_venv(
        self,
        download_tmp_dir: Path,
    ) -> None:
        """download_model should call make_venv for a valid model."""
        from trtutils.download._download import download_model

        fake_python = download_tmp_dir / ".venv" / "bin" / "python"
        fake_bin = download_tmp_dir / ".venv" / "bin"
        fake_python.parent.mkdir(parents=True, exist_ok=True)
        fake_python.touch()

        with patch(
            "trtutils.download._download.make_venv",
            return_value=(fake_python, fake_bin),
        ) as mock_venv, patch(
            "trtutils.download._download.export_yolov10",
            return_value=download_tmp_dir / "model.onnx",
        ):
            (download_tmp_dir / "model.onnx").touch()
            with contextlib.suppress(Exception):
                download_model("yolov10n", download_tmp_dir)
            mock_venv.assert_called_once()


# ---------------------------------------------------------------------------
# Simplify -- validation
# ---------------------------------------------------------------------------
class TestSimplifyValidation:
    """Tests for the simplify module tool validation."""

    @pytest.mark.cpu
    def test_unknown_tool_raises(self, tmp_path: Path) -> None:
        """Unknown tool name should raise ValueError."""
        from trtutils.download._simplify import simplify

        fake_path = tmp_path / "fake.onnx"
        with pytest.raises(ValueError, match="Unknown simplification tools"):
            simplify(fake_path, tools=["not_a_real_tool"])

    @pytest.mark.cpu
    def test_valid_tool_names_accepted(self) -> None:
        """Valid tool names should not raise ValueError."""
        from trtutils.download._simplify import _VALID_TOOLS

        assert "polygraphy" in _VALID_TOOLS
        assert "onnxslim" in _VALID_TOOLS
        assert "onnxsim" in _VALID_TOOLS


# ---------------------------------------------------------------------------
# Tools -- uv availability checks
# ---------------------------------------------------------------------------
class TestToolsUvChecks:
    """Tests for uv availability/version checking."""

    @pytest.mark.cpu
    def test_check_uv_available_no_uv(self) -> None:
        """Should raise RuntimeError when uv is not on PATH."""
        from trtutils.download._tools import check_uv_available

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="uv is not available"):
                check_uv_available()

    @pytest.mark.cpu
    def test_check_uv_available_with_uv(self) -> None:
        """Should not raise when uv is found on PATH."""
        from trtutils.download._tools import check_uv_available

        with patch("shutil.which", return_value="/usr/bin/uv"):
            check_uv_available()


# ---------------------------------------------------------------------------
# handle_imgsz
# ---------------------------------------------------------------------------
class TestHandleImgsz:
    """Tests for the handle_imgsz utility."""

    @pytest.mark.cpu
    def test_default_value(self) -> None:
        from trtutils.download._tools import handle_imgsz

        result = handle_imgsz(None, 640, "test_model")
        assert result == 640

    @pytest.mark.cpu
    def test_custom_value_passthrough(self) -> None:
        from trtutils.download._tools import handle_imgsz

        result = handle_imgsz(320, 640, "test_model")
        assert result == 320

    @pytest.mark.cpu
    def test_enforce_overrides_custom(self) -> None:
        from trtutils.download._tools import handle_imgsz

        result = handle_imgsz(320, 640, "test_model", enforce=True)
        assert result == 640

    @pytest.mark.cpu
    def test_adjust_div_rounds_down(self) -> None:
        from trtutils.download._tools import handle_imgsz

        result = handle_imgsz(641, 640, "test_model", adjust_div=32)
        assert result % 32 == 0
        assert result == 640

    @pytest.mark.cpu
    def test_enforce_same_as_default_is_noop(self) -> None:
        from trtutils.download._tools import handle_imgsz

        result = handle_imgsz(640, 640, "test_model", enforce=True)
        assert result == 640

    @pytest.mark.cpu
    def test_adjust_div_already_divisible(self) -> None:
        from trtutils.download._tools import handle_imgsz

        result = handle_imgsz(640, 640, "test_model", adjust_div=32)
        assert result == 640


# ---------------------------------------------------------------------------
# Export function routing (mock-based, no network)
# ---------------------------------------------------------------------------
class TestExportFunctionRouting:
    """Verify download_model routes to the correct export function per model."""

    def _run_routing_test(
        self,
        download_tmp_dir: Path,
        model: str,
        export_func_name: str,
    ) -> None:
        from trtutils.download._download import download_model

        fake_python = download_tmp_dir / ".venv" / "bin" / "python"
        fake_bin = download_tmp_dir / ".venv" / "bin"
        fake_python.parent.mkdir(parents=True, exist_ok=True)
        fake_python.touch()
        fake_output = download_tmp_dir / "model.onnx"
        fake_output.touch()

        with patch(
            "trtutils.download._download.make_venv",
            return_value=(fake_python, fake_bin),
        ), patch(
            f"trtutils.download._download.{export_func_name}",
            return_value=fake_output,
        ) as mock_export:
            download_model(model, download_tmp_dir)
            mock_export.assert_called_once()

    @pytest.mark.cpu
    @pytest.mark.parametrize(
        ("model", "export_func_name"),
        [
            pytest.param("yolov8n", "export_ultralytics", id="ultralytics-yolov8"),
            pytest.param("yolov11n", "export_ultralytics", id="ultralytics-yolov11"),
            pytest.param("yolov7t", "export_yolov7", id="yolov7"),
            pytest.param("yolov9t", "export_yolov9", id="yolov9"),
            pytest.param("yolov10n", "export_yolov10", id="yolov10"),
            pytest.param("yolov12n", "export_yolov12", id="yolov12"),
            pytest.param("yolov13n", "export_yolov13", id="yolov13"),
            pytest.param("yoloxn", "export_yolox", id="yolox"),
            pytest.param("rtdetrv1_r18", "export_rtdetrv1", id="rtdetrv1"),
            pytest.param("rtdetrv2_r18", "export_rtdetrv2", id="rtdetrv2"),
            pytest.param("rtdetrv3_r18", "export_rtdetrv3", id="rtdetrv3"),
            pytest.param("dfine_n", "export_dfine", id="dfine"),
            pytest.param("deim_dfine_n", "export_deim", id="deim"),
            pytest.param("deimv2_atto", "export_deimv2", id="deimv2"),
            pytest.param("rfdetr_n", "export_rfdetr", id="rfdetr"),
            pytest.param("alexnet", "export_torchvision_classifier", id="torchvision"),
            pytest.param(
                "depth_anything_v2_small",
                "export_depth_anything_v2",
                id="depth_anything",
            ),
        ],
    )
    def test_routing_logic(
        self,
        download_tmp_dir: Path,
        model: str,
        export_func_name: str,
    ) -> None:
        """download_model should route each model to the correct export function."""
        self._run_routing_test(download_tmp_dir, model, export_func_name)


# ---------------------------------------------------------------------------
# get_valid_models
# ---------------------------------------------------------------------------
class TestGetValidModels:
    """Tests for the get_valid_models utility from _utils."""

    @pytest.mark.cpu
    @pytest.mark.parametrize(
        ("model_type", "expected_count"),
        [
            pytest.param("yolov8", 5, id="yolov8"),
            pytest.param("yolov10", 11, id="yolov10"),
            pytest.param("rtdetrv1", 7, id="rtdetrv1"),
            pytest.param("rfdetr", 3, id="rfdetr"),
            pytest.param("deimv2", 8, id="deimv2"),
            pytest.param("yolox", 7, id="yolox"),
            pytest.param("dfine", 5, id="dfine"),
        ],
    )
    def test_model_count_per_family(
        self,
        model_type: str,
        expected_count: int,
    ) -> None:
        """Each family should have the expected number of models."""
        from trtutils.models._utils import get_valid_models

        models = get_valid_models(model_type)
        assert len(models) == expected_count

    @pytest.mark.cpu
    def test_returns_list_of_strings(self) -> None:
        """get_valid_models should return a list of strings."""
        from trtutils.models._utils import get_valid_models

        models = get_valid_models("yolov8")
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)

    @pytest.mark.cpu
    def test_invalid_family_raises(self) -> None:
        """Requesting a non-existent family should raise KeyError."""
        from trtutils.models._utils import get_valid_models

        with pytest.raises(KeyError):
            get_valid_models("nonexistent_family")


# ---------------------------------------------------------------------------
# download_model -- simplify and requirements_export branches
# ---------------------------------------------------------------------------
class TestDownloadModelSimplifyBranch:
    """Tests for the simplify parameter routing in download_model."""

    @pytest.mark.cpu
    def test_simplify_bool_true_calls_simplify(
        self,
        download_tmp_dir: Path,
    ) -> None:
        """simplify=True should invoke _simplify.simplify with tools=None."""
        from trtutils.download._download import download_model

        fake_python = download_tmp_dir / ".venv" / "bin" / "python"
        fake_bin = download_tmp_dir / ".venv" / "bin"
        fake_python.parent.mkdir(parents=True, exist_ok=True)
        fake_python.touch()
        fake_output = download_tmp_dir / "model.onnx"
        fake_output.touch()

        with patch(
            "trtutils.download._download.make_venv",
            return_value=(fake_python, fake_bin),
        ), patch(
            "trtutils.download._download.export_yolov10",
            return_value=fake_output,
        ), patch(
            "trtutils.download._download._simplify.simplify",
        ) as mock_simplify:
            download_model("yolov10n", download_tmp_dir, simplify=True)
            mock_simplify.assert_called_once()
            _, kwargs = mock_simplify.call_args
            assert kwargs.get("tools") is None or mock_simplify.call_args[1].get("tools") is None

    @pytest.mark.cpu
    def test_simplify_list_passes_tools(
        self,
        download_tmp_dir: Path,
    ) -> None:
        """simplify=['onnxslim'] should pass tools=['onnxslim']."""
        from trtutils.download._download import download_model

        fake_python = download_tmp_dir / ".venv" / "bin" / "python"
        fake_bin = download_tmp_dir / ".venv" / "bin"
        fake_python.parent.mkdir(parents=True, exist_ok=True)
        fake_python.touch()
        fake_output = download_tmp_dir / "model.onnx"
        fake_output.touch()

        with patch(
            "trtutils.download._download.make_venv",
            return_value=(fake_python, fake_bin),
        ), patch(
            "trtutils.download._download.export_yolov10",
            return_value=fake_output,
        ), patch(
            "trtutils.download._download._simplify.simplify",
        ) as mock_simplify:
            download_model("yolov10n", download_tmp_dir, simplify=["onnxslim"])
            mock_simplify.assert_called_once()
            call_kwargs = mock_simplify.call_args[1]
            assert call_kwargs["tools"] == ["onnxslim"]

    @pytest.mark.cpu
    def test_simplify_string_wraps_in_list(
        self,
        download_tmp_dir: Path,
    ) -> None:
        """simplify='onnxslim' (string) should be wrapped in a list."""
        from trtutils.download._download import download_model

        fake_python = download_tmp_dir / ".venv" / "bin" / "python"
        fake_bin = download_tmp_dir / ".venv" / "bin"
        fake_python.parent.mkdir(parents=True, exist_ok=True)
        fake_python.touch()
        fake_output = download_tmp_dir / "model.onnx"
        fake_output.touch()

        with patch(
            "trtutils.download._download.make_venv",
            return_value=(fake_python, fake_bin),
        ), patch(
            "trtutils.download._download.export_yolov10",
            return_value=fake_output,
        ), patch(
            "trtutils.download._download._simplify.simplify",
        ) as mock_simplify:
            download_model("yolov10n", download_tmp_dir, simplify="onnxslim")
            mock_simplify.assert_called_once()
            call_kwargs = mock_simplify.call_args[1]
            assert call_kwargs["tools"] == ["onnxslim"]

    @pytest.mark.cpu
    def test_requirements_export_calls_export_requirements(
        self,
        download_tmp_dir: Path,
    ) -> None:
        """requirements_export should trigger export_requirements."""
        from trtutils.download._download import download_model

        fake_python = download_tmp_dir / ".venv" / "bin" / "python"
        fake_bin = download_tmp_dir / ".venv" / "bin"
        fake_python.parent.mkdir(parents=True, exist_ok=True)
        fake_python.touch()
        fake_output = download_tmp_dir / "model.onnx"
        fake_output.touch()
        req_path = download_tmp_dir / "requirements.txt"

        with patch(
            "trtutils.download._download.make_venv",
            return_value=(fake_python, fake_bin),
        ), patch(
            "trtutils.download._download.export_yolov10",
            return_value=fake_output,
        ), patch(
            "trtutils.download._download.export_requirements",
        ) as mock_export_req:
            download_model(
                "yolov10n",
                download_tmp_dir,
                requirements_export=req_path,
            )
            mock_export_req.assert_called_once()


# ---------------------------------------------------------------------------
# download (high-level) -- mock-based tests
# ---------------------------------------------------------------------------
class TestDownloadHighLevel:
    """Mock-based tests for the high-level download() function."""

    @pytest.mark.cpu
    def test_download_calls_download_model_and_copies(
        self,
        download_tmp_dir: Path,
    ) -> None:
        """download() should call download_model in a temp dir and copy output."""
        from trtutils.download._download import download

        output_path = download_tmp_dir / "output.onnx"
        download_tmp_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "trtutils.download._download.check_uv_version",
        ), patch(
            "trtutils.download._download.download_model",
        ) as mock_dm, patch(
            "trtutils.download._download.shutil.copy",
        ) as mock_copy:
            # download_model returns a path inside the temp dir
            mock_dm.return_value = Path("/tmp/fake/model.onnx")
            download("yolov10n", output_path)
            mock_dm.assert_called_once()
            mock_copy.assert_called_once()

    @pytest.mark.cpu
    def test_download_verbose_logs(
        self,
        download_tmp_dir: Path,
    ) -> None:
        """download() with verbose should log after completion."""
        from trtutils.download._download import download

        output_path = download_tmp_dir / "output.onnx"
        download_tmp_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "trtutils.download._download.check_uv_version",
        ), patch(
            "trtutils.download._download.download_model",
            return_value=Path("/tmp/fake/model.onnx"),
        ), patch(
            "trtutils.download._download.shutil.copy",
        ):
            # verbose=True triggers the LOG.info at end -- should not raise
            download("yolov10n", output_path, verbose=True)


# ---------------------------------------------------------------------------
# load_model_configs -- error handler branch
# ---------------------------------------------------------------------------
class TestLoadModelConfigsErrorHandler:
    """Test the JSON parse error handler in load_model_configs."""

    @pytest.mark.cpu
    def test_malformed_json_skipped_with_warning(self, tmp_path: Path) -> None:
        """If a JSON file is malformed, the family is skipped with a warning."""
        from pathlib import Path as _Path
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        # Create a fake malformed JSON file Path mock
        bad_file = MagicMock(spec=_Path)
        bad_file.stem = "bad_family"
        bad_file.name = "bad_family.json"
        bad_file.open.return_value.__enter__ = lambda s: s
        bad_file.open.return_value.__exit__ = MagicMock(return_value=False)
        bad_file.open.return_value.read = lambda: "not valid json {"

        # Call the underlying (non-cached) function
        from trtutils.download._download import load_model_configs

        orig_fn = load_model_configs.__wrapped__
        with _patch(
            "trtutils.download._download.Path.glob",
            return_value=[bad_file],
        ), _patch(
            "trtutils.download._download.json.load",
            side_effect=KeyError("bad_key"),
        ):
            result = orig_fn()
        # The bad family should be absent from results
        assert "bad_family" not in result
