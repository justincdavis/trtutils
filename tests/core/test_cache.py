# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

import pytest

from trtutils.core import cache
from trtutils.core.cache import _delete_folder, _get_cache_file_path, get_cache_dir


# ---------------------------------------------------------------------------
# get_cache_dir
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestGetCacheDir:
    """Tests for get_cache_dir()."""

    def test_returns_path(self) -> None:
        result = get_cache_dir()
        assert isinstance(result, Path)
        assert result.name == "_engine_cache"

    def test_is_idempotent(self) -> None:
        assert get_cache_dir() == get_cache_dir()

    def test_directory_parent_exists(self) -> None:
        assert get_cache_dir().parent.exists()


# ---------------------------------------------------------------------------
# _get_cache_file_path
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestGetCacheFilePath:
    """Tests for _get_cache_file_path extension handling."""

    @pytest.mark.parametrize(
        ("filename", "expected_suffix"),
        [
            pytest.param("mymodel", "mymodel.engine", id="no_ext"),
            pytest.param("mymodel.engine", "mymodel.engine", id="engine_ext"),
            pytest.param("mymodel.onnx", "mymodel.onnx", id="onnx_ext"),
            pytest.param("global.cache", "global.cache", id="cache_ext"),
            pytest.param("mymodel.txt", "mymodel.txt.engine", id="invalid_ext"),
            pytest.param("my.weird.model", "my.weird.model.engine", id="multi_dot"),
        ],
    )
    def test_extension_handling(self, filename, expected_suffix) -> None:
        result = _get_cache_file_path(filename, "engine")
        assert result == get_cache_dir() / expected_suffix


# ---------------------------------------------------------------------------
# _delete_folder
# ---------------------------------------------------------------------------
def _setup_empty(d: Path) -> None:
    d.mkdir()


def _setup_files(d: Path) -> None:
    d.mkdir()
    (d / "a.txt").write_text("hello")
    (d / "b.txt").write_text("world")


def _setup_nested(d: Path) -> None:
    d.mkdir()
    sub = d / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("content")


def _setup_mixed(d: Path) -> None:
    d.mkdir()
    (d / "file.txt").write_text("data")
    sub = d / "child"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested data")


@pytest.mark.cpu
class TestDeleteFolder:
    """Tests for _delete_folder helper."""

    @pytest.mark.parametrize(
        "setup_fn",
        [
            pytest.param(_setup_empty, id="empty"),
            pytest.param(_setup_files, id="with_files"),
            pytest.param(_setup_nested, id="with_subdirs"),
            pytest.param(_setup_mixed, id="mixed"),
        ],
    )
    def test_delete_folder(self, tmp_path, setup_fn) -> None:
        d = tmp_path / "target"
        setup_fn(d)
        _delete_folder(d)
        assert not d.exists()


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestClear:
    """Tests for clear()."""

    @pytest.mark.parametrize(
        "setup_fn",
        [
            pytest.param(lambda _d: None, id="empty"),
            pytest.param(lambda d: (d / "test.engine").write_text("data"), id="with_file"),
            pytest.param(
                lambda d: (d / "sub").mkdir() or (d / "sub" / "f.txt").write_text("t"),
                id="nested",
            ),
        ],
    )
    def test_clear_recreates_empty_dir(self, patched_cache_dir, setup_fn) -> None:
        setup_fn(patched_cache_dir)
        cache.clear(no_warn=True)
        assert patched_cache_dir.exists()
        assert list(patched_cache_dir.iterdir()) == []

    @pytest.mark.parametrize(
        "no_warn",
        [
            pytest.param(None, id="default"),
            pytest.param(False, id="explicit_false"),
        ],
    )
    def test_clear_warning_path(self, patched_cache_dir, no_warn) -> None:
        kwargs = {} if no_warn is None else {"no_warn": no_warn}
        cache.clear(**kwargs)
        assert patched_cache_dir.exists()


# ---------------------------------------------------------------------------
# query / store
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestQueryStore:
    """Tests for query, query_file, store, store_file."""

    @pytest.mark.parametrize(
        ("query_fn", "query_args", "expected_name"),
        [
            pytest.param(cache.query, ("nofile",), None, id="query"),
            pytest.param(cache.query_file, ("nofile", "cache"), "nofile.cache", id="query_file"),
        ],
    )
    def test_query_nonexistent(self, patched_cache_dir, query_fn, query_args, expected_name) -> None:
        exists, path = query_fn(*query_args)
        assert exists is False
        assert isinstance(path, Path)
        if expected_name:
            assert path.name == expected_name

    def test_store_then_query(self, patched_cache_dir, tmp_path) -> None:
        src = tmp_path / "test.engine"
        src.write_text("engine data")
        cache.store(src)
        exists, path = cache.query("test")
        assert exists is True
        assert path.exists()

    def test_store_file_then_query_file(self, patched_cache_dir, tmp_path) -> None:
        src = tmp_path / "data.txt"
        src.write_text("some data")
        new_path = cache.store_file(src, cache_filename="data.cache")
        assert new_path.exists()
        exists, _path = cache.query_file("data", "cache")
        assert exists is True

    @pytest.mark.parametrize(
        ("overwrite", "expected_content"),
        [
            pytest.param(True, "version2", id="overwrite"),
            pytest.param(False, "version1", id="no_overwrite"),
        ],
    )
    def test_store_overwrite(self, patched_cache_dir, tmp_path, overwrite, expected_content) -> None:
        src1 = tmp_path / "v1.engine"
        src1.write_text("version1")
        src2 = tmp_path / "v2.engine"
        src2.write_text("version2")
        cache.store_file(src1, cache_filename="model.engine")
        cache.store_file(src2, cache_filename="model.engine", overwrite=overwrite)
        stored = patched_cache_dir / "model.engine"
        assert stored.read_text() == expected_content

    def test_store_clear_old_true(self, patched_cache_dir, tmp_path) -> None:
        src = tmp_path / "model.engine"
        src.write_text("data")
        cache.store(src, clear_old=True)
        assert not src.exists()

    def test_store_clear_old_no_overwrite_existing(self, patched_cache_dir, tmp_path) -> None:
        (patched_cache_dir / "model.engine").write_text("old")
        src = tmp_path / "model.engine"
        src.write_text("new")
        cache.store(src, overwrite=False, clear_old=True)
        assert not src.exists()
        assert (patched_cache_dir / "model.engine").read_text() == "old"

    def test_store_file_uses_original_name(self, patched_cache_dir, tmp_path) -> None:
        src = tmp_path / "original.engine"
        src.write_text("data")
        result = cache.store_file(src)
        assert result.name == "original.engine"
        assert result.exists()

    def test_store_clear_old_with_overwrite(self, patched_cache_dir, tmp_path) -> None:
        (patched_cache_dir / "model.engine").write_text("old")
        src = tmp_path / "model.engine"
        src.write_text("new")
        cache.store(src, overwrite=True, clear_old=True)
        assert not src.exists()
        assert (patched_cache_dir / "model.engine").read_text() == "new"


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestRemove:
    """Tests for remove and remove_file."""

    @pytest.mark.parametrize(
        ("setup_name", "remove_fn", "remove_args"),
        [
            pytest.param("model.engine", cache.remove, ("model",), id="remove"),
            pytest.param("data.cache", cache.remove_file, ("data", "cache"), id="remove_file"),
            pytest.param(
                "model.engine",
                cache.remove_file,
                ("model.engine", "engine"),
                id="remove_file_ext_in_name",
            ),
        ],
    )
    def test_remove_existing(self, patched_cache_dir, setup_name, remove_fn, remove_args) -> None:
        (patched_cache_dir / setup_name).write_text("data")
        remove_fn(*remove_args)
        assert not (patched_cache_dir / setup_name).exists()

    @pytest.mark.parametrize(
        ("remove_fn", "remove_args"),
        [
            pytest.param(cache.remove, ("nonexistent",), id="remove"),
            pytest.param(cache.remove_file, ("nonexistent", "cache"), id="remove_file"),
        ],
    )
    def test_remove_nonexistent_raises(self, patched_cache_dir, remove_fn, remove_args) -> None:
        with pytest.raises(FileNotFoundError):
            remove_fn(*remove_args)


# ---------------------------------------------------------------------------
# timing cache
# ---------------------------------------------------------------------------
@pytest.mark.cpu
class TestTimingCache:
    """Tests for timing cache operations."""

    def test_query_timing_cache_missing(self, patched_cache_dir) -> None:
        exists, path = cache.query_timing_cache()
        assert exists is False
        assert path.name == "global.cache"

    def test_store_and_query_roundtrip(self, patched_cache_dir, tmp_path) -> None:
        src = tmp_path / "timing.cache"
        src.write_bytes(b"timing data bytes")
        cache.store_timing_cache(src)
        exists, path = cache.query_timing_cache()
        assert exists is True
        assert path.read_bytes() == b"timing data bytes"

    def test_store_timing_cache_overwrite(self, patched_cache_dir, tmp_path) -> None:
        src1 = tmp_path / "v1.cache"
        src1.write_bytes(b"v1")
        cache.store_timing_cache(src1)
        src2 = tmp_path / "v2.cache"
        src2.write_bytes(b"v2")
        cache.store_timing_cache(src2, overwrite=True)
        _, path = cache.query_timing_cache()
        assert path.read_bytes() == b"v2"

    def test_store_timing_cache_clear_old(self, patched_cache_dir, tmp_path) -> None:
        src = tmp_path / "timing.cache"
        src.write_bytes(b"data")
        cache.store_timing_cache(src, clear_old=True)
        assert not src.exists()

    def test_save_timing_cache_to_global(self, patched_cache_dir) -> None:
        class MockTimingCache:
            def serialize(self) -> bytes:
                return b"serialized cache data"

        result = cache.save_timing_cache_to_global(MockTimingCache())
        assert result.exists()
        assert result.name == "global.cache"
        exists, path = cache.query_timing_cache()
        assert exists is True
        assert path.read_bytes() == b"serialized cache data"

    @pytest.mark.parametrize(
        ("overwrite", "expected"),
        [
            pytest.param(True, b"second", id="overwrite"),
            pytest.param(False, b"first", id="no_overwrite"),
        ],
    )
    def test_save_timing_cache_to_global_overwrite(
        self, patched_cache_dir, overwrite, expected
    ) -> None:
        class MockTimingCache:
            def __init__(self, data) -> None:
                self._data = data

            def serialize(self):
                return self._data

        cache.save_timing_cache_to_global(MockTimingCache(b"first"))
        cache.save_timing_cache_to_global(MockTimingCache(b"second"), overwrite=overwrite)
        _, path = cache.query_timing_cache()
        assert path.read_bytes() == expected
