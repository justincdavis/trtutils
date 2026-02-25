"""Tests for src/trtutils/core/cache.py -- 100% branch coverage target."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.cpu
class TestGetCacheDir:
    """Tests for get_cache_dir()."""

    def test_returns_path(self) -> None:
        """get_cache_dir() should return a Path object."""
        from trtutils.core.cache import get_cache_dir

        result = get_cache_dir()
        assert isinstance(result, Path)

    def test_points_to_engine_cache(self) -> None:
        """get_cache_dir() should point to _engine_cache subdir."""
        from trtutils.core.cache import get_cache_dir

        result = get_cache_dir()
        assert result.name == "_engine_cache"

    def test_is_idempotent(self) -> None:
        """Multiple calls return the same directory."""
        from trtutils.core.cache import get_cache_dir

        assert get_cache_dir() == get_cache_dir()

    def test_directory_exists(self) -> None:
        """The cache directory should already exist in the source tree."""
        from trtutils.core.cache import get_cache_dir

        assert get_cache_dir().exists()


@pytest.mark.cpu
class TestGetCacheFilePath:
    """Tests for internal _get_cache_file_path helper."""

    def test_filename_without_extension(self) -> None:
        """Filenames without dots get the extension appended."""
        from trtutils.core.cache import _get_cache_file_path, get_cache_dir

        result = _get_cache_file_path("mymodel", "engine")
        assert result == get_cache_dir() / "mymodel.engine"

    def test_filename_with_valid_extension(self) -> None:
        """Filenames with a known valid extension are used as-is."""
        from trtutils.core.cache import _get_cache_file_path, get_cache_dir

        result = _get_cache_file_path("mymodel.engine", "engine")
        assert result == get_cache_dir() / "mymodel.engine"

    def test_filename_with_onnx_extension(self) -> None:
        """Filenames with .onnx are valid extensions and used as-is."""
        from trtutils.core.cache import _get_cache_file_path, get_cache_dir

        result = _get_cache_file_path("mymodel.onnx", "engine")
        assert result == get_cache_dir() / "mymodel.onnx"

    def test_filename_with_cache_extension(self) -> None:
        """Filenames with .cache are valid extensions and used as-is."""
        from trtutils.core.cache import _get_cache_file_path, get_cache_dir

        result = _get_cache_file_path("global.cache", "engine")
        assert result == get_cache_dir() / "global.cache"

    def test_filename_with_invalid_extension(self) -> None:
        """Filenames with non-valid extensions get the given extension appended."""
        from trtutils.core.cache import _get_cache_file_path, get_cache_dir

        result = _get_cache_file_path("mymodel.txt", "engine")
        assert result == get_cache_dir() / "mymodel.txt.engine"

    def test_filename_with_dot_no_valid_ext(self) -> None:
        """Filenames with dots but no recognized extension get extension appended."""
        from trtutils.core.cache import _get_cache_file_path, get_cache_dir

        result = _get_cache_file_path("my.weird.model", "engine")
        assert result == get_cache_dir() / "my.weird.model.engine"


@pytest.mark.cpu
class TestDeleteFolder:
    """Tests for internal _delete_folder helper."""

    def test_delete_empty_dir(self, tmp_path) -> None:
        """Empty directory is removed without error."""
        from trtutils.core.cache import _delete_folder

        d = tmp_path / "empty"
        d.mkdir()
        _delete_folder(d)
        assert not d.exists()

    def test_delete_dir_with_files(self, tmp_path) -> None:
        """Directory with files is fully removed."""
        from trtutils.core.cache import _delete_folder

        d = tmp_path / "withfiles"
        d.mkdir()
        (d / "a.txt").write_text("hello")
        (d / "b.txt").write_text("world")
        _delete_folder(d)
        assert not d.exists()

    def test_delete_dir_with_subdirs(self, tmp_path) -> None:
        """Directory with nested subdirectories is fully removed."""
        from trtutils.core.cache import _delete_folder

        d = tmp_path / "nested"
        d.mkdir()
        sub = d / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("content")
        _delete_folder(d)
        assert not d.exists()

    def test_delete_dir_mixed(self, tmp_path) -> None:
        """Directory with mixed files and subdirs is fully removed."""
        from trtutils.core.cache import _delete_folder

        d = tmp_path / "mixed"
        d.mkdir()
        (d / "file.txt").write_text("data")
        sub = d / "child"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested data")
        _delete_folder(d)
        assert not d.exists()


@pytest.mark.cpu
class TestClear:
    """Tests for clear()."""

    def test_clear_creates_empty_dir(self, tmp_path, monkeypatch) -> None:
        """clear() should remove contents and recreate the directory."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        (cache_dir / "test.engine").write_text("data")

        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        cache.clear(no_warn=True)
        assert cache_dir.exists()
        assert list(cache_dir.iterdir()) == []

    def test_clear_empty_dir(self, tmp_path, monkeypatch) -> None:
        """clear() on an empty dir should not error."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()

        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        cache.clear(no_warn=True)
        assert cache_dir.exists()

    def test_clear_with_nested(self, tmp_path, monkeypatch) -> None:
        """clear() on directory with nested content works."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        sub = cache_dir / "sub"
        sub.mkdir()
        (sub / "file.txt").write_text("text")

        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        cache.clear(no_warn=True)
        assert cache_dir.exists()
        assert list(cache_dir.iterdir()) == []

    def test_clear_issues_warning_by_default(self, tmp_path, monkeypatch) -> None:
        """clear() with no_warn=None (default) should issue a log warning."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()

        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        # Should not error; the LOG.warning call is exercised
        cache.clear()
        assert cache_dir.exists()

    def test_clear_no_warn_false_issues_warning(self, tmp_path, monkeypatch) -> None:
        """clear() with no_warn=False should issue the warning."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()

        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        cache.clear(no_warn=False)
        assert cache_dir.exists()


@pytest.mark.cpu
class TestQueryStore:
    """Tests for query, query_file, store, store_file."""

    def test_query_nonexistent(self, tmp_path, monkeypatch) -> None:
        """query() for a non-existent file returns (False, path)."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        exists, path = cache.query("nofile")
        assert exists is False
        assert isinstance(path, Path)

    def test_query_file_nonexistent(self, tmp_path, monkeypatch) -> None:
        """query_file() for a non-existent file returns (False, path)."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        exists, path = cache.query_file("nofile", "cache")
        assert exists is False
        assert path.name == "nofile.cache"

    def test_store_then_query(self, tmp_path, monkeypatch) -> None:
        """Store a file then query should return (True, path)."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        src = tmp_path / "test.engine"
        src.write_text("engine data")

        cache.store(src)
        exists, path = cache.query("test")
        assert exists is True
        assert path.exists()

    def test_store_file_then_query_file(self, tmp_path, monkeypatch) -> None:
        """store_file then query_file roundtrip works."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        src = tmp_path / "data.txt"
        src.write_text("some data")

        new_path = cache.store_file(src, cache_filename="data.cache")
        assert new_path.exists()

        exists, _path = cache.query_file("data", "cache")
        assert exists is True

    def test_store_overwrite_true(self, tmp_path, monkeypatch) -> None:
        """Store with overwrite=True replaces existing file."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        src1 = tmp_path / "v1.engine"
        src1.write_text("version1")

        src2 = tmp_path / "v2.engine"
        src2.write_text("version2")

        # Both have the same name when stored
        cache.store_file(src1, cache_filename="model.engine")
        cache.store_file(src2, cache_filename="model.engine", overwrite=True)

        stored = cache_dir / "model.engine"
        assert stored.read_text() == "version2"

    def test_store_overwrite_false_keeps_old(self, tmp_path, monkeypatch) -> None:
        """Store with overwrite=False keeps the original file."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        src1 = tmp_path / "v1.engine"
        src1.write_text("version1")

        src2 = tmp_path / "v2.engine"
        src2.write_text("version2")

        cache.store_file(src1, cache_filename="model.engine")
        cache.store_file(src2, cache_filename="model.engine", overwrite=False)

        stored = cache_dir / "model.engine"
        assert stored.read_text() == "version1"

    def test_store_clear_old_true(self, tmp_path, monkeypatch) -> None:
        """Store with clear_old=True removes the source file."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        src = tmp_path / "model.engine"
        src.write_text("data")

        cache.store(src, clear_old=True)
        assert not src.exists()

    def test_store_clear_old_no_overwrite_existing(self, tmp_path, monkeypatch: object) -> None:
        """Store with clear_old=True, overwrite=False and existing target removes source."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        # Pre-populate the cache
        (cache_dir / "model.engine").write_text("old")

        src = tmp_path / "model.engine"
        src.write_text("new")

        cache.store(src, overwrite=False, clear_old=True)
        # Source should be removed even though overwrite is False
        assert not src.exists()
        # Cache should keep old version
        assert (cache_dir / "model.engine").read_text() == "old"

    def test_store_file_uses_original_name_when_cache_filename_is_none(
        self, tmp_path, monkeypatch: object
    ) -> None:
        """store_file without cache_filename uses the original filename."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        src = tmp_path / "original.engine"
        src.write_text("data")

        result = cache.store_file(src)
        assert result.name == "original.engine"
        assert result.exists()

    def test_store_clear_old_true_with_overwrite_true(self, tmp_path, monkeypatch: object) -> None:
        """Store with clear_old=True and overwrite=True removes source and writes new."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        # Pre-populate the cache
        (cache_dir / "model.engine").write_text("old")

        src = tmp_path / "model.engine"
        src.write_text("new")

        cache.store(src, overwrite=True, clear_old=True)
        assert not src.exists()
        assert (cache_dir / "model.engine").read_text() == "new"


@pytest.mark.cpu
class TestRemove:
    """Tests for remove and remove_file."""

    def test_remove_existing(self, tmp_path, monkeypatch) -> None:
        """remove() deletes an existing engine from cache."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        (cache_dir / "model.engine").write_text("data")
        cache.remove("model")
        assert not (cache_dir / "model.engine").exists()

    def test_remove_nonexistent_raises(self, tmp_path, monkeypatch) -> None:
        """remove() on a non-existent file raises FileNotFoundError."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        with pytest.raises(FileNotFoundError):
            cache.remove("nonexistent")

    def test_remove_file_existing(self, tmp_path, monkeypatch) -> None:
        """remove_file() deletes a file with the given extension."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        (cache_dir / "data.cache").write_text("cache data")
        cache.remove_file("data", "cache")
        assert not (cache_dir / "data.cache").exists()

    def test_remove_file_nonexistent_raises(self, tmp_path, monkeypatch) -> None:
        """remove_file() on a non-existent file raises FileNotFoundError."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        with pytest.raises(FileNotFoundError):
            cache.remove_file("nonexistent", "cache")

    def test_remove_file_with_valid_extension_in_name(self, tmp_path, monkeypatch: object) -> None:
        """remove_file() with valid extension in the name itself."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        (cache_dir / "model.engine").write_text("data")
        cache.remove_file("model.engine", "engine")
        assert not (cache_dir / "model.engine").exists()


@pytest.mark.cpu
class TestTimingCache:
    """Tests for query_timing_cache, store_timing_cache, save_timing_cache_to_global."""

    def test_query_timing_cache_missing(self, tmp_path, monkeypatch) -> None:
        """query_timing_cache() returns (False, path) when no global cache exists."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        exists, path = cache.query_timing_cache()
        assert exists is False
        assert path.name == "global.cache"

    def test_store_and_query_timing_cache(self, tmp_path, monkeypatch) -> None:
        """store_timing_cache followed by query_timing_cache roundtrip works."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        src = tmp_path / "timing.cache"
        src.write_bytes(b"timing data bytes")

        cache.store_timing_cache(src)
        exists, path = cache.query_timing_cache()
        assert exists is True
        assert path.read_bytes() == b"timing data bytes"

    def test_store_timing_cache_overwrite(self, tmp_path, monkeypatch) -> None:
        """store_timing_cache with overwrite=True replaces existing."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        src1 = tmp_path / "v1.cache"
        src1.write_bytes(b"v1")
        cache.store_timing_cache(src1)

        src2 = tmp_path / "v2.cache"
        src2.write_bytes(b"v2")
        cache.store_timing_cache(src2, overwrite=True)

        _, path = cache.query_timing_cache()
        assert path.read_bytes() == b"v2"

    def test_store_timing_cache_clear_old(self, tmp_path, monkeypatch) -> None:
        """store_timing_cache with clear_old=True removes original file."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        src = tmp_path / "timing.cache"
        src.write_bytes(b"data")
        cache.store_timing_cache(src, clear_old=True)
        assert not src.exists()

    def test_save_timing_cache_to_global(self, tmp_path, monkeypatch) -> None:
        """save_timing_cache_to_global serializes and stores the cache object."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        class MockTimingCache:
            def serialize(self) -> bytes:
                return b"serialized cache data"

        result = cache.save_timing_cache_to_global(MockTimingCache())
        assert result.exists()
        assert result.name == "global.cache"

        exists, path = cache.query_timing_cache()
        assert exists is True
        assert path.read_bytes() == b"serialized cache data"

    def test_save_timing_cache_to_global_overwrite(self, tmp_path, monkeypatch: object) -> None:
        """save_timing_cache_to_global overwrites by default (overwrite=True)."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        class MockTimingCache:
            def __init__(self, data) -> None:
                self._data = data

            def serialize(self):
                return self._data

        cache.save_timing_cache_to_global(MockTimingCache(b"first"))
        cache.save_timing_cache_to_global(MockTimingCache(b"second"))

        _, path = cache.query_timing_cache()
        assert path.read_bytes() == b"second"

    def test_save_timing_cache_to_global_no_overwrite(self, tmp_path, monkeypatch: object) -> None:
        """save_timing_cache_to_global with overwrite=False keeps original."""
        from trtutils.core import cache

        cache_dir = tmp_path / "_engine_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(cache, "get_cache_dir", lambda: cache_dir)

        class MockTimingCache:
            def __init__(self, data) -> None:
                self._data = data

            def serialize(self):
                return self._data

        cache.save_timing_cache_to_global(MockTimingCache(b"first"))
        cache.save_timing_cache_to_global(MockTimingCache(b"second"), overwrite=False)

        _, path = cache.query_timing_cache()
        assert path.read_bytes() == b"first"
