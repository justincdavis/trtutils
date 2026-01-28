# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
Tools for managing the trtutils TensorRT engine cache.

Useful for quickly recalling pre-compiled TRT engines,
without having to implement your own caching mechanisms.
Used in the TRTPreprocessor to manage compiled engines
for different image sizes.

Functions
---------
:func:`get_cache_dir`
    Gets the cache directory inside of the trtutils install.
:func:`clear`
    Clears the cache directory.
:func:`query`
    Queries the cache to see if an engine with that name already exists.
:func:`store`
    Stores a compiled TensorRT engine in the cache.
:func:`remove`
    Removes an engine file from the cache.
:func:`query_file`
    Queries the cache for a file with a specific extension.
:func:`store_file`
    Stores a file in the cache with a specific name.
:func:`remove_file`
    Removes a file from the cache.
:func:`query_timing_cache`
    Queries the cache for the global timing cache.
:func:`store_timing_cache`
    Stores the global timing cache in the cache directory.
:func:`save_timing_cache_to_global`
    Saves a TensorRT timing cache object directly to the global timing cache.

"""
# POTENTIAL CHANGE: Update to use platformdirs behind the scenes
# https://github.com/tox-dev/platformdirs/tree/main?tab=readme-ov-file

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Protocol

from trtutils._log import LOG


class _TimingCache(Protocol):
    def serialize(self) -> bytes: ...


# Known valid cache file extensions
_VALID_EXTENSIONS = {"engine", "onnx", "cache"}


def _get_cache_file_path(filename: str, extension: str) -> Path:
    """Get the full path for a cache file, handling extension logic."""
    has_valid_extension = False
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1]
        has_valid_extension = ext in _VALID_EXTENSIONS

    if has_valid_extension:
        return get_cache_dir() / filename
    return get_cache_dir() / f"{filename}.{extension}"


def _delete_folder(directory: Path) -> None:
    for item in directory.iterdir():
        if item.is_dir():
            _delete_folder(item)
        else:
            item.unlink()
    directory.rmdir()


def get_cache_dir() -> Path:
    """
    Get the location of the trtutils engine cache directory.

    Returns
    -------
    Path
        The trtutils engine cache directory Path

    """
    file_path = Path(__file__)
    return file_path.parent / "_engine_cache"


def clear(*, no_warn: bool | None = None) -> None:
    """
    Use to clear the cache folder for the trtutils engines.

    Parameters
    ----------
    no_warn : bool, optional
        Whether or not to issue a warning that the cache directory
        is being cleared.

    """
    if not no_warn:
        LOG.warning("Engine cache is being cleared")
    cache_dir = get_cache_dir()
    _delete_folder(cache_dir)
    cache_dir.mkdir()


def query_file(filename: str, extension: str = "engine") -> tuple[bool, Path]:
    """
    Check if a file with the given name and extension is present in the cache.

    Parameters
    ----------
    filename : str
        The filename to check for. Can be with or without extension.
        If extension is provided in filename, it will be used.
    extension : str, optional
        The file extension to use (without the dot).
        By default, "engine".

    Returns
    -------
    tuple[bool, Path]
        Whether or not the file exists and its Path (whether or not it exists)

    """
    file_path = _get_cache_file_path(filename, extension)
    return file_path.exists(), file_path


def query(filename: str) -> tuple[bool, Path]:
    """
    Check if the engine filename is present in the cache.

    Parameters
    ----------
    filename : str
        The filename to check for without a suffix.

    Returns
    -------
    tuple[bool, Path]
        Whether or not the file exists and its Path (whether or not it exists)

    """
    return query_file(filename, extension="engine")


def store_file(
    filepath: Path,
    cache_filename: str | None = None,
    *,
    overwrite: bool = False,
    clear_old: bool = False,
) -> Path:
    """
    Store a file in the trtutils cache.

    Parameters
    ----------
    filepath : Path
        The path to the file to store in the cache.
    cache_filename : str, optional
        The name to use in the cache. If None, uses the original filename.
        By default, None.
    overwrite : bool, optional
        Whether or not to overwrite an existing file with the same name.
        By default False, will keep the older version.
    clear_old : bool, optional
        Whether or not to automatically clear the original file.
        By default, False, will not remove original (old) file.

    Returns
    -------
    Path
        The new path of the file in the cache.

    """
    if cache_filename is None:
        cache_filename = filepath.name

    new_file_path = get_cache_dir() / cache_filename
    exists = new_file_path.exists()

    if not overwrite and exists:
        if clear_old:
            filepath.unlink()
        return new_file_path

    # otherwise we write the file
    shutil.copy(filepath, new_file_path)
    if clear_old:
        filepath.unlink()
    return new_file_path


def store(filepath: Path, *, overwrite: bool = False, clear_old: bool = False) -> Path:
    """
    Store an engine file in the trtutils engine cache.

    Parameters
    ----------
    filepath : Path
        The path to the engine file to store in the cache.
    overwrite : bool, optional
        Whether or not to overwrite an existing file with the same name.
        By default False, will keep the older version.
    clear_old : bool, optional
        Whether or not to automatically clear the original file.
        By default, False, will not remove original (old) file.

    Returns
    -------
    Path
        The new path of the file in the cache.

    """
    return store_file(filepath, overwrite=overwrite, clear_old=clear_old)


def remove_file(filename: str, extension: str = "engine") -> None:
    """
    Remove a file from the cache.

    Parameters
    ----------
    filename : str
        The filename to remove from the cache. Can be with or without extension.
        If extension is provided in filename, it will be used.
    extension : str, optional
        The file extension to use (without the dot).
        By default, "engine".

    Raises
    ------
    FileNotFoundError
        If the file does not exist in the cache.

    """
    file_path = _get_cache_file_path(filename, extension)
    if not file_path.exists():
        err_msg = f"File {file_path} does not exist in the cache"
        raise FileNotFoundError(err_msg)
    file_path.unlink()


def remove(filename: str) -> None:
    """
    Remove an engine file from the cache.

    Parameters
    ----------
    filename : str
        The filename to remove from the cache.

    """
    remove_file(filename, extension="engine")


def query_timing_cache() -> tuple[bool, Path]:
    """
    Query the cache for the global timing cache.

    Returns
    -------
    tuple[bool, Path]
        Whether or not the global timing cache exists and its Path.

    """
    return query_file("global", extension="cache")


def store_timing_cache(filepath: Path, *, overwrite: bool = False, clear_old: bool = False) -> Path:
    """
    Store the global timing cache in the cache directory.

    Parameters
    ----------
    filepath : Path
        The path to the timing cache file to store.
    overwrite : bool, optional
        Whether or not to overwrite an existing global timing cache.
        By default False, will keep the older version.
    clear_old : bool, optional
        Whether or not to automatically clear the original file.
        By default, False, will not remove original (old) file.

    Returns
    -------
    Path
        The path of the global timing cache in the cache directory.

    """
    return store_file(
        filepath, cache_filename="global.cache", overwrite=overwrite, clear_old=clear_old
    )


def save_timing_cache_to_global(timing_cache_obj: _TimingCache, *, overwrite: bool = True) -> Path:
    """
    Save a TensorRT timing cache object to the global timing cache.

    Parameters
    ----------
    timing_cache_obj
        The TensorRT timing cache object (from config.get_timing_cache()).
    overwrite : bool, optional
        Whether or not to overwrite an existing global timing cache.
        By default True.

    Returns
    -------
    Path
        The path of the global timing cache in the cache directory.

    """
    serialized_cache = memoryview(timing_cache_obj.serialize())

    # create a temporary file to store the serialized cache
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cache") as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_path.write_bytes(serialized_cache)
        return store_timing_cache(tmp_path, overwrite=overwrite, clear_old=True)
