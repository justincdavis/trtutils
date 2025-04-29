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
:func:`clear_cache`
    Clears the cache directory.
:func:`query_cache`
    Queries the cache to see if an engine with that name already exists.
:func:`store_in_cache`
    Stores a compiled TensorRT engine in the cache.

"""

from __future__ import annotations

import shutil
from pathlib import Path

from trtutils._log import LOG


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


def clear_cache(*, no_warn: bool | None = None) -> None:
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


def query_cache(filename: str) -> tuple[bool, Path]:
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
    file_path = get_cache_dir() / f"{filename}.engine"
    success = file_path.exists()
    return success, file_path


def store_in_cache(
    filepath: Path, *, overwrite: bool = False, clear_old: bool = False
) -> Path:
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
    exists, existing_path = query_cache(filepath.stem)
    if not overwrite and exists:
        if clear_old:
            filepath.unlink()
        return existing_path

    # otherwise we write the file
    new_file_path = get_cache_dir() / filepath.name
    shutil.copy(filepath, new_file_path)
    if clear_old:
        filepath.unlink()
    return new_file_path
