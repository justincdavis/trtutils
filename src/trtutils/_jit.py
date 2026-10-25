# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import functools
from typing import TYPE_CHECKING, Generic

from typing_extensions import TypeVar

from ._flags import FLAGS
from ._log import LOG

_R = TypeVar("_R")

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from typing_extensions import Self

with contextlib.suppress(ImportError):
    from numba import jit as _jit

    FLAGS.FOUND_NUMBA = True


class _JitFunction(Generic[_R]):
    """A function that runs ``jitted`` while ``FLAGS.JIT`` is on and ``py_func`` otherwise."""

    def __init__(self: Self, py_func: Callable[..., _R], jitted: Callable[..., _R]) -> None:
        functools.update_wrapper(self, py_func)
        self.py_func = py_func
        self.jitted = jitted

    def __call__(self: Self, *args: object, **kwargs: object) -> _R:
        if FLAGS.JIT:
            return self.jitted(*args, **kwargs)
        return self.py_func(*args, **kwargs)


def register_jit(
    *,
    fastmath: bool = False,
    parallel: bool = False,
    nogil: bool = False,
    cache: bool = False,
    inline: str = "never",
) -> Callable[[Callable[..., _R]], _JitFunction[_R]]:
    """
    Decorate a function to run Numba-compiled while JIT is enabled.

    The returned function checks ``FLAGS.JIT`` on every call, so :func:`enable_jit` and
    :func:`disable_jit` take effect at any time. Compilation happens on the first call
    with JIT enabled. Without Numba the function always runs as plain Python.

    Parameters
    ----------
    fastmath : bool, optional
        If True, enable fastmath during jit.
        Default is False.
    parallel : bool, optional
        If True, enable parallel jit.
        Default is False.
    nogil : bool, optional
        If True, disable the GIL when running jit compiled functions.
        Default is False.
    cache : bool, optional
        If True, cache jit compiled functions to disk.
        Default is False.
    inline : str, optional
        Whether or not to inline functions at the Numba IR level.
        Default is 'never'.
        Options are: ['never', 'always']

    Returns
    -------
    Callable[[Callable[_P, _R]], _JitFunction[_R]]
        A decorator returning the JIT-switchable function; ``py_func`` and ``jitted``
        hold the plain and Numba versions (the same function without Numba).

    Examples
    --------
    >>> @register_jit(fastmath=True, parallel=True)
    ... def my_func(x):
    ...     return x * x

    """

    def decorator(func: Callable[..., _R]) -> _JitFunction[_R]:
        if not FLAGS.FOUND_NUMBA:
            return _JitFunction(func, func)
        jitted = _jit(
            func,
            nopython=True,
            fastmath=fastmath,
            parallel=parallel,
            nogil=nogil,
            cache=cache,
            inline=inline,
        )
        return _JitFunction(func, jitted)

    return decorator


def enable_jit() -> None:
    """Enable just-in-time compilation using Numba for some functions."""
    FLAGS.JIT = True
    LOG.info("ENABLED JIT")

    if not FLAGS.FOUND_NUMBA and not FLAGS.WARNED_NUMBA_NOT_FOUND:
        LOG.warning("JIT has been enabled, but Numba could not be found.")
        FLAGS.WARNED_NUMBA_NOT_FOUND = True


def disable_jit() -> None:
    """Disable JIT compilation."""
    FLAGS.JIT = False
    LOG.info("DISABLED JIT")


class _JIT:
    def __init__(self: Self) -> None:
        self._was_on = False

    def __enter__(self: Self) -> Self:
        self._was_on = FLAGS.JIT
        enable_jit()
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._was_on:
            disable_jit()


JIT = _JIT()
