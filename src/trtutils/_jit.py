# Copyright (c) 2025-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import contextlib
import functools
from typing import TYPE_CHECKING

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


def register_jit(
    *,
    fastmath: bool = False,
    parallel: bool = False,
    nogil: bool = False,
    cache: bool = False,
    inline: str = "never",
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
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
    Callable[[Callable[_P, _R]], Callable[_P, _R]]
        A decorator returning the JIT-switchable function.

    Examples
    --------
    >>> @register_jit(fastmath=True, parallel=True)
    ... def my_func(x):
    ...     return x * x

    """

    def decorator(func: Callable[..., _R]) -> Callable[..., _R]:
        if not FLAGS.FOUND_NUMBA:
            return func
        jitted = _jit(
            func,
            nopython=True,
            fastmath=fastmath,
            parallel=parallel,
            nogil=nogil,
            cache=cache,
            inline=inline,
        )

        @functools.wraps(func)
        def dispatch(*args: object, **kwargs: object) -> _R:
            if FLAGS.JIT:
                return jitted(*args, **kwargs)
            return func(*args, **kwargs)

        dispatch.jitted = jitted  # ty: ignore[unresolved-attribute]
        return dispatch

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
        pass

    def __enter__(self: Self) -> Self:
        enable_jit()
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        disable_jit()


JIT = _JIT()
