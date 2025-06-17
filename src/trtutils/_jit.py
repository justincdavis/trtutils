# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import ParamSpec, TypeVar

from ._flags import FLAGS
from ._log import LOG

_P = ParamSpec("_P")
_R = TypeVar("_R")

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from typing_extensions import Self

try:
    from numba import jit as _jit  # type: ignore[import-untyped]

    FLAGS.FOUND_NUMBA = True
except ImportError:

    def _jit(
        func: Callable[_P, _R],
        *,
        nopython: bool,  # noqa: ARG001
        fastmath: bool,  # noqa: ARG001
        parallel: bool,  # noqa: ARG001
        nogil: bool,  # noqa: ARG001
        cache: bool,  # noqa: ARG001
        inline: str = "never",  # noqa: ARG001
    ) -> Callable[_P, _R]:
        LOG.debug(f"Using mock JIT on {func.__name__}")
        return func


_JIT_FUNCS: list[Callable] = []


def jit(
    func: Callable[_P, _R],
    *,
    fastmath: bool = False,
    parallel: bool = False,
    nogil: bool = False,
    cache: bool = False,
    inline: str = "never",
) -> Callable[_P, _R]:
    """
    Optionally JIT compile a function based on the flags for cv2ext.

    Parameters
    ----------
    func : Callable[_P, _R]
        The function to jit compile
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
    Callable[_P, _R]
        The JIT compiled or untouched function.

    """
    funcname = func.__name__
    if FLAGS.JIT:
        LOG.debug(f"Marking: {funcname} for JIT compilation")
        func = _jit(
            func,
            nopython=True,
            fastmath=fastmath,
            parallel=parallel,
            nogil=nogil,
            cache=cache,
            inline=inline,
        )
    LOG.debug(f"Resolved: {funcname} -> {type(func)}")
    return func


def register_jit(
    *,
    fastmath: bool = False,
    parallel: bool = False,
    nogil: bool = False,
    cache: bool = False,
    inline: str = "never",
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Register a function to be re-imported whenever JIT status changes.

    Parameters
    ----------
    func : Callable[_P, _R], optional
        The function to optionally JIT compile. If None, the decorator
        returns a partially applied function.
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
        The registered and optionally JIT-compiled function.

    Examples
    --------
    >>> @register_jit(fastmath=True, parallel=True)
    ... def my_func(x):
    ...     return x * x

    """
    LOG.debug(
        f"register_jit: fastmath={fastmath}, parallel={parallel}, nogil={nogil}, cache={cache}, inline={inline}",
    )

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        LOG.debug(f"Registering func: {func.__name__} for potential JIT")
        _JIT_FUNCS.append(func)
        return jit(
            func,
            fastmath=fastmath,
            parallel=parallel,
            nogil=nogil,
            cache=cache,
            inline=inline,
        )

    return decorator


def _reset_funcs() -> None:
    # re-compile if needed
    for func in _JIT_FUNCS:
        globals()[func.__name__] = jit(func)


def enable_jit() -> None:
    """Enable just-in-time compilation using Numba for some functions."""
    FLAGS.JIT = True
    LOG.info("ENABLED JIT")

    if not FLAGS.FOUND_NUMBA and not FLAGS.WARNED_NUMBA_NOT_FOUND:
        LOG.warning("JIT has been enabled, but Numba could not be found.")
        FLAGS.WARNED_NUMBA_NOT_FOUND = True

    _reset_funcs()


def disable_jit() -> None:
    """Disable JIT compilation."""
    FLAGS.JIT = False
    LOG.info("DISABLED JIT")

    _reset_funcs()


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
