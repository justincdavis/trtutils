# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/core/_lock.py -- thread locks."""

from __future__ import annotations

import threading
import time

import pytest

from trtutils.core._lock import MEM_ALLOC_LOCK, NVRTC_LOCK

LOCKS = [
    pytest.param(MEM_ALLOC_LOCK, id="MEM_ALLOC_LOCK"),
    pytest.param(NVRTC_LOCK, id="NVRTC_LOCK"),
]


@pytest.mark.cpu
@pytest.mark.parametrize("lock", LOCKS)
def test_lock_acquire_release(lock: threading.Lock) -> None:
    """Lock can be acquired and released."""
    acquired = lock.acquire(timeout=1)
    assert acquired
    lock.release()


@pytest.mark.cpu
@pytest.mark.parametrize("lock", LOCKS)
def test_lock_context_manager(lock: threading.Lock) -> None:
    """Lock works as a context manager."""
    with lock:
        pass


@pytest.mark.cpu
@pytest.mark.parametrize("lock", LOCKS)
def test_lock_thread_safety(lock: threading.Lock) -> None:
    """Lock serializes access across threads."""
    results: list[str] = []

    def worker(thread_id: str) -> None:
        with lock:
            results.append(f"{thread_id}_start")
            time.sleep(0.05)
            results.append(f"{thread_id}_end")

    t1 = threading.Thread(target=worker, args=("t1",))
    t2 = threading.Thread(target=worker, args=("t2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results[0].endswith("_start")
    assert results[1].endswith("_end")
    assert results[0][:2] == results[1][:2]
    assert results[2].endswith("_start")
    assert results[3].endswith("_end")
    assert results[2][:2] == results[3][:2]


@pytest.mark.cpu
def test_locks_are_independent() -> None:
    """MEM_ALLOC_LOCK and NVRTC_LOCK are distinct and can be held simultaneously."""
    assert MEM_ALLOC_LOCK is not NVRTC_LOCK
    with MEM_ALLOC_LOCK:
        acquired = NVRTC_LOCK.acquire(timeout=1)
        assert acquired
        NVRTC_LOCK.release()
