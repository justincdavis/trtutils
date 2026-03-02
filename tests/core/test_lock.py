# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
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
class TestLockBehavior:
    """Tests for MEM_ALLOC_LOCK and NVRTC_LOCK behavior."""

    def test_acquire_release(self, lock) -> None:
        acquired = lock.acquire(timeout=1)
        assert acquired
        lock.release()

    def test_context_manager(self, lock) -> None:
        with lock:
            pass

    def test_is_threading_lock(self, lock) -> None:
        assert isinstance(lock, type(threading.Lock()))

    def test_thread_safety(self, lock) -> None:
        results = []

        def worker(thread_id) -> None:
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
class TestLocksIndependence:
    """Tests that the two locks are independent."""

    def test_locks_are_independent(self) -> None:
        with MEM_ALLOC_LOCK:
            acquired = NVRTC_LOCK.acquire(timeout=1)
            assert acquired
            NVRTC_LOCK.release()

    def test_locks_are_distinct_objects(self) -> None:
        assert MEM_ALLOC_LOCK is not NVRTC_LOCK
