"""Tests for src/trtutils/core/_lock.py -- thread safety primitives."""

from __future__ import annotations

import threading
import time

import pytest


@pytest.mark.cpu
class TestMemAllocLock:
    """Tests for MEM_ALLOC_LOCK."""

    def test_acquire_release(self) -> None:
        """MEM_ALLOC_LOCK can be acquired and released."""
        from trtutils.core._lock import MEM_ALLOC_LOCK

        acquired = MEM_ALLOC_LOCK.acquire(timeout=1)
        assert acquired
        MEM_ALLOC_LOCK.release()

    def test_context_manager(self) -> None:
        """MEM_ALLOC_LOCK can be used as a context manager."""
        from trtutils.core._lock import MEM_ALLOC_LOCK

        with MEM_ALLOC_LOCK:
            pass  # Should not raise

    def test_is_threading_lock(self) -> None:
        """MEM_ALLOC_LOCK should be a threading.Lock."""
        from trtutils.core._lock import MEM_ALLOC_LOCK

        assert isinstance(MEM_ALLOC_LOCK, type(threading.Lock()))

    def test_thread_safety(self) -> None:
        """Two threads cannot hold MEM_ALLOC_LOCK simultaneously."""
        from trtutils.core._lock import MEM_ALLOC_LOCK

        results = []

        def worker(thread_id) -> None:
            with MEM_ALLOC_LOCK:
                results.append(f"{thread_id}_start")
                time.sleep(0.05)
                results.append(f"{thread_id}_end")

        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Verify no interleaving: pattern should be Xs, Xe, Ys, Ye
        assert results[0].endswith("_start")
        assert results[1].endswith("_end")
        assert results[0][:2] == results[1][:2]  # Same thread for first pair
        assert results[2].endswith("_start")
        assert results[3].endswith("_end")
        assert results[2][:2] == results[3][:2]  # Same thread for second pair


@pytest.mark.cpu
class TestNvrtcLock:
    """Tests for NVRTC_LOCK."""

    def test_acquire_release(self) -> None:
        """NVRTC_LOCK can be acquired and released."""
        from trtutils.core._lock import NVRTC_LOCK

        acquired = NVRTC_LOCK.acquire(timeout=1)
        assert acquired
        NVRTC_LOCK.release()

    def test_context_manager(self) -> None:
        """NVRTC_LOCK can be used as a context manager."""
        from trtutils.core._lock import NVRTC_LOCK

        with NVRTC_LOCK:
            pass  # Should not raise

    def test_is_threading_lock(self) -> None:
        """NVRTC_LOCK should be a threading.Lock."""
        from trtutils.core._lock import NVRTC_LOCK

        assert isinstance(NVRTC_LOCK, type(threading.Lock()))

    def test_thread_safety(self) -> None:
        """Two threads cannot hold NVRTC_LOCK simultaneously."""
        from trtutils.core._lock import NVRTC_LOCK

        results = []

        def worker(thread_id) -> None:
            with NVRTC_LOCK:
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
        """Acquiring MEM_ALLOC_LOCK does not block NVRTC_LOCK."""
        from trtutils.core._lock import MEM_ALLOC_LOCK, NVRTC_LOCK

        with MEM_ALLOC_LOCK:
            # Should be able to acquire NVRTC_LOCK while holding MEM_ALLOC_LOCK
            acquired = NVRTC_LOCK.acquire(timeout=1)
            assert acquired
            NVRTC_LOCK.release()

    def test_locks_are_distinct_objects(self) -> None:
        """MEM_ALLOC_LOCK and NVRTC_LOCK are different lock instances."""
        from trtutils.core._lock import MEM_ALLOC_LOCK, NVRTC_LOCK

        assert MEM_ALLOC_LOCK is not NVRTC_LOCK
