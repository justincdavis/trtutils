# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for TRTEngine multi-threading -- thread safety, coexistence, CUDA graphs."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from trtutils import TRTEngine
from trtutils._flags import FLAGS

NUM_ENGINES = 4
NUM_ITERS = 1_000


# ============================================================================
# Single engine in a child thread
# ============================================================================


class TestSingleEngineThread:
    """Test creating and running an engine entirely inside a child thread."""

    def test_engine_run_in_thread(self, engine_path) -> None:
        """Create a TRTEngine in a child thread, run mock_execute, verify output."""
        result: list[bool] = [False]

        def run(res: list[bool]) -> None:
            engine = TRTEngine(engine_path, warmup=False)
            outputs = engine.mock_execute()
            assert outputs is not None
            res[0] = True
            del engine

        thread = threading.Thread(target=run, args=(result,), daemon=True)
        thread.start()
        thread.join()

        assert result[0], "Thread did not complete successfully"


# ============================================================================
# Multiple engines in the same thread
# ============================================================================


class TestMultipleEnginesCoexistence:
    """Test multiple TRTEngine instances coexisting in a single thread."""

    def test_multiple_engines_same_thread(self, engine_path) -> None:
        """Create NUM_ENGINES engines, run mock_execute on each, verify outputs."""
        engines = [TRTEngine(engine_path, warmup=False) for _ in range(NUM_ENGINES)]

        outputs = [engine.mock_execute() for engine in engines]

        for o in outputs:
            assert o is not None
            assert isinstance(o, list)
            for arr in o:
                assert isinstance(arr, np.ndarray)

        for engine in engines:
            del engine


# ============================================================================
# Multiple engines in separate threads
# ============================================================================


class TestMultipleEnginesInThreads:
    """Test engines running concurrently in separate threads."""

    def test_engines_in_separate_threads(self, engine_path) -> None:
        """NUM_ENGINES threads each create an engine and run NUM_ITERS iterations."""
        result: list[int] = [0] * NUM_ENGINES

        def run(threadid: int, res: list[int], iters: int) -> None:
            engine = TRTEngine(engine_path, warmup=False)
            successes = 0
            for _ in range(iters):
                outputs = engine.mock_execute()
                if outputs is not None:
                    successes += 1
            res[threadid] = successes
            del engine

        threads = [
            threading.Thread(
                target=run,
                args=(tid, result, NUM_ITERS),
                daemon=True,
            )
            for tid in range(NUM_ENGINES)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r in result:
            assert r == NUM_ITERS

    def test_threads_produce_valid_outputs(self, engine_path) -> None:
        """Lighter variant (10 iters) verifying output structure from each thread."""
        num_iters = 10
        results: dict[int, list[list[np.ndarray]]] = {}

        def run(
            threadid: int,
            res: dict[int, list[list[np.ndarray]]],
            iters: int,
        ) -> None:
            engine = TRTEngine(engine_path, warmup=False)
            collected: list[list[np.ndarray]] = []
            for _ in range(iters):
                outputs = engine.mock_execute()
                assert outputs is not None
                collected.append(outputs)
            res[threadid] = collected
            del engine

        threads = [
            threading.Thread(
                target=run,
                args=(tid, results, num_iters),
                daemon=True,
            )
            for tid in range(NUM_ENGINES)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == NUM_ENGINES
        for tid in range(NUM_ENGINES):
            assert len(results[tid]) == num_iters
            for output_list in results[tid]:
                assert isinstance(output_list, list)
                for arr in output_list:
                    assert isinstance(arr, np.ndarray)


# ============================================================================
# CUDA graph thread safety
# ============================================================================


@pytest.mark.cuda_graph
class TestCUDAGraphThreadSafety:
    """Test CUDA graph independence across threads."""

    def test_independent_graphs_in_threads(self, engine_path) -> None:
        """Each thread creates engine with cuda_graph=True, replays NUM_ITERS times."""
        if not FLAGS.EXEC_ASYNC_V3:
            pytest.skip("async_v3 required for CUDA graph")

        results: dict[int, dict] = {}

        def run(threadid: int, res: dict[int, dict], iters: int) -> None:
            engine = TRTEngine(
                engine_path,
                warmup=True,
                backend="async_v3",
                cuda_graph=True,
            )

            cuda_graph = engine._cuda_graph
            if cuda_graph is None:
                res[threadid] = {"success": False, "graph_exec_id": None}
                del engine
                return

            graph_exec_id = id(cuda_graph._graph_exec)

            for _ in range(iters):
                engine.mock_execute()
                assert id(cuda_graph._graph_exec) == graph_exec_id

            res[threadid] = {
                "graph_exec_id": graph_exec_id,
                "success": True,
            }
            del engine

        threads = [
            threading.Thread(
                target=run,
                args=(tid, results, NUM_ITERS),
                daemon=True,
            )
            for tid in range(NUM_ENGINES)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == NUM_ENGINES
        for tid in range(NUM_ENGINES):
            assert results[tid]["success"] is True

        graph_ids = [results[i]["graph_exec_id"] for i in range(NUM_ENGINES)]
        assert len(set(graph_ids)) == NUM_ENGINES, "Threads should have independent CUDA graphs"

    def test_lazy_graph_capture_in_thread(self, engine_path) -> None:
        """A child thread can lazily capture a CUDA graph on first execute."""
        if not FLAGS.EXEC_ASYNC_V3:
            pytest.skip("async_v3 required for CUDA graph")

        result: dict[str, object] = {}

        def run(res: dict[str, object]) -> None:
            try:
                engine = TRTEngine(
                    engine_path,
                    warmup=False,
                    backend="async_v3",
                    cuda_graph=True,
                )

                cuda_graph = engine._cuda_graph
                if cuda_graph is None:
                    res["success"] = False
                    res["error"] = "no graph"
                    del engine
                    return

                res["pre_captured"] = cuda_graph.is_captured
                engine.mock_execute()
                res["post_captured"] = cuda_graph.is_captured
                res["success"] = True
                del engine
            except Exception as exc:
                res["success"] = False
                res["error"] = str(exc)

        thread = threading.Thread(target=run, args=(result,), daemon=True)
        thread.start()
        thread.join()

        assert result.get("success") is True, f"Thread failed: {result.get('error')}"
        assert result["pre_captured"] is False, "Graph should not be captured before execute"
        assert result["post_captured"] is True, "Graph should be captured after execute"
