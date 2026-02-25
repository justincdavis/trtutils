# Phase 2: TRTEngine Tests (100% Branch Coverage)

**Branch:** `test-overhaul/phase-2-engine`
**Goal:** Achieve 100% branch coverage on `src/trtutils/_engine.py` (TRTEngine) and `src/trtutils/core/_interface.py` (TRTEngineInterface).

---

## Prerequisites
- Phase 0 (infrastructure)
- Phase 1 (core tests provide fixtures: `simple_engine_path`, `cuda_stream`)

## Deliverables
- [ ] `tests/engine/conftest.py` — Engine-specific fixtures
- [ ] `tests/engine/test_engine.py` — TRTEngine init, backends, properties
- [ ] `tests/engine/test_execute.py` — execute() with all branch paths
- [ ] `tests/engine/test_direct_exec.py` — direct_exec() paths
- [ ] `tests/engine/test_raw_exec.py` — raw_exec() paths
- [ ] `tests/engine/test_graph_exec.py` — CUDA graph lifecycle
- [ ] `tests/engine/test_mock_execute.py` — mock_execute(), warmup(), get_random_input()
- [ ] `tests/engine/test_interface.py` — Abstract interface contract
- [ ] 100% branch coverage on both files verified

---

## Engine Conftest (tests/engine/conftest.py)

```python
"""Engine test fixtures — built engines, test data, memory mode configs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# tests/engine/conftest.py -> tests/engine -> tests -> project_root -> data/
ONNX_PATH = Path(__file__).parent.parent.parent / "data" / "simple.onnx"


@pytest.fixture(scope="session")
def engine_path(build_test_engine) -> Path:
    """Session-scoped built engine for general engine tests."""
    return build_test_engine(ONNX_PATH)


@pytest.fixture
def engine(engine_path):
    """Create a fresh TRTEngine instance per test."""
    from trtutils import TRTEngine
    eng = TRTEngine(engine_path, warmup=False)
    yield eng
    del eng


@pytest.fixture
def engine_with_warmup(engine_path):
    """TRTEngine with warmup enabled."""
    from trtutils import TRTEngine
    eng = TRTEngine(engine_path, warmup=True, warmup_iterations=2)
    yield eng
    del eng


@pytest.fixture
def random_input(engine):
    """Generate random input matching engine spec."""
    return engine.get_random_input()
```

---

## Detailed Test Plans

### tests/engine/test_engine.py — Initialization & Properties

**All tests: `@pytest.mark.gpu`**

#### Branch Coverage Targets in `__init__`:

| Branch | Condition | Test |
|--------|-----------|------|
| Backend validation | `backend not in _backends` | `test_invalid_backend_raises` |
| async_v3 selection | `FLAGS.EXEC_ASYNC_V3 and (backend == "async_v3" or "auto")` | `test_backend_auto`, `test_backend_async_v3` |
| async_v2 fallback | `not FLAGS.EXEC_ASYNC_V3 or backend == "async_v2"` | `test_backend_async_v2` |
| CUDA graph enable | `cuda_graph and self._async_v3` | `test_cuda_graph_enabled`, `test_cuda_graph_without_v3` |
| CUDA graph init | `self._cuda_graph_enabled` → create CUDAGraph | `test_cuda_graph_object_created` |
| async_v3 bindings | `self._async_v3` → set bindings | `test_async_v3_sets_bindings` |
| verbose flag | `verbose is not None` | `test_verbose_true`, `test_verbose_false`, `test_verbose_none` |
| warmup decision | `self._warmup` | `test_warmup_true`, `test_warmup_false` |

#### Test Cases

```
class TestEngineInit:
    test_init_default_args                 # Minimal init works
    test_init_with_warmup                  # warmup=True, warmup_iterations=5
    test_init_without_warmup               # warmup=False (default)
    test_invalid_backend_raises            # ValueError("Invalid backend")

    @parametrize("backend", ["auto", "async_v3", "async_v2"])
    test_backend_selection                 # Each backend accepted

    test_cuda_graph_enabled                # cuda_graph=True with async_v3
    test_cuda_graph_disabled_without_v3    # cuda_graph=True but async_v2 → disabled
    test_cuda_graph_false                  # cuda_graph=False → no graph object

    @parametrize("verbose", [True, False, None])
    test_verbose_flag                      # verbose stored correctly

    @parametrize("pagelocked", [True, False, None])
    test_pagelocked_mem                    # Memory mode selection

class TestEngineProperties:
    test_name                              # Returns string (filename stem)
    test_engine_object                     # Returns trt.ICudaEngine
    test_context_object                    # Returns trt.IExecutionContext
    test_stream_object                     # Returns cudart.cudaStream_t
    test_memsize_positive                  # Returns int > 0
    test_input_spec                        # Returns list of (shape, dtype) tuples
    test_input_shapes                      # Returns list of tuple[int, ...]
    test_input_dtypes                      # Returns list of np.dtype
    test_input_names                       # Returns list of str
    test_output_spec                       # list of (shape, dtype)
    test_output_shapes                     # list of tuple[int, ...]
    test_output_dtypes                     # list of np.dtype
    test_output_names                      # list of str
    test_batch_size                        # Returns int >= 1
    test_is_dynamic_batch                  # Returns bool
    test_input_bindings                    # Returns list of Binding
    test_output_bindings                   # Returns list of Binding
    test_pagelocked_mem_property           # Matches init parameter
    test_unified_mem_property              # Matches init parameter
    test_dla_core_property                 # None on desktop GPU

class TestEngineDestruction:
    test_del_no_error                      # Delete engine without error
    test_del_with_cuda_graph               # Graph invalidated on delete
```

---

### tests/engine/test_execute.py — execute() Method

**All tests: `@pytest.mark.gpu`**

#### Branch Coverage Targets (45+ branches in execute):

| Branch Group | Conditions | Tests |
|-------------|-----------|-------|
| Verbose resolution | `verbose is not None` → use, else `self._verbose` | 2 tests |
| Binding reset | `_using_engine_tensors is False` | 1 test |
| Input copy (3-way) | pagelocked+unified, pagelocked only, neither | 3 tests |
| Debug sync before exec | `debug is True` | 2 tests |
| CUDA graph path | graph enabled + captured, graph enabled + not captured, graph disabled | 3 tests |
| Graph capture during exec | First execute captures, second replays | 2 tests |
| Async v3 vs v2 | Backend-specific execute call | 2 tests |
| Debug sync after exec | `debug is True` | covered above |
| Output copy (3-way) | pagelocked+unified, pagelocked only, neither | 3 tests |
| Stream sync | `_capturing_graph is False` → sync | 1 test |
| Return copy | `no_copy is True` → return direct, else copy | 2 tests |

#### Test Cases

```
class TestExecuteBasic:
    test_execute_returns_outputs           # Returns list[np.ndarray]
    test_execute_output_shapes_match       # Output shapes match engine spec
    test_execute_output_dtypes_match       # Output dtypes match engine spec
    test_execute_deterministic             # Same input → same output (2 runs)

class TestExecuteFlags:
    @parametrize("no_copy", [True, False, None])
    test_execute_no_copy                   # no_copy=True returns refs, False returns copies

    @parametrize("verbose", [True, False, None])
    test_execute_verbose                   # No error with any verbose setting

    @parametrize("debug", [True, False, None])
    test_execute_debug                     # debug=True adds extra syncs

class TestExecuteMemoryModes:
    @parametrize("pagelocked,unified", [
        pytest.param(True, False, id="pagelocked"),
        pytest.param(False, False, id="default"),
        pytest.param(True, True, id="unified"),
    ])
    test_execute_memory_modes              # All 3 memory paths produce correct output

class TestExecuteBindingReset:
    test_binding_reset_after_direct_exec   # execute() after direct_exec() resets bindings

class TestExecuteNoCopyBehavior:
    test_no_copy_true_returns_same_buffer  # Returned arrays point to host allocations
    test_no_copy_false_returns_copy        # Returned arrays are independent copies
    test_no_copy_none_defaults_to_copy     # Default behavior is copy
```

---

### tests/engine/test_direct_exec.py — direct_exec() Method

**All tests: `@pytest.mark.gpu`**

#### Branch Coverage Targets:

| Branch | Condition | Test |
|--------|-----------|------|
| Verbose resolution | `verbose is not None` | parametrized |
| Warning logic | `no_warn is False/None` → log warning | 2 tests |
| async_v3 path | `self._async_v3` → set_tensor_address | 1 test |
| async_v2 path | `not self._async_v3` → execute_async_v2 | 1 test |
| set_pointers flag | `set_pointers=True` vs `False` | 2 tests |
| Debug sync | `debug is True` | 1 test |
| Output copy (3-way) | pagelocked+unified, pagelocked, neither | 3 tests |

#### Test Cases

```
class TestDirectExec:
    test_direct_exec_returns_outputs       # Returns list[np.ndarray]
    test_direct_exec_matches_execute       # Same result as execute() for same input

    @parametrize("set_pointers", [True, False])
    test_set_pointers_flag                 # Both paths work

    @parametrize("no_warn", [True, False, None])
    test_no_warn_flag                      # Warning suppression

    test_direct_exec_sets_using_engine_false  # Marks _using_engine_tensors=False
    test_verbose_logging                   # No error with verbose=True
    test_debug_synchronization             # No error with debug=True
```

---

### tests/engine/test_raw_exec.py — raw_exec() Method

**All tests: `@pytest.mark.gpu`**

#### Test Cases

```
class TestRawExec:
    test_raw_exec_returns_pointers         # Returns list[int]
    test_raw_exec_pointers_are_valid       # Pointers are non-zero ints
    test_raw_exec_no_sync                  # Does NOT sync by default (verify timing)

    @parametrize("set_pointers", [True, False])
    test_set_pointers_flag                 # Both paths work

    @parametrize("no_warn", [True, False, None])
    test_no_warn_flag                      # Warning suppression

    test_debug_synchronization             # debug=True adds sync
    test_verbose_logging                   # verbose=True works
```

---

### tests/engine/test_graph_exec.py — CUDA Graph Lifecycle

**All tests: `@pytest.mark.gpu`, `@pytest.mark.cuda_graph`**

#### Branch Coverage Targets in _capture_cuda_graph:

| Branch | Condition | Test |
|--------|-----------|------|
| Recursion guard | `_capturing_graph is True` → return | 1 test |
| Graph existence | `_cuda_graph is None` → RuntimeError | 1 test |
| Warmup need | `not self._warmup` → attempt warmup | 1 test |
| Warmup exception | RuntimeError during warmup → invalidate graph | 1 test |
| Post-capture check | `is_captured is False` → raise | 1 test |
| Finally block | Always reset `_capturing_graph` | covered |

#### Test Cases

```
class TestCUDAGraphCapture:
    test_first_execute_captures_graph      # cuda_graph=True → captured on first exec
    test_graph_exec_after_capture          # graph_exec() succeeds after capture
    test_graph_exec_without_capture_raises # RuntimeError if no graph captured
    test_graph_exec_deterministic          # Same output on graph replay
    test_graph_invalidate_and_recapture    # Invalidate → next execute recaptures

class TestCUDAGraphExecuteIntegration:
    test_execute_with_cuda_graph_enabled   # Full flow: init → execute (capture) → execute (replay)
    test_execute_no_copy_with_graph        # no_copy works with graph path
    test_graph_exec_no_default_sync        # graph_exec() does not sync by default
    test_graph_exec_debug_sync             # graph_exec(debug=True) syncs

class TestCUDAGraphEdgeCases:
    test_cuda_graph_disabled               # cuda_graph=False → no capture on execute
    test_cuda_graph_with_async_v2          # cuda_graph=True + async_v2 → disabled
    test_binding_change_invalidates_graph  # Changing bindings invalidates captured graph
    test_delete_engine_with_graph          # Cleanup doesn't crash
```

---

### tests/engine/test_mock_execute.py — Utilities

**All tests: `@pytest.mark.gpu`**

#### Branch Coverage Targets:

| Method | Branches |
|--------|----------|
| `get_random_input()` | new=True generates fresh; new=None/False uses cache; float vs non-float dtypes; verbose logging |
| `mock_execute()` | data=None generates random; data provided uses it; verbose resolution |
| `warmup()` | iterations loop; verbose flag |
| `__call__()` | Delegates to execute |

#### Test Cases

```
class TestGetRandomInput:
    test_returns_list_of_arrays            # Returns list[np.ndarray]
    test_shapes_match_engine               # Shapes match input_shapes
    test_dtypes_match_engine               # Dtypes match input_dtypes
    test_cached_by_default                 # Two calls return same arrays
    test_new_generates_fresh               # new=True returns different arrays
    test_verbose_logging                   # verbose=True logs generation info

class TestMockExecute:
    test_returns_outputs                   # Returns list[np.ndarray]
    test_with_no_data                      # data=None → uses random input
    test_with_provided_data                # data provided → uses it
    test_verbose_flag                      # verbose=True works

class TestWarmup:
    @parametrize("iterations", [1, 3, 5])
    test_warmup_iterations                 # Runs N mock executions
    test_warmup_verbose                    # verbose=True works

class TestCall:
    test_call_delegates_to_execute         # __call__ produces same output as execute
```

---

### tests/engine/test_interface.py — Abstract Interface Contract

**All tests: `@pytest.mark.gpu`**

Tests that validate the TRTEngineInterface abstract contract through the TRTEngine concrete implementation.

#### Branch Coverage Targets in __init__:

| Branch | Condition | Test |
|--------|-----------|------|
| pagelocked default | `None` → True | `test_pagelocked_default_true` |
| unified default | `None` → FLAGS.IS_JETSON | `test_unified_default` |
| MEMSIZE_V2 | version-aware memsize | `test_memsize_version_aware` |
| verbose output | verbose=True → logs engine info | `test_verbose_init_logging` |

#### Test Cases

```
class TestInterfaceProperties:
    test_name_is_string                    # Property returns str
    test_all_cached_properties_cached      # Access twice, same object
    test_batch_size_with_inputs             # batch_size from first input dim
    test_batch_size_no_inputs               # Falls back to 1
    test_is_dynamic_batch                   # -1 in first dim → True

class TestInterfaceDefaults:
    test_pagelocked_default_true           # Default is True
    test_pagelocked_explicit_false         # Can be set to False
    test_unified_default_matches_jetson    # Default tracks IS_JETSON
    test_unified_explicit_true             # Can be forced True

class TestInterfaceDestructor:
    test_del_frees_bindings                # Input/output bindings freed
    test_del_deletes_context_engine        # Context and engine attrs deleted
    test_del_suppresses_errors             # No exception even if already freed
```

---

## Coverage Verification

```bash
# Verify engine coverage
pytest tests/engine/ \
    --cov=src/trtutils/_engine.py \
    --cov=src/trtutils/core/_interface.py \
    --cov-branch \
    --cov-report=term-missing \
    -v

# Target: 100% branch coverage on both files
```

---

## Files Created Summary

| File | Lines (est.) | Tests (est.) | GPU Required |
|------|-------------|-------------|--------------|
| `tests/engine/conftest.py` | 50 | — | Yes |
| `tests/engine/test_engine.py` | 250 | 30 | Yes |
| `tests/engine/test_execute.py` | 200 | 20 | Yes |
| `tests/engine/test_direct_exec.py` | 120 | 12 | Yes |
| `tests/engine/test_raw_exec.py` | 100 | 10 | Yes |
| `tests/engine/test_graph_exec.py` | 180 | 15 | Yes |
| `tests/engine/test_mock_execute.py` | 150 | 14 | Yes |
| `tests/engine/test_interface.py` | 150 | 12 | Yes |
| **Total** | **~1200** | **~113** | |
