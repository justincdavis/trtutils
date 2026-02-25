# Phase 1: Core Submodule Tests (100% Branch Coverage)

**Branch:** `test-overhaul/phase-1-core`
**Goal:** Achieve 100% branch coverage on `src/trtutils/core/` with well-structured, parametrized pytest tests.

---

## Prerequisites
- Phase 0 complete (infrastructure, directory structure, root conftest)

## Deliverables
- [ ] `tests/core/conftest.py` — Core-specific fixtures
- [ ] `tests/core/test_cache.py` — 12 functions, all edge cases
- [ ] `tests/core/test_lock.py` — Thread safety primitives
- [ ] `tests/core/test_cuda.py` — Error checking and cuda_call
- [ ] `tests/core/test_context.py` — CUDA context create/destroy
- [ ] `tests/core/test_stream.py` — Stream lifecycle
- [ ] `tests/core/test_memory.py` — All 14 memcpy/alloc functions
- [ ] `tests/core/test_nvrtc.py` — Path detection, compilation, error handling
- [ ] `tests/core/test_engine_load.py` — Engine loading and name extraction
- [ ] `tests/core/test_bindings.py` — Binding dataclass, allocation
- [ ] `tests/core/test_graph.py` — CUDAGraph lifecycle
- [ ] `tests/core/test_kernels.py` — Kernel class and arg management
- [ ] `tests/core/test_flags.py` — Current FLAGS values
- [ ] 100% branch coverage on `src/trtutils/core/` verified

---

## Core Conftest (tests/core/conftest.py)

```python
"""Core test fixtures — CUDA streams, temp engine files, etc."""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def simple_onnx_path() -> Path:
    """Path to a minimal ONNX model for core tests."""
    # tests/core/conftest.py -> tests/core -> tests -> project_root -> data/
    return Path(__file__).parent.parent.parent / "data" / "simple.onnx"


@pytest.fixture(scope="session")
def simple_engine_path(build_test_engine, simple_onnx_path) -> Path:
    """Build and return path to a simple test engine."""
    return build_test_engine(simple_onnx_path)


@pytest.fixture
def cuda_stream():
    """Create a CUDA stream for the test, destroy after."""
    from trtutils.core import create_stream, destroy_stream
    stream = create_stream()
    yield stream
    destroy_stream(stream)


@pytest.fixture
def device_ptr():
    """Allocate 1KB of device memory, free after."""
    from trtutils.core import cuda_malloc, cuda_free
    ptr = cuda_malloc(1024)
    yield ptr
    cuda_free(ptr)
```

---

## Test Plans Per File

### tests/core/test_cache.py

**Source:** `src/trtutils/core/cache.py` (12 public functions + 2 internal helpers)
**Marker:** `@pytest.mark.cpu` (no GPU needed)

#### Functions to Test

| Function | Parameters | Branches |
|----------|-----------|----------|
| `get_cache_dir()` | none | Linear (creates dir if needed) |
| `clear()` | none | Empty dir, dir with files, dir with subdirs |
| `query(filename)` | str | File exists, file doesn't exist |
| `query_file(filename, extension)` | str, str | File with valid ext, invalid ext, no dot |
| `store(filepath, *, overwrite, clear_old)` | Path, bool, bool | New store, overwrite=True, overwrite=False existing, clear_old=True |
| `store_file(filepath, extension, *, overwrite, clear_old)` | Path, str, bool, bool | Same as store but with extension handling |
| `remove(filename)` | str | File exists, file doesn't exist |
| `remove_file(filename, extension)` | str, str | File exists, file doesn't exist |
| `query_timing_cache()` | none | Cache exists, doesn't exist |
| `store_timing_cache(filepath, *, overwrite)` | Path, bool | New store, overwrite existing |
| `save_timing_cache_to_global(cache, *, overwrite)` | Protocol obj, bool | With serialize() method |

#### Internal Helpers to Cover

| Function | Branches |
|----------|----------|
| `_get_cache_file_path(filename, extension)` | Filename with valid ext, invalid ext, no dot |
| `_delete_folder(directory)` | Empty dir, files only, subdirs only, mixed |

#### Parametrization Strategy

```python
# Parametrize query/store/remove generic vs engine-specific pairs
@pytest.mark.parametrize("query_fn,store_fn,remove_fn,ext", [
    pytest.param(query, store, remove, "engine", id="engine"),
    pytest.param(query_file, store_file, remove_file, "cache", id="generic"),
])
def test_store_query_remove_roundtrip(tmp_path, query_fn, store_fn, remove_fn, ext):
    ...

# Parametrize overwrite behavior
@pytest.mark.parametrize("overwrite", [True, False], ids=["overwrite", "no_overwrite"])
def test_store_overwrite(tmp_path, overwrite):
    ...
```

#### Test Cases

```
class TestCacheDir:
    test_get_cache_dir_creates_dir         # Dir created on first call
    test_get_cache_dir_returns_path        # Returns Path object
    test_get_cache_dir_idempotent          # Second call returns same dir

class TestClear:
    test_clear_empty_dir                   # No error on empty dir
    test_clear_with_files                  # Files removed
    test_clear_with_nested_dirs            # Recursive deletion
    test_clear_mixed_content               # Files and dirs removed

class TestQueryStore:
    test_query_nonexistent                 # Returns (False, None)
    test_store_then_query                  # Returns (True, Path)
    test_store_overwrite_true              # Replaces existing
    test_store_overwrite_false_existing    # Does NOT replace
    test_store_clear_old_true              # Removes old files
    test_store_file_with_extension         # Extension appended correctly
    test_query_file_valid_extension        # .engine, .cache extensions
    test_query_file_invalid_extension      # Extension appended

class TestRemove:
    test_remove_existing                   # File removed
    test_remove_nonexistent                # No error
    test_remove_file_with_extension        # Extension-based removal

class TestTimingCache:
    test_query_timing_cache_missing        # Returns (False, None)
    test_store_and_query_timing_cache      # Roundtrip
    test_save_timing_cache_to_global       # Protocol object with serialize()
    test_save_timing_cache_overwrite       # Overwrite existing

class TestInternalHelpers:
    test_get_cache_file_path_no_dot        # Appends extension
    test_get_cache_file_path_valid_ext     # Uses as-is
    test_get_cache_file_path_invalid_ext   # Appends extension
    test_delete_folder_empty               # Removes empty dir
    test_delete_folder_recursive           # Removes nested content
```

**All tests use `tmp_path` fixture** — monkeypatch `get_cache_dir()` to return tmp_path.

---

### tests/core/test_lock.py

**Source:** `src/trtutils/core/_lock.py` (2 Lock objects)
**Marker:** `@pytest.mark.cpu`

#### Objects to Test
- `MEM_ALLOC_LOCK` — threading.Lock for memory allocation
- `NVRTC_LOCK` — threading.Lock for NVRTC compilation

#### Test Cases

```
test_mem_alloc_lock_acquire_release        # Basic acquire/release
test_nvrtc_lock_acquire_release            # Basic acquire/release
test_mem_alloc_lock_context_manager        # with MEM_ALLOC_LOCK: ...
test_nvrtc_lock_context_manager            # with NVRTC_LOCK: ...
test_mem_alloc_lock_thread_safety          # 2 threads compete for lock
test_nvrtc_lock_thread_safety              # 2 threads compete for lock
test_locks_are_independent                 # Acquiring one doesn't block other
```

#### Thread Safety Test Pattern
```python
def test_mem_alloc_lock_thread_safety():
    """Two threads cannot hold MEM_ALLOC_LOCK simultaneously."""
    results = []
    def worker(lock, thread_id):
        with lock:
            results.append(f"{thread_id}_start")
            time.sleep(0.05)
            results.append(f"{thread_id}_end")

    t1 = Thread(target=worker, args=(MEM_ALLOC_LOCK, "t1"))
    t2 = Thread(target=worker, args=(MEM_ALLOC_LOCK, "t2"))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # Verify no interleaving: t1_start,t1_end,t2_start,t2_end or reverse
    assert results[0].endswith("_start")
    assert results[1].endswith("_end")
    assert results[0][:2] == results[1][:2]  # Same thread
```

---

### tests/core/test_cuda.py

**Source:** `src/trtutils/core/_cuda.py` (3 functions)
**Marker:** `@pytest.mark.gpu`

#### Functions to Test

| Function | Branches |
|----------|----------|
| `check_cuda_err(err)` | cudaError_t success, failure; CUresult success, failure; unknown type |
| `cuda_call(result)` | Single return, tuple return (err, value), tuple with success, tuple with error |
| `init_cuda()` | Already initialized, first init |

#### Test Cases

```
class TestCheckCudaErr:
    test_success_cudaError                 # cudart.cudaError_t.cudaSuccess
    test_failure_cudaError                 # Raises RuntimeError
    test_success_CUresult                  # cuda.CUresult.CUDA_SUCCESS
    test_failure_CUresult                  # Raises RuntimeError
    test_unknown_type                      # Non-enum type, no error raised

class TestCudaCall:
    test_single_value_success              # Returns value unchanged
    test_tuple_with_success                # Unpacks and returns value
    test_tuple_with_error                  # Raises RuntimeError
    test_tuple_multiple_values             # Returns all values after error check

class TestInitCuda:
    test_init_succeeds                     # No exception raised
    test_init_idempotent                   # Can call multiple times
```

---

### tests/core/test_context.py

**Source:** `src/trtutils/core/_context.py` (2 functions)
**Marker:** `@pytest.mark.gpu`

#### Test Cases

```
test_create_context_default_device         # Device 0
test_create_destroy_context_lifecycle      # Create then destroy without error
test_create_context_returns_context        # Return type is cuda.CUcontext
test_destroy_context_valid                 # No exception on valid context
```

---

### tests/core/test_stream.py

**Source:** `src/trtutils/core/_stream.py` (3 functions)
**Marker:** `@pytest.mark.gpu`

#### Test Cases

```
test_create_stream                         # Returns valid stream object
test_destroy_stream                        # No exception on valid stream
test_create_destroy_lifecycle              # Create, use, destroy
test_stream_synchronize                    # Sync returns without error
test_multiple_streams                      # Create 3 streams, destroy all
```

---

### tests/core/test_memory.py

**Source:** `src/trtutils/core/_memory.py` (14+ public functions)
**Marker:** `@pytest.mark.gpu`

#### Functions and Parametrization

```python
# Group sync/async memcpy variants
MEMCPY_H2D = [
    pytest.param("memcpy_host_to_device", False, id="h2d_sync"),
    pytest.param("memcpy_host_to_device_async", True, id="h2d_async"),
]
MEMCPY_D2H = [
    pytest.param("memcpy_device_to_host", False, id="d2h_sync"),
    pytest.param("memcpy_device_to_host_async", True, id="d2h_async"),
]
MEMCPY_D2D = [
    pytest.param("memcpy_device_to_device", False, id="d2d_sync"),
    pytest.param("memcpy_device_to_device_async", True, id="d2d_async"),
]

# Parametrize data types
DTYPES = [
    pytest.param(np.float32, id="float32"),
    pytest.param(np.float16, id="float16"),
    pytest.param(np.int32, id="int32"),
    pytest.param(np.int8, id="int8"),
]

# Parametrize data sizes
SIZES = [
    pytest.param(1, id="1_element"),
    pytest.param(100, id="100_elements"),
    pytest.param(10000, id="10k_elements"),
]
```

#### Test Cases

```
class TestMemcpyRoundtrip:
    @parametrize("dtype", DTYPES)
    @parametrize("size", SIZES)
    test_h2d_d2h_roundtrip_sync            # Data integrity after H2D then D2H
    test_h2d_d2h_roundtrip_async           # Same with async + stream sync

class TestMemcpyD2D:
    test_d2d_sync                          # D2D copy, verify via D2H
    test_d2d_async                         # D2D async + sync, verify

class TestMemcpyOffset:
    test_h2d_offset                        # H2D with byte offset
    test_h2d_offset_async                  # H2D async with offset

class TestAllocation:
    test_cuda_malloc                        # Allocate, returns int pointer
    test_cuda_malloc_different_sizes        # 1B, 1KB, 1MB, 16MB
    test_cuda_free                          # Free allocated memory, no error
    test_allocate_pinned_memory             # Returns (host_ptr, nbytes)
    test_cuda_host_free                     # Free pinned memory
    test_allocate_managed_memory            # Returns device pointer
    test_get_ptr_pair                       # Returns (host_ptr, device_ptr)

class TestAllocateToDevice:
    @parametrize("dtype", DTYPES)
    test_allocate_to_device                # Allocate + copy array to device

class TestFreeDevicePtrs:
    test_free_single_ptr                   # Free one pointer
    test_free_multiple_ptrs                # Free list of pointers
    test_free_empty_list                   # No error on empty list
```

---

### tests/core/test_nvrtc.py

**Source:** `src/trtutils/core/_nvrtc.py` (6 public functions)
**Marker:** `@pytest.mark.gpu`

#### Functions to Test

| Function | Key Branches |
|----------|-------------|
| `find_cuda_include_dir()` | CUDA_HOME set, CUDA_PATH set, nvcc in PATH, default paths, not found |
| `check_nvrtc_err(err)` | Success, failure |
| `nvrtc_call(result)` | Single val, tuple success, tuple error |
| `compile_kernel(source, name, ...)` | With/without flags, with/without include dir |
| `load_kernel(ptx, name)` | Valid PTX, module load |
| `compile_and_load_kernel(path, name)` | Full lifecycle from .cu file |

#### Test Cases

```
class TestFindCudaIncludeDir:
    test_finds_cuda_include                # Returns a Path that exists
    test_result_is_cached                  # Second call returns same path (lru_cache)

class TestNvrtcErrorHandling:
    test_check_nvrtc_err_success           # No exception on success
    test_nvrtc_call_success                # Returns value

class TestCompileKernel:
    test_compile_simple_kernel             # Compile a trivial .cu source
    test_compile_returns_ptx              # Result is bytes (PTX)
    test_compile_with_include_dir         # Pass explicit include dir

class TestLoadKernel:
    test_load_from_ptx                    # Load compiled PTX
    test_load_returns_module_and_func     # Returns (module, function)

class TestCompileAndLoad:
    test_full_lifecycle                     # .cu file -> compiled kernel
    test_with_existing_kernel_file         # Uses actual kernel from image/_kernels/
```

---

### tests/core/test_engine_load.py

**Source:** `src/trtutils/core/_engine.py` (in core/, not top-level)
**Marker:** `@pytest.mark.gpu`

#### Functions to Test

| Function | Branches |
|----------|----------|
| `create_engine(path, stream, ...)` | Valid path, invalid path, with/without stream, with/without DLA, no_warn, verbose |
| `get_engine_names(engine)` | Single input/output, multiple inputs/outputs |

#### Test Cases

```
class TestCreateEngine:
    test_create_from_valid_path            # Returns (engine, context, logger, stream)
    test_create_with_external_stream       # Uses provided stream
    test_create_without_stream             # Creates new stream
    test_invalid_path_raises               # FileNotFoundError or RuntimeError
    test_verbose_logging                   # No error with verbose=True
    test_no_warn                           # No error with no_warn=True

class TestGetEngineNames:
    test_returns_input_output_names        # Returns (list[str], list[str])
    test_names_match_engine_spec           # Names match actual engine tensors
```

---

### tests/core/test_bindings.py

**Source:** `src/trtutils/core/_bindings.py`
**Marker:** `@pytest.mark.gpu`

#### Classes and Functions to Test

| Item | Branches |
|------|----------|
| `Binding` dataclass | All fields, free() method |
| `create_binding(array, ...)` | Different dtypes, pagelocked, unified, default |
| `allocate_bindings(engine, context, ...)` | With engine, pagelocked/unified modes |

#### Parametrization

```python
@pytest.mark.parametrize("pagelocked,unified", [
    pytest.param(True, False, id="pagelocked"),
    pytest.param(False, False, id="default"),
    pytest.param(True, True, id="unified"),
])
def test_allocate_bindings(simple_engine_path, pagelocked, unified):
    ...
```

#### Test Cases

```
class TestBinding:
    test_binding_fields                    # All dataclass fields populated
    test_binding_free                      # free() deallocates without error
    test_binding_shape_matches             # Shape matches engine spec
    test_binding_dtype_matches             # Dtype matches engine spec

class TestCreateBinding:
    @parametrize("dtype", [np.float32, np.float16, np.int32])
    test_create_binding_dtypes             # Works with various dtypes

class TestAllocateBindings:
    @parametrize("pagelocked,unified", [...])
    test_allocate_returns_io_bindings      # Returns (inputs, outputs)
    test_bindings_count_matches_engine     # Correct number of I/O bindings
    test_bindings_have_device_allocation   # Each has non-zero allocation ptr
    test_bindings_have_host_allocation     # Each has numpy array
    test_free_all_bindings                 # All can be freed without error
```

---

### tests/core/test_graph.py

**Source:** `src/trtutils/core/_graph.py`
**Marker:** `@pytest.mark.gpu`

#### Classes and Functions to Test

| Item | Key Branches |
|------|-------------|
| `CUDAGraph.__init__()` | Initialize with stream |
| `CUDAGraph.__enter__/__exit__` | Context manager protocol |
| `CUDAGraph.start()` | Begin capture |
| `CUDAGraph.stop()` | End capture, instantiate |
| `CUDAGraph.launch()` | Execute captured graph |
| `CUDAGraph.invalidate()` | Cleanup resources |
| `CUDAGraph.is_captured` | Property transitions |
| `cuda_stream_begin_capture()` | Start capture on stream |
| `cuda_stream_end_capture()` | End capture, return graph |
| `cuda_graph_instantiate()` | Create executable |
| `cuda_graph_launch()` | Launch executable |
| `cuda_graph_destroy()` | Cleanup |

#### Test Cases

```
class TestCUDAGraphLifecycle:
    test_init_not_captured                 # is_captured == False after init
    test_context_manager_capture           # with CUDAGraph as g: ... → captured
    test_start_stop_capture                # Manual start/stop → captured
    test_launch_after_capture              # Launch succeeds
    test_invalidate_clears_state           # is_captured == False after invalidate
    test_launch_without_capture_raises     # RuntimeError

class TestCUDAGraphFunctions:
    test_begin_end_capture                 # Low-level capture functions
    test_instantiate_graph                 # Creates executable from graph
    test_graph_destroy_cleanup             # No error on cleanup
```

---

### tests/core/test_kernels.py

**Source:** `src/trtutils/core/_kernels.py`
**Marker:** `@pytest.mark.gpu`

#### Classes and Functions to Test

| Item | Key Branches |
|------|-------------|
| `Kernel.__init__()` | Load from .cu file, compile |
| `Kernel.free()` | Unload module |
| `Kernel.create_args()` | Different arg types (int, float, ptr) |
| `Kernel.__call__()` | Launch kernel |
| `launch_kernel()` | Grid/block config, stream |
| `create_kernel_args()` | Numpy array arg conversion |

#### Test Cases

```
class TestKernelCompilation:
    test_compile_from_cu_file              # Uses actual .cu kernel file
    test_kernel_has_function               # _function attribute set
    test_kernel_has_module                 # _module attribute set
    test_free_unloads                      # free() succeeds

class TestKernelArgs:
    @parametrize("dtype", [np.float32, np.float16, np.int32, np.uint8])
    test_create_args_dtype                 # Args created for various dtypes
    test_create_args_pointer               # Int pointer arg
    test_create_args_scalar                # Scalar values
    test_args_caching                      # Deque-based arg caching works

class TestKernelLaunch:
    test_launch_simple_kernel              # Execute a compiled kernel
    test_launch_with_stream                # Launch on specific stream
```

---

### tests/core/test_flags.py

**Source:** `src/trtutils/_flags.py` (note: project root, not in core/)
**Marker:** `@pytest.mark.cpu`

#### Test Strategy
FLAGS is initialized at import time and cannot be altered. Test the current values to ensure they're consistent with the installed TRT version.

#### Test Cases

```
class TestFlags:
    test_flags_is_dataclass                # FLAGS is a dataclass instance
    test_flags_has_all_attributes          # All expected attributes exist
    test_exec_backend_mutually_exclusive   # At most one EXEC_ flag is True (for primary)
    test_trt_10_consistency                # If TRT_10, certain features should be True
    test_jetson_detection                  # IS_JETSON matches platform

    @parametrize("attr", [
        "TRT_10", "TRT_HAS_UINT8", "TRT_HAS_INT64",
        "EXEC_ASYNC_V3", "EXEC_ASYNC_V2", "EXEC_ASYNC_V1",
        "EXEC_V2", "EXEC_V1",
        "BUILD_PROGRESS", "BUILD_SERIALIZED",
        "NEW_CAN_RUN_ON_DLA", "MEMSIZE_V2",
        "IS_JETSON", "JIT", "FOUND_NUMBA",
    ])
    test_flag_is_bool(attr)                # Each flag is a bool
```

---

## Coverage Verification

After all tests are written, run:
```bash
pytest tests/core/ --cov=src/trtutils/core --cov-branch --cov-report=term-missing -v
```

Target: 100% branch coverage on all files in `src/trtutils/core/`.

If any branches are unreachable on the current platform (e.g., Jetson-specific code on desktop), mark with `# pragma: no cover` and document why.

---

## Files Created Summary

| File | Lines (est.) | Tests (est.) | GPU Required |
|------|-------------|-------------|--------------|
| `tests/core/conftest.py` | 40 | — | Yes (fixtures) |
| `tests/core/test_cache.py` | 250 | 25 | No |
| `tests/core/test_lock.py` | 80 | 7 | No |
| `tests/core/test_cuda.py` | 100 | 9 | Yes |
| `tests/core/test_context.py` | 50 | 4 | Yes |
| `tests/core/test_stream.py` | 60 | 5 | Yes |
| `tests/core/test_memory.py` | 300 | 30+ | Yes |
| `tests/core/test_nvrtc.py` | 120 | 10 | Yes |
| `tests/core/test_engine_load.py` | 100 | 8 | Yes |
| `tests/core/test_bindings.py` | 150 | 12 | Yes |
| `tests/core/test_graph.py` | 120 | 9 | Yes |
| `tests/core/test_kernels.py` | 150 | 10 | Yes |
| `tests/core/test_flags.py` | 80 | 17 | No |
| **Total** | **~1600** | **~146** | |
