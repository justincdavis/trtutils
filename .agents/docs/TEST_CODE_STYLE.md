# Testing Code Standards

Standards and conventions for the test suite. All new and modified tests must follow these rules.

## File Structure

```python
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""Tests for src/trtutils/[module].py -- [brief description]."""

from __future__ import annotations
```

- Copyright header + MIT license + module docstring on every file.
- `from __future__ import annotations` is always the first import.
- Import order: stdlib, third-party (`numpy`, `pytest`), local (`trtutils.*`).

## Imports

- All imports at the top of the file. No lazy imports inside functions, methods, or fixtures.
- Do not use `TYPE_CHECKING` blocks unless necessary for avoiding circular imports.

## Test Organization

- **Flat functions by default.** Tests should be module-level functions, not wrapped in classes.
- **No section comments.** Do not use `# ====` block separators between tests.
- **Classes only for parametrized groups.** Use a class only when you need `@pytest.mark.parametrize` on the class itself to run all methods against multiple inputs (e.g. multiple engine paths). Use an autouse fixture for setup/teardown:

```python
@pytest.mark.parametrize("engine_path", ENGINE_PATHS)
class TestEngine:
    @pytest.fixture(autouse=True)
    def _setup_engine(self, engine_path) -> None:
        self.engine = TRTEngine(engine_path, warmup=False)
        yield
        del self.engine

    def test_spec_properties(self) -> None:
        """Spec properties return correctly typed lists."""
        assert isinstance(self.engine.input_spec, list)
```

- **No verbose/debug-only tests.** Do not write tests that only assert "verbose=True doesn't crash" or "debug=True doesn't crash". These verify nothing meaningful.
- **No trivial "doesn't crash" tests.** If a test only calls a method and asserts the result is not None, it's likely redundant with a more comprehensive test.

## Strategy

- A single comprehensive test with multiple stages and parameterizations is cleaner than many small tests.
- If we have an end-to-end test that performs full input/output validation, parametrize it rather than splitting into separate functions.
- Prefer round-trip validation over checking intermediate state.
- When a test covers the same behavior as another (e.g. `test_returns_outputs` vs `test_matches_execute`), keep only the more comprehensive one.

## Real CUDA Over Mocks

Use real CUDA calls. Do not mock CUDA runtime, TensorRT, or NVRTC APIs.

Mocking is acceptable **only** to verify cleanup routing (e.g. confirming `cudaFreeHost` is called instead of `cudaFree`). When mocking is necessary, use `patch.object` with `wraps=` to preserve real behavior while tracking calls.

Environment-specific variance (TRT versions, Jetson, include paths) is handled by Docker CI containers, not by mocking flags or paths.

## Parametrize Over Separate Tests

Combine input variants into `@pytest.mark.parametrize` instead of writing separate test functions.

```python
@pytest.mark.parametrize(
    "stream_source",
    [
        pytest.param("none_default", id="no-stream"),
        pytest.param("explicit_none", id="explicit-none"),
        pytest.param("real_stream", id="real-stream"),
    ],
)
def test_allocate_managed_memory(stream_source, request) -> None:
```

- Always use `pytest.param(..., id="...")` for readable test IDs.
- IDs use kebab-case: `no-stream`, `delete-source`, `engine-custom-name`.
- Use `itertools.product` for exhaustive dtype/shape coverage.

## Markers

- `@pytest.mark.cpu` -- tests that genuinely need no GPU.
- `@pytest.mark.jetson` -- Jetson-specific tests (auto-skipped on non-Jetson via autouse fixture).
- `@pytest.mark.cuda_graph` -- tests that exercise CUDA graph capture/replay. All tests touching cuda_graph must be marked.
- `@pytest.mark.usefixtures("cuda_context")` -- inject fixture without naming it as a parameter.
- Do not use unregistered markers (e.g. `pytest.mark.gpu` is not registered -- do not use it).
- Use `pytest.skip()` instead of `xfail` for features that may not be available (e.g. `async_v2`).

## Fixtures

- `yield` for resource cleanup; the teardown after `yield` frees the resource.
- Session-scoped for expensive operations (engine builds, image loading).
- Function-scoped for test-local CUDA resources (`cuda_context`, `cuda_stream`, `device_ptr`).
- Prefix autouse/internal fixtures with `_` (e.g. `_skip_if_cannot_build` in `tests/builder/_gpu_fixtures.py`).
- No type hints on fixture parameters (pytest provides them).
- Remove unused fixtures.

## Round-Trip Validation

Test that data survives a full cycle rather than checking intermediate state.

```python
def test_memcpy_roundtrip_sync(dtype, shape) -> None:
    """Data survives a H2D->D2H roundtrip (sync)."""
    src = np.random.rand(*shape).astype(dtype)
    # ... H2D, D2H ...
    np.testing.assert_array_equal(src, dst)
```

For caches: `store -> query -> remove` in a single test.
For graphs: `capture -> launch -> sync -> invalidate` lifecycle.

## Error Paths

Use `pytest.raises` with a `match` pattern:

```python
with pytest.raises(RuntimeError, match="no graph has been captured"):
    graph.launch()
```

Only test error paths that can be triggered with real inputs (invalid paths, wrong types, bad state). Do not mock internals to simulate errors.

## Resource Cleanup

- Call `.free()` explicitly when resources are not managed by fixtures.
- Use fixture `yield` for automatic teardown.
- For collections: `free_device_ptrs(ptrs)` or a loop.
- Prefix unused fixture return values with `_`: `_engine, _context, _stream = simple_engine`.

## Docstrings

One sentence, present tense, describing what the test verifies:

```python
"""Data survives a H2D->D2H roundtrip (sync)."""
"""store_file -> query_file -> remove_file round-trip for various extensions."""
"""invalidate() on fresh or already-invalidated graph does not raise."""
```

## Type Hints

- Always `-> None` on test functions.
- No type hints on fixture parameters (pytest provides them).
- Type hints on parametrized values only when non-obvious.

## Assertions

- `assert` for most checks.
- `np.testing.assert_array_equal` for array comparisons.
- `assert x is True` / `assert x is False` for explicit boolean checks.

## What Not to Test

- TRT version branching (Docker CI handles this).
- Error message wording from CUDA/NVRTC (brittle, vendor-controlled).
- Flag detection and Jetson identification (environment-dependent).
- Thread lock behavior (CUDA APIs are inherently thread-safe).
- Verbose/debug flag paths in isolation (no meaningful assertion possible).
