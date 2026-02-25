# Task Plan: Test Suite Overhaul

## Goal
Redesign the trtutils test suite from the ground up using modern pytest features (Python 3.8+), with 100% branch coverage targets, parametrized infrastructure, ratcheting coverage thresholds, and phased rollout with separate PRs per phase.

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| GPU-required for majority of tests | GPU-centric library; avoid brittle mocking. CPU-only tests marked `@pytest.mark.cpu` for CI workflows |
| 100% branch coverage | All code paths exercised including verbose, alternative allocation methods, error handling |
| Ratcheting coverage thresholds | Start at current %, never allow regression. Increases naturally as tests are added |
| Pre-commit checks coverage only | Most tests are long-running GPU tests; pre-commit only validates coverage config, not execution |
| Mirror src/ directory layout | tests/core/, tests/engine/, tests/builder/, etc. map 1:1 to source modules |
| Move existing tests to tests/legacy/ | Preserve for reference during transition, remove when new tests are complete |
| Separate PR per phase | Incremental, reviewable progress |
| Minimal mocking | Only mock where absolutely necessary (e.g., network calls in download tests). Use real GPU for core/engine/builder tests |
| Parametrize proactively | If 2+ cases share code, make parametrizable. Use `pytest.param(id=...)` for readability |
| FLAGS tested as-is | Global _flags.py cannot be altered at runtime; test current TRT version only |
| Design core fixtures upfront | Key shared fixtures planned; module-specific fixtures emerge organically |
| Preserve image test logic | Port existing kernel/preprocessing test assertions into new structure, don't redesign logic |
| pytest-cov + xdist + timeout | Coverage reporting, parallel GPU execution, timeout protection for hanging tests |
| tmp_path for cache tests | Real filesystem operations, more realistic than mocking pathlib |
| Test threading in ImageBatcher | Both: logic tests without threading + integration tests with real threading |
| Parametrize all 19 model classes | Single parametrized suite for general tests; specific correctness tests per model type (detector, classifier, depth_estimator) |

---

## Current Phase
Phase 0

## Phases

### Phase 0: Infrastructure Setup
- [ ] Add pytest-cov, pytest-xdist, pytest-timeout to dev dependencies in pyproject.toml
- [ ] Create .coveragerc or [tool.coverage] config in pyproject.toml
- [ ] Configure branch coverage measurement
- [ ] Set up ratcheting coverage threshold mechanism
- [ ] Move existing tests to tests/legacy/
- [ ] Create new tests/ directory structure mirroring src/
- [ ] Create root conftest.py with core shared fixtures
- [ ] Add `cpu` marker to pytest config
- [ ] Update Makefile test targets
- [ ] Update .pre-commit-config.yaml with coverage check
- **Status:** pending
- **PR:** `test-overhaul/phase-0-infrastructure`

### Phase 1: Core Submodule Tests (100% Branch Coverage)
- [ ] tests/core/test_cache.py — cache.py (tmp_path, all functions, edge cases)
- [ ] tests/core/test_lock.py — _lock.py (thread safety, acquire/release)
- [ ] tests/core/test_cuda.py — _cuda.py (error checking, cuda_call unwrapping)
- [ ] tests/core/test_nvrtc.py — _nvrtc.py (path detection, compile options, error handling)
- [ ] tests/core/test_memory.py — _memory.py (all 14 memcpy/alloc functions, async variants, offsets)
- [ ] tests/core/test_context.py — _context.py (create/destroy CUDA contexts)
- [ ] tests/core/test_stream.py — _stream.py (create/destroy/synchronize streams)
- [ ] tests/core/test_engine_load.py — _engine.py (engine loading, name extraction)
- [ ] tests/core/test_bindings.py — _bindings.py (Binding dataclass, create_binding, allocate_bindings, dtypes, shapes, memory modes)
- [ ] tests/core/test_graph.py — _graph.py (CUDAGraph lifecycle, context manager, capture/launch/invalidate)
- [ ] tests/core/test_kernels.py — _kernels.py (Kernel class, arg creation, launch, caching)
- [ ] tests/core/test_flags.py — _flags.py (current version flag values, dataclass structure)
- [ ] tests/core/conftest.py — core-specific fixtures (CUDA streams, temp engine files, etc.)
- [ ] Verify 100% branch coverage on src/trtutils/core/
- **Status:** pending
- **PR:** `test-overhaul/phase-1-core`

### Phase 2: TRTEngine Tests (100% Branch Coverage)
- [ ] tests/engine/test_engine.py — TRTEngine (init, backends, warmup, properties)
- [ ] tests/engine/test_execute.py — execute() (H2D/D2H, output copy, no_copy, verbose, debug)
- [ ] tests/engine/test_direct_exec.py — direct_exec() (GPU pointers, set_pointers flag)
- [ ] tests/engine/test_raw_exec.py — raw_exec() (returns GPU pointers, no sync)
- [ ] tests/engine/test_graph_exec.py — graph_exec() + CUDA graph lifecycle (capture, replay, invalidate, recapture)
- [ ] tests/engine/test_mock_execute.py — mock_execute(), warmup(), get_random_input()
- [ ] tests/engine/test_interface.py — TRTEngineInterface abstract contract (properties, cached_property, __call__)
- [ ] tests/engine/conftest.py — engine fixtures (engine paths, built engines, test data)
- [ ] Verify 100% branch coverage on src/trtutils/_engine.py + src/trtutils/core/_interface.py
- **Status:** pending
- **PR:** `test-overhaul/phase-2-engine`

### Phase 3: Pre-commit Coverage Check (Core + Engine)
- [ ] Add coverage pre-commit hook for src/trtutils/core/ and src/trtutils/_engine.py
- [ ] Record baseline coverage numbers (ratchet starting point)
- [ ] Validate pre-commit hook runs correctly
- [ ] Document coverage workflow in CONTRIBUTING.md or similar
- **Status:** pending
- **PR:** `test-overhaul/phase-3-precommit-core-engine`

### Phase 4: Builder Tests (100% Branch Coverage)
- [ ] tests/builder/test_build.py — build_engine() (all parameter combos, precision flags, DLA, timing cache)
- [ ] tests/builder/test_onnx.py — read_onnx() (file validation, errors, network creation)
- [ ] tests/builder/test_calibrator.py — EngineCalibrator (cache file I/O, batcher, get_batch)
- [ ] tests/builder/test_image_batcher.py — ImageBatcher (loading, resizing, batching, threading behavior, cleanup)
- [ ] tests/builder/test_synthetic_batcher.py — SyntheticBatcher (random generation, shape validation, edge cases)
- [ ] tests/builder/test_dla.py — can_run_on_dla(), build_dla_engine()
- [ ] tests/builder/test_hooks.py — yolo_efficient_nms_hook(), make_plugin_field()
- [ ] tests/builder/test_onnx_utils.py — extract_subgraph(), split_model() (ONNX graph ops)
- [ ] tests/builder/test_progress.py — ProgressBar (phase tracking, interruption)
- [ ] tests/builder/conftest.py — builder fixtures (ONNX files, temp dirs, built configs)
- [ ] Add coverage pre-commit hook for src/trtutils/builder/
- [ ] Verify 100% branch coverage on src/trtutils/builder/
- **Status:** pending
- **PR:** `test-overhaul/phase-4-builder`

### Phase 5: Download, Inspect, Models Tests (100% Branch Coverage)
- [ ] tests/download/test_download.py — download(), download_model() (validation, config loading, error paths)
- [ ] tests/download/test_config.py — get_supported_models(), load_model_configs() (all config files)
- [ ] tests/download/conftest.py — download fixtures (temp dirs, mock network where needed)
- [ ] tests/inspect/test_inspect.py — inspect_engine(), inspect_onnx_layers(), get_engine_names()
- [ ] tests/inspect/conftest.py — inspect fixtures
- [ ] tests/models/test_models.py — Parametrized across all 19 model classes (config validation, download/build signatures)
- [ ] tests/models/test_detector_correctness.py — Detector-specific correctness tests
- [ ] tests/models/test_classifier_correctness.py — Classifier-specific correctness tests
- [ ] tests/models/test_depth_estimator_correctness.py — DepthEstimator-specific correctness tests
- [ ] tests/models/conftest.py — model fixtures (engine paths, test images, ground truth)
- [ ] Add coverage pre-commit hooks for download/, inspect/, models/
- [ ] Verify 100% branch coverage
- **Status:** pending
- **PR:** `test-overhaul/phase-5-download-inspect-models`

### Phase 6: Image Module Port
- [ ] tests/image/test_preproc.py — Port preprocessor tests (CPU/CUDA/TRT, parity, batch, performance)
- [ ] tests/image/test_postproc.py — Port postprocessor tests (YOLOv10, EfficientNMS, RF-DETR, DETR, classification)
- [ ] tests/image/test_detector.py — Detector class tests (init, inference modes, batch processing)
- [ ] tests/image/test_classifier.py — Classifier class tests
- [ ] tests/image/test_depth_estimator.py — DepthEstimator class tests
- [ ] tests/image/test_image_model.py — ImageModel base class tests
- [ ] tests/image/kernels/test_letterbox.py — Port letterbox kernel tests
- [ ] tests/image/kernels/test_linear.py — Port linear resize kernel tests
- [ ] tests/image/kernels/test_sst.py — Port SST kernel tests (all variants: standard, fast, imagenet)
- [ ] tests/image/kernels/test_performance.py — Port kernel performance benchmarks
- [ ] tests/image/onnx/test_preproc_engine.py — Port TRT preprocessing engine tests
- [ ] tests/image/test_sahi.py — NEW: SAHI tests (currently untested)
- [ ] tests/image/conftest.py — Port and expand image fixtures
- [ ] tests/image/kernels/conftest.py — Port kernel fixtures
- [ ] Add coverage pre-commit hook for src/trtutils/image/
- [ ] Verify 100% branch coverage
- [ ] Remove tests/legacy/ after all ports verified
- **Status:** pending
- **PR:** `test-overhaul/phase-6-image-port`

---

## Test Infrastructure Design

### Pytest Configuration (pyproject.toml additions)

```toml
[tool.pytest.ini_options]
markers = [
    "gpu: tests that require a CUDA-capable GPU",
    "cpu: tests that can run without GPU hardware",
    "dla: tests that run on NVIDIA DLA cores (Jetson only)",
    "performance: benchmarks and performance regression tests",
    "slow: tests that take more than 30 seconds",
    "download: tests that download models from the internet",
    "cuda_graph: tests that require CUDA graph support (async_v3)",
    "correctness: tests that validate detection/classification output content",
    "regression: tests for specific bug regressions",
]
timeout = 300  # 5-minute default timeout
addopts = "--timeout=300"

[tool.coverage.run]
source = ["src/trtutils"]
branch = true
omit = [
    "src/trtutils/compat/*",
    "src/trtutils/_log.py",
]

[tool.coverage.report]
fail_under = 0  # Will be ratcheted up per-module
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "if __name__ ==",
    "@overload",
]
```

### New Dependencies (pyproject.toml)

```
pytest-cov>=4.0
pytest-xdist>=3.0
pytest-timeout>=2.0
```

### Directory Structure

```
tests/
├── conftest.py              # Root: GPU detection, markers, shared fixtures
├── core/
│   ├── __init__.py
│   ├── conftest.py          # Core fixtures: CUDA streams, temp engines
│   ├── test_cache.py
│   ├── test_lock.py
│   ├── test_cuda.py
│   ├── test_nvrtc.py
│   ├── test_memory.py
│   ├── test_context.py
│   ├── test_stream.py
│   ├── test_engine_load.py
│   ├── test_bindings.py
│   ├── test_graph.py
│   ├── test_kernels.py
│   └── test_flags.py
├── engine/
│   ├── __init__.py
│   ├── conftest.py          # Engine fixtures: built engines, test data
│   ├── test_engine.py
│   ├── test_execute.py
│   ├── test_direct_exec.py
│   ├── test_raw_exec.py
│   ├── test_graph_exec.py
│   ├── test_mock_execute.py
│   └── test_interface.py
├── builder/
│   ├── __init__.py
│   ├── conftest.py          # Builder fixtures: ONNX files, temp dirs
│   ├── test_build.py
│   ├── test_onnx.py
│   ├── test_calibrator.py
│   ├── test_image_batcher.py
│   ├── test_synthetic_batcher.py
│   ├── test_dla.py
│   ├── test_hooks.py
│   ├── test_onnx_utils.py
│   └── test_progress.py
├── download/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_download.py
│   └── test_config.py
├── inspect/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_inspect.py
├── image/
│   ├── __init__.py
│   ├── conftest.py          # Image fixtures: preprocessor factories, output mocks
│   ├── test_preproc.py
│   ├── test_postproc.py
│   ├── test_detector.py
│   ├── test_classifier.py
│   ├── test_depth_estimator.py
│   ├── test_image_model.py
│   ├── test_sahi.py
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── conftest.py      # Kernel fixtures: CUDA streams, compilation
│   │   ├── test_letterbox.py
│   │   ├── test_linear.py
│   │   ├── test_sst.py
│   │   └── test_performance.py
│   └── onnx/
│       ├── __init__.py
│       └── test_preproc_engine.py
├── models/
│   ├── __init__.py
│   ├── conftest.py          # Model fixtures: engine paths, test images
│   ├── test_models.py       # Parametrized across all 19 model classes
│   ├── test_detector_correctness.py
│   ├── test_classifier_correctness.py
│   └── test_depth_estimator_correctness.py
├── legacy/                  # Existing tests (preserved for reference)
│   └── ... (all current test files)
└── data/                    # Test data (engines, images, ONNX files)
    ├── engines/
    └── images/
```

### Core Shared Fixtures (root conftest.py)

```python
# GPU availability detection
@pytest.fixture(scope="session")
def gpu_available():
    """Check if CUDA GPU is available."""
    try:
        from trtutils.compat._libs import cudart
        err, count = cudart.cudaGetDeviceCount()
        return count > 0
    except Exception:
        return False

# Auto-skip GPU tests when no GPU
@pytest.fixture(autouse=True)
def skip_gpu_tests(request, gpu_available):
    """Skip GPU-marked tests when no GPU is available."""
    if request.node.get_closest_marker("gpu") and not gpu_available:
        pytest.skip("No CUDA GPU available")

# Shared test images
@pytest.fixture(scope="session")
def test_images():
    """Load test images from data directory."""
    ...

# Parametrized batch sizes
@pytest.fixture(params=[1, 2, 4, 8], ids=["batch1", "batch2", "batch4", "batch8"])
def batch_size(request):
    return request.param

# Test engine builder with caching
@pytest.fixture(scope="session")
def build_test_engine():
    """Factory fixture: builds and caches test engines."""
    def _build(onnx_path, **kwargs):
        ...
    return _build
```

### Parametrization Patterns

```python
# Pattern 1: Memory function testing (core/test_memory.py)
@pytest.mark.gpu
@pytest.mark.parametrize("memcpy_fn,is_async", [
    pytest.param(memcpy_host_to_device, False, id="h2d_sync"),
    pytest.param(memcpy_host_to_device_async, True, id="h2d_async"),
    pytest.param(memcpy_device_to_host, False, id="d2h_sync"),
    pytest.param(memcpy_device_to_host_async, True, id="d2h_async"),
])
def test_memcpy_roundtrip(memcpy_fn, is_async, cuda_stream):
    ...

# Pattern 2: Model class testing (models/test_models.py)
ALL_MODELS = [
    pytest.param(YOLO3, "yolov3", id="yolo3"),
    pytest.param(YOLO5, "yolov5n", id="yolo5"),
    ...
    pytest.param(RFDETR, "rfdetr_b", id="rfdetr"),
]

@pytest.mark.parametrize("model_cls,model_name", ALL_MODELS)
def test_model_has_download(model_cls, model_name):
    ...

# Pattern 3: Preprocessor parity (image/test_preproc.py)
@pytest.mark.gpu
@pytest.mark.parametrize("preproc_type", ["cpu", "cuda", "trt"])
@pytest.mark.parametrize("resize_method", ["linear", "letterbox"])
@pytest.mark.parametrize("use_mean_std", [False, True], ids=["no_norm", "imagenet_norm"])
def test_preprocessor_output_shape(preproc_type, resize_method, use_mean_std, test_images):
    ...
```

### Coverage Ratcheting Mechanism

A script or pre-commit hook that:
1. Runs `pytest --cov` to generate coverage report
2. Reads current coverage % from a `.coverage-threshold` file (JSON per-module)
3. Fails if any module's coverage dropped below its recorded threshold
4. On success, updates the threshold file if coverage increased

```json
// .coverage-threshold (committed to repo)
{
    "trtutils.core": 0,
    "trtutils._engine": 0,
    "trtutils.builder": 0,
    "trtutils.download": 0,
    "trtutils.inspect": 0,
    "trtutils.models": 0,
    "trtutils.image": 0
}
```

---

## Per-Module Testing Plans

### Core: cache.py
- All 12 functions tested with `tmp_path`
- Parametrize: `query`/`query_file`, `store`/`store_file`, `remove`/`remove_file` (generic vs engine-specific)
- Edge cases: nonexistent dirs, duplicate stores, concurrent access, timing cache serialize

### Core: _memory.py
- All 14 functions tested on GPU
- Parametrize: sync vs async variants, different dtypes (float32, float16, int8, int32), different sizes
- Test: roundtrip integrity (H2D then D2H, verify data matches), offset transfers, managed memory, pinned memory
- Cleanup: verify cuda_free/cuda_host_free don't leak

### Core: _bindings.py
- Parametrize: dtypes (float32, float16, int32), shapes (1D, 2D, 4D), memory modes (pagelocked, unified, default)
- Test: Binding creation, field values, free(), allocate_bindings with real engine

### Core: _graph.py
- Context manager protocol (enter/exit)
- Capture lifecycle: start -> stop -> launch -> invalidate
- Error cases: launch without capture, double capture
- is_captured property transitions

### Core: _kernels.py
- Kernel compilation from .cu files
- Arg creation with different numpy dtypes
- Arg caching (deque behavior)
- Kernel launch and results
- Cleanup (free/unload)

### Engine: TRTEngine
- Backend selection: auto, async_v2, async_v3, invalid
- Warmup: with/without, different iteration counts
- Properties: all cached properties, batch_size, dynamic batch
- Execute: normal flow, no_copy, verbose, debug flags
- CUDA graph: capture on first execute, graph_exec, invalidate, recapture
- Error handling: invalid engine path, wrong input shapes

### Builder: build_engine()
- Parametrize: fp16/int8/none precision, with/without DLA, with/without timing cache
- Shapes: fixed shapes, dynamic shapes
- Hooks: with/without NMS hook
- Cache: with/without engine caching
- Error paths: invalid ONNX, missing files

### Builder: Batchers
- ImageBatcher: image loading, resize methods, batch sizes, threading cleanup, prefetch queue
- SyntheticBatcher: shapes, dtypes, data ranges, batch counts, edge cases (0 batches)
- AbstractBatcher: interface compliance

### Download
- get_supported_models(): all models present, no duplicates
- load_model_configs(): all JSON files parse, required keys present
- Model name validation: valid names pass, invalid names raise
- Image size constraints per model family

### Inspect
- inspect_engine(): tensor info extraction, memory size, version-aware paths
- inspect_onnx_layers(): layer enumeration, precision info
- get_engine_names(): input/output name ordering

### Models (Parametrized)
- All 19 classes: has download(), has build(), correct defaults (imgsz, input_range, resize_method, preprocessor)
- Detector correctness: expected detections on known images
- Classifier correctness: expected classifications on known images
- DepthEstimator correctness: expected depth output structure

### Image (Port)
- Preprocessors: preserve all parity tests (CPU==CUDA==TRT), batch tests, performance tests
- Postprocessors: preserve all format-specific tests with output factories
- Kernels: preserve correctness tests (letterbox, linear, SST variants)
- NEW: SAHI tests, dynamic batch tests

---

## Key Questions (Answered)
1. GPU vs CPU split? → GPU-required majority, minimal mocking, `@pytest.mark.cpu` for CI
2. 100% coverage definition? → 100% branch coverage, all code paths
3. Coverage thresholds? → Ratcheting, never regress
4. Pre-commit? → Coverage check only, not test execution
5. Parametrization? → Proactive, parametrize if 2+ cases share code
6. FLAGS testing? → Current version only, no import-time mocking
7. Model testing? → Parametrize all 19 for general, specific per model type
8. Directory structure? → Mirror src/ layout
9. Existing tests? → Move to tests/legacy/
10. PR strategy? → Separate PR per phase
11. Cache tests? → tmp_path (real filesystem)
12. Fixtures? → Design core upfront, organic for module-specific
13. Image port? → Preserve logic, new structure
14. Plugins? → pytest-cov + xdist + timeout
15. CI workflows? → Tests only, CI is separate task
16. Threading (ImageBatcher)? → Both logic and integration tests

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

## Notes
- _flags.py is at project root (src/trtutils/_flags.py), NOT in core/
- Target hardware: Any NVIDIA GPU + Jetson with JetPack 5, 6, 7
- Python 3.8 compatibility required for all test code
- Dev env: WSL2, RTX 5080, Python 3.13.1
