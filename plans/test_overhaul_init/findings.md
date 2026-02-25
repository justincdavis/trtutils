# Findings & Decisions

## Requirements
- Full test suite redesign from the ground up (not a remake of existing tests)
- Modern pytest features (Python 3.8+ compatible)
- Maximize parametrization
- Phased rollout: core -> TRTEngine -> builder -> download/inspect/models -> image (port existing)
- Pre-commit checks for coverage at each phase
- 100% coverage targets for core, TRTEngine, builder
- Target hardware: Any NVIDIA GPU + Jetson with JetPack 5, 6, 7

## Current Test Suite Summary
- **24 test files**, ~1,195 tests, ~7,167 lines
- Organized by submodule: tests/image/, tests/models/, tests/download/, tests/inspect/, tests/jetson/, tests/dla/, tests/trtexec/
- Pytest markers: gpu, dla, performance, slow, download, cuda_graph, correctness, regression
- Good parametrization (85 uses), 32 test classes, conftest hierarchy
- **No coverage reporting** (no pytest-cov, no .coveragerc)
- **No pre-commit coverage checks**
- Limited mocking (14 pytest.raises, 3 unittest.mock.patch uses)
- Almost all tests require GPU hardware

## Core Submodule Analysis (src/trtutils/core/)
| File | GPU Required | Mock Strategy |
|------|-------------|---------------|
| cache.py | No | Fully testable (filesystem ops) |
| _lock.py | No | Standard threading tests |
| _cuda.py | Partial | Mock error enums |
| _nvrtc.py | Partial | Mock nvrtc + path detection |
| _flags.py | Partial | Mock TRT at import |
| _interface.py | Yes | Mock engine/context/bindings |
| _bindings.py | Yes | Mock engine/context |
| _context.py | Yes | Mock cuda module |
| _stream.py | Yes | Mock cudart module |
| _engine.py | Yes | Mock TRT Runtime |
| _graph.py | Yes | Mock cudart graph functions |
| _kernels.py | Yes | Mock NVRTC + arg tests |
| _memory.py | Yes | NOT easily mockable |

## TRTEngine Analysis (src/trtutils/_engine.py)
- Concrete implementation of TRTEngineInterface
- Backends: auto, async_v3, async_v2
- CUDA graph support (capture/replay/invalidate)
- Methods: execute(), direct_exec(), raw_exec(), graph_exec()
- All execution paths require GPU

## Builder Analysis (src/trtutils/builder/)
- build_engine(): 30+ parameters, ONNX parsing, network hooks, profiles, formats
- read_onnx(): File validation + TRT network creation
- EngineCalibrator: TRT INT8 calibrator with file cache
- ImageBatcher: Multithreaded image loading, resize, batching (CPU-testable)
- SyntheticBatcher: Random data generation (fully CPU-testable)
- DLA utilities: can_run_on_dla(), build_dla_engine()
- Hooks: yolo_efficient_nms_hook() - network manipulation
- ONNX utils: extract_subgraph(), split_model() (fully CPU-testable)
- ProgressBar: TRT progress monitor with tqdm

## Download/Inspect/Models Analysis
- download module: get_supported_models(), load_model_configs() CPU-testable; actual download requires network + uv
- inspect module: All functions require TRT/CUDA
- models module: 19 model classes (YOLO family + DETR family), validation logic CPU-testable

## Image Submodule (Existing Tests to Port)
- Preprocessors: CPU, CUDA, TRT backends - parity tests, batch tests, performance
- Postprocessors: YOLOv10, EfficientNMS, RF-DETR, DETR, classification
- Kernels: letterbox, linear, SST, SST_FAST, SST_IMAGENET - correctness + perf
- ONNX preprocessing engines
- SAHI: no tests currently
- ~2000+ lines of existing test code

## Dev Environment
- WSL2, Ubuntu 20.04, Python 3.13.1 (venv), RTX 5080 16GB
- gcc 9.4, cmake 3.26, uv 0.9.18
- pytest >=6.2.0,<8.5

## Resources
- pyproject.toml: pytest markers config (line ~150)
- .pre-commit-config.yaml: ruff-format, ruff-lint, ty-typecheck
- Makefile: test, typecheck, ci targets
- tests/common.py: Engine building with caching
- tests/helpers.py: Image loading, path constants

---
*Update this file after every 2 view/browser/search operations*
