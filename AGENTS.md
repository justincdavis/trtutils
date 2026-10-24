# AGENTS.md

Instructions for coding agents (Claude Code, Codex, Cursor, ...) working on trtutils.
`CLAUDE.md` imports this file; edit this one.

## Project Overview

trtutils is a Python library providing high-level utilities for NVIDIA TensorRT. It simplifies
TensorRT engine creation, inference execution, and image model deployment. It supports CUDA 11,
12, and 13 with the corresponding TensorRT versions (8.6 on JetPack 6 through 10.x).

## Build & Development Commands

```bash
make fix             # ruff format + ruff lint with autofix (ci/run_ruff.sh)
make typecheck       # ty static type checker (ci/run_ci.sh --typecheck)
make test            # ci/run_tests.sh: python3 -m pytest -rP -v tests/
make test-cpu        # pytest -m cpu
make test-cov        # branch coverage report

# single file / test
python3 -m pytest tests/engine/test_engine.py -v
python3 -m pytest tests/engine/test_engine.py::test_init_default_args -v
```

- The shared `.venv` has a non-editable trtutils install: always run with `PYTHONPATH=src`,
  or you test the stale installed copy.
- ruff/ty come from `.venv-ci/bin/` when present (also used by the pre-commit hooks), else PATH.
  If CI lint fails but `make fix` passes, `.venv-ci` lags the pins in `pyproject.toml`.
- `pytest.ini` deselects `research` by default; opt in with `-m research`.

## Code Style

- **Line length**: 101 (`ruff.toml`)
- **Target Python**: 3.8 (all code must be compatible)
- **Linter**: ruff (`ruff.toml`); **type checker**: ty, not mypy (`ty.toml`)
- **All modules use** `from __future__ import annotations`
- **Copyright header** on every Python file (`ci/check_copyright.sh` pre-commit hook)
- **Simple**: write code as simply as possible, with minimal exception handling / try-except
- **Tidy comments**: minimal, only where names and flow cannot explain; lowercase, short, casual
- **Logging**: no prints ever; use `trtutils.LOG`, log only in `if verbose` blocks or for
  warnings/errors in exceptions we must handle
- **Imports**: at file level, no lazy imports inside functions (tests included)
- **Defaults**: per-call overrides fall back with `x if x is not None else self._x`, never
  `x or self._x` (0.0 / False are valid values)

## Documentation

- **NumPy docstrings** for all public functions; re-check them when you change a signature.
- **Module docstrings** on modules that export public APIs.
- User-facing changes get a `CHANGELOG.rst` entry under `Unreleased`.

## Architecture

### Core Engine Abstraction (`src/trtutils/`)
- `TRTEngineInterface` (`core/_interface.py`): abstract base defining the engine contract.
- `TRTEngine` (`_engine.py`): concrete implementation. async_v2 and async_v3 backends, CUDA
  graphs, pagelocked/unified memory.
- `core/`: low-level CUDA ops: memory (`Buffer` in `_buffer.py`), streams, CUDA graphs,
  NVRTC kernel compilation.
- `FLAGS` (`_flags.py`): dataclass of TensorRT/platform capabilities detected at import
  (`FLAGS.EXEC_ASYNC_V3`, `FLAGS.TRT_10`, `FLAGS.TRT_VERSION`, `FLAGS.IS_JETSON`,
  `FLAGS.IS_BLACKWELL`, `FLAGS.NVTX_ENABLED`). `FLAGS.TRT_10` marks the I/O tensor API and is
  True on TRT 8.6 too; gate on versions with `FLAGS.TRT_VERSION`.

### Image Processing (`src/trtutils/image/`)
- `ImageModel` (`_image_model.py`): base for image models; preprocessing config, `preprocess`,
  `prepare_inputs`, `infer`, shared `_run_core` / `_end2end_core`, timing profiles.
- `Detector` (`_detector.py`), `Classifier` (`_classifier.py`), `DepthEstimator`
  (`_depth_estimator.py`), `HandInteractionDetector` (`_hand_interaction.py`): `ImageModel` +
  the matching interface in `interfaces.py`.
- `_schema.py`: `InputSchema` / `OutputSchema` detection for detector exports.
- `preprocessors/`: CPU, CUDA, and TensorRT preprocessors (letterbox/linear resize, normalize).
- `postprocessors/`: output decoding per family (detection, classifier, depth, hand interaction).
- `kernels.py` / `_kernels/`: CUDA kernels for GPU preprocessing. `sahi/`: sliced inference.

### Model Implementations (`src/trtutils/models/`)
Pre-configured wrappers (`detectors/`, `classifiers/`, `depth_estimators/`,
`hand_interaction/`) that subclass an image model plus the `Model` mixin (`_model.py`), which
provides `download()` / `build()` from class attributes (`_model_type`, `_default_imgsz`, ...).

### Download / Export (`src/trtutils/download/`)
`configs/*.json` list downloadable models; `_models/` holds the export functions dispatched in
`_download.py`; `model_deps/*.txt` pin each exporter's venv requirements. CLI:
`trtutils download --model <name> --output <path.onnx> [--imgsz N] [--dynamic]`.

### Engine Building (`src/trtutils/builder/`)
- `build_engine()` (`_build.py`): build TensorRT engines from ONNX
- `hooks/`: build hooks for model-specific optimizations
- `onnx/`: ONNX graph manipulation utilities

## Key Patterns

- GPU test fixtures live in `tests/_gpu_fixtures.py` (loaded by `tests/conftest.py` unless
  `TRTUTILS_IGNORE_MISSING_CUDA=1`). `build_test_engine` caches engines in `data/engines/`
  with the TensorRT version in the filename; `tests/models/conftest.py::build_model_engine`
  builds through `Model.build()` and skips when the ONNX is missing.
- Execution backends are selected via `FLAGS.EXEC_ASYNC_V3` / `FLAGS.EXEC_ASYNC_V2`.
- NVTX: enable with `trtutils.enable_nvtx()`; use `with NVTX("scope"):` or the
  `if FLAGS.NVTX_ENABLED: nvtx.push_range(...)` / `pop_range()` pattern used in
  `TRTEngineInterface`, `ImageModel`, and `Detector`.

## Workflow Rules

- One git worktree per PR; never edit another agent's worktree or the main checkout
  (`trtutils-worktree` skill).
- After every change: `make fix` then `make typecheck`, without being asked.
- Full suite before reporting: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests`.
  Compare failures against `main` on the same machine and data before calling one a regression.
- Commits end with the `Co-Authored-By:` / `Claude-Session:` trailers used on `main`;
  PRs are squash-merged, and only after the user says merge.
- Never commit credentials, hostnames, IPs, or user-specific absolute paths.
- Follow-up plans are delta-only: point at the prior plan instead of restating it.

## Testing Standard

All tests follow [docs/TEST_CODE_STYLE.md](docs/TEST_CODE_STYLE.md) (`trtutils-test-style`
skill): flat functions, `pytest.param(..., id="kebab-case")` matrices, real CUDA over mocks,
registered markers only (see `pytest.ini`).

## Skills

Reusable workflows live in `.agents/skills/<name>/SKILL.md` (`.claude/skills` links there).

| Skill | Use it to |
| --- | --- |
| [trtutils-worktree](.agents/skills/trtutils-worktree/SKILL.md) | Set up a runnable PR worktree (PYTHONPATH, data symlinks, .venv-ci) |
| [trtutils-pr-pipeline](.agents/skills/trtutils-pr-pipeline/SKILL.md) | Take a PR through review, fix, simplify, test, report, merge; rebuild stale PRs on main |
| [trtutils-new-model-family](.agents/skills/trtutils-new-model-family/SKILL.md) | Add a model family with API parity to Detector/Classifier |
| [trtutils-accuracy-validation](.agents/skills/trtutils-accuracy-validation/SKILL.md) | Check trtutils outputs against the reference implementation |
| [trtutils-jetson-testing](.agents/skills/trtutils-jetson-testing/SKILL.md) | Run tests/benchmarks on a remote Jetson Orin over SSH |
| [trtutils-benchmarking](.agents/skills/trtutils-benchmarking/SKILL.md) | Make fair branch-vs-main perf comparisons |
| [trtutils-test-style](.agents/skills/trtutils-test-style/SKILL.md) | Write tests per docs/TEST_CODE_STYLE.md |
| [trtutils-docker-ci](.agents/skills/trtutils-docker-ci/SKILL.md) | Run the CUDA 11/12/13 Docker test containers |
| [trtutils-memcheck](.agents/skills/trtutils-memcheck/SKILL.md) | Compare compute-sanitizer memcheck errors against main |
