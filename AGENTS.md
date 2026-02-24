# AGENTS.md

## Cursor Cloud specific instructions

### Overview

`trtutils` is a Python library providing a high-level interface for NVIDIA TensorRT inference. It is a pure Python package (no web servers, databases, or services to run).

### Development environments

Two virtual environments are used:

- **`.venv`** (Python 3.13): Primary dev environment. Created with `uv venv .venv --clear --python 3.13 --seed` and install with `uv pip install -e ".[ci,docs,test]"`.
- **`.venv-ci`** (Python 3.8): Used by CI scripts and pre-commit hooks. Created via `./ci/make_venv.sh` or manually with `uv venv .venv-ci --clear --python 3.8 && source .venv-ci/bin/activate && uv pip install -e ".[ci]" ".[test]"`.

Activate the dev venv before running commands: `source .venv/bin/activate`.

### CI commands (run without GPU)

All CI checks run on standard Ubuntu without an NVIDIA GPU:

- **Format check**: `./ci/run_format.sh --check --diff`
- **Lint check**: `./ci/run_lint.sh --no-fix`
- **Type check**: `./ci/run_type_check.sh`
- **Auto-fix formatting/lint**: `make fix`
- **Build package**: `python -m build`

The CI scripts auto-detect ruff/ty from `.venv-ci` if present, then fall back to system installations.

### Testing caveat

`import trtutils` raises `ImportError` unless `tensorrt` and `cuda-python` are installed (requires NVIDIA GPU + CUDA). This means **all pytest tests fail** in a non-GPU environment. The CI pipeline (ruff, ty, build) is the primary validation path in environments without GPU hardware.

### Key tool versions (pinned in `pyproject.toml`)

- `ruff==0.15.2`
- `ty==0.0.18`
