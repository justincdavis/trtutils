# Research Submodule Developer Guide

How to add a new research submodule to `trtutils.research`. Each submodule
implements a research paper and plugs into trtutils' build system, CLI, and
lazy-loading infrastructure. See `axonn/` as the canonical reference.

---

## Naming

- Lowercase paper/method name: `axonn`, `layer_fusion`
- Must be a valid Python identifier (no hyphens, no leading digits)
- Name becomes the CLI subcommand and import path

---

## Required Files

Every submodule needs at minimum:

```
src/trtutils/research/<name>/
    __init__.py     # public API, __all__, module docstring with paper reference
    _cli.py         # CLI registration
    # ... other implementation files (underscore-prefixed for private)
```

The `__init__.py` must export public functions via `__all__` and include a
module docstring referencing the paper (title, venue, DOI).

---

## Registration Checklist

### 1. Create the submodule directory

Create `src/trtutils/research/<name>/` with `__init__.py` and `_cli.py`.

### 2. Register in `src/trtutils/research/__init__.py`

Four changes:

- Add to `_SUBMODULES` list
- Add to `_LAZY_SUBMODULES` set
- Add to `__all__`
- Add a `TYPE_CHECKING` import: `from . import <name> as <name>`

### 3. Add optional dependency extra in `pyproject.toml`

```toml
[project.optional-dependencies]
<name> = ["some-dep>=x.y.z"]
```

Then add `"trtutils[<name>]"` to the `dev` extra.

### 4. Create `_cli.py` with `register_cli()`

See [CLI Integration](#cli-integration) below.

### 5. Create tests

```
tests/research/<name>/
    __init__.py
    test_<name>.py
```

Unit test dataclasses, utilities, and solvers with synthetic data. Integration
tests requiring GPU/DLA are run manually on target hardware.

---

## CLI Integration

### `register_cli()` contract

```python
def register_cli(
    subparsers: argparse._SubParsersAction,
    parents: list[argparse.ArgumentParser],
) -> None:
```

Create a parser via `subparsers.add_parser("<name>")`, add sub-subparsers for
each command, and set the default action to print help. Use `trtutils._log.LOG`
for all output.

### Parent parsers

Passed as a list by index:

| Index | Parser | Key Arguments | Use When |
|-------|--------|---------------|----------|
| 0 | `general_parser` | `--log_level`, `--verbose`, `--nvtx` | Always |
| 1 | `dla_parser` | `--dla_core` | DLA acceleration |
| 2 | `build_common_parser` | `--onnx`, `--output`, `--workspace`, `--optimization_level`, `--shape`, `--timing_cache`, etc. | Building engines from ONNX |
| 3 | `calibration_parser` | `--calibration_dir`, `--input_shape`, `--input_dtype`, `--batch_size`, etc. | INT8 calibration |

Only include the parsers your subcommand needs in `parents=`.

---

## Optional Dependencies

- Extra name matches module name in `pyproject.toml`
- Missing dependencies cause `ImportError` at import time, which
  `discover_submodules()` silently suppresses — no custom error handling needed

---

## Core Module Reference

### Core

| Module | Import Path | Description |
|--------|-------------|-------------|
| Engine | `trtutils._engine` | `TRTEngine` for loading/executing engines |
| Core CUDA | `trtutils.core` | Memory ops, streams, CUDA graphs, `Binding` wrappers |
| Flags | `trtutils._flags` | `TENSORRT_VERSION`, `IS_JETSON`, execution API detection |
| Log | `trtutils._log` | `LOG` logger instance |

### Submodules

| Module | Import Path | Description |
|--------|-------------|-------------|
| Builder | `trtutils.builder` | ONNX-to-TensorRT engine building, DLA support, calibration |
| Image | `trtutils.image` | Image model abstractions, pre/postprocessing, CUDA kernels |
| Jetson | `trtutils.jetson` | Jetson-specific profiling and benchmarking with power metrics |
| Profiling | `trtutils.profiling` | Engine profiling, per-layer timing, quantization analysis |
| Inspect | `trtutils.inspect` | Engine inspection and metadata extraction |
| Parallel | `trtutils.parallel` | Multi-engine parallel inference |
| Benchmark | `trtutils._benchmark` | Performance measurement dataclasses |
