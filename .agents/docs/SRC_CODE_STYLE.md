# Source Code Standards

Standards and conventions for `src/trtutils/`. All new and modified library code must follow
these rules. Tests follow [TEST_CODE_STYLE.md](TEST_CODE_STYLE.md).

## File Structure

```python
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""One-line summary of what the module provides."""

from __future__ import annotations
```

- Copyright header on every Python file (`ci/check_copyright.sh` pre-commit hook).
- `from __future__ import annotations` is always the first import.
- Module docstring on modules that export public APIs.
- All imports at file level; no lazy imports inside functions. Type-only imports go in an
  `if TYPE_CHECKING:` block.

## Tooling

- **Line length**: 101 (`ruff.toml`).
- **Target Python**: 3.8; no syntax or stdlib features newer than 3.8.
- **Linter**: ruff (`ruff.toml`); **type checker**: ty, not mypy (`ty.toml`).
- Run `make fix` then `make typecheck` after every change.

## Simplicity

- Write code as simply as possible. No heavy branching, no speculative options, no
  abstractions with a single user.
- Minimal exception handling: no try/except unless the error must be handled.
- Errors are raised with a named message:

  ```python
  if not host_arr.flags.c_contiguous:
      err_msg = "host_arr must be C-contiguous to copy into"
      raise ValueError(err_msg)
  ```

- Comments only where names and flow cannot explain; lowercase, short, casual.

## Logging

- No prints, ever. Use `trtutils.LOG`.
- Log only inside `if verbose:` blocks, or for warnings/errors around exceptions we handle.
- Prefix model logs with the instance tag: `LOG.debug(f"{self._tag}: postprocess")`.

## Defaults and Overrides

- Per-call overrides fall back to the constructor value with `is not None`, never `or`
  (`0.0` and `False` are valid values):

  ```python
  conf_thres = conf_thres if conf_thres is not None else self._conf_thres
  ```

## Documentation

- NumPy docstrings on every public function, method and class; re-check them whenever a
  signature changes (Parameters, Returns, Raises).
- User-facing changes get a `CHANGELOG.rst` entry under `Unreleased`. Test-only and internal
  tooling changes do not.

## Image Models (`src/trtutils/image/`)

Every task model is a peer subclass of `ImageModel` plus its interface in `interfaces.py`
(`Detector`, `Classifier`, `DepthEstimator`, ...). Do not subclass another task model, and do
not add private intermediate bases between `ImageModel` and a task model.

Result types are `TypeAlias`es defined once in `image/types.py` and used in all signatures.

Each task model provides the same API surface, in this order:

- `__init__(engine_path, warmup_iterations, input_range, preprocessor, resize_method, <task
  thresholds>, mean, std, <schema overrides>, dla_core, device, backend, *, warmup,
  pagelocked_mem, unified_mem, cuda_graph, no_warn, verbose)`
- `preprocess` (inherited), `postprocess`, `run`, `__call__`, `get_<results>`, `end2end`.
- `run` / `__call__` / `get_*` / `end2end` have `@overload`s for single-image vs batch
  inputs: a single image returns that image's result, a list returns a list of results.
- `run(..., preprocessed=, postprocess=, no_copy=, verbose=)` keyword semantics match
  `Detector`.
- Per-call thresholds on `postprocess` / `end2end` follow
  `(inputs, conf_thres, nms_iou_thres, *, <task-specific thresholds>, verbose)`; the shared
  thresholds are positional, task-specific ones (`mask_thres`, `pair_thres`, ...) keyword-only.

### Schemas

- A task model selects its postprocess function from an output schema enum, never from
  `isinstance` checks or flags. Each task has its own enum in `_schema.py` (members are the
  engine output names), and the model dispatches through a small table.
- Schemas are auto-detected from the engine's output names (order-insensitive: TensorRT 8.6
  reorders outputs) and can be overridden with an enum member or its name. A model on an engine
  that matches none of its schemas raises a `ValueError` naming the valid options.

### NVTX and Profiling

- Register tags in `self._nvtx_tags` as `"<prefix>_<method>": f"<task>::<method> [{self._tag}]"`
  and wrap `postprocess` and `get_*` bodies:

  ```python
  if FLAGS.NVTX_ENABLED:
      nvtx.push_range(self._nvtx_tags["cls_postprocess"])
  ...
  if FLAGS.NVTX_ENABLED:
      nvtx.pop_range()  # postprocess
  ```

- Record stage timings in `_pre_profile` / `_infer_profile` / `_post_profile` as `ImageModel`
  does.

### Pre/Postprocessing Functions

- Model-independent pre/postprocessing lives as plain functions in `preprocessors/` and
  `postprocessors/`, exported from their `__init__.py`. Model classes stay thin wrappers.
- Postprocess functions return per-image lists of numpy arrays; `get_*` turns them into Python
  results (tuples of boxes, scores, class ids, ...).

## Model Wrappers (`src/trtutils/models/`)

- Pre-configured wrappers subclass a task model plus the `Model` mixin (`_model.py`) and
  configure `download()` / `build()` through class attributes (`_model_type`,
  `_default_imgsz`, ...). They add no inference logic.
