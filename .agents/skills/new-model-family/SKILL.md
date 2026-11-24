---
name: new-model-family
description: Use when adding a new model family or task type to trtutils - "add support for <model>", "new ImageModel subclass", "add a segmentation/pose/OCR model", "add download/export for <model>", "add a models/ wrapper". Checklist for API parity, postprocessors, wrappers, download configs, docs, and tests.
---

# Adding a model family

Mirror the existing families. For a new task type read `image/_depth_estimator.py` (the
smallest task: no thresholds, CPU postprocess only), then `image/_classifier.py` and
`image/_detector.py`. A family that fits an existing task (e.g. another detector) only needs
steps 4-8.

## 0. Feasibility first

Before writing library code: confirm the weight license and source, export the model to
ONNX, build it in FP32 and compare with onnxruntime. If the export or the TensorRT build
needs more than a small patch, stop and report instead of forcing it.

## 1. Interface, schema, types

- `image/interfaces.py`: `<Task>Interface(ABC)` next to `DetectorInterface` /
  `ClassifierInterface`: the `engine`, `name`, `input_shape`, `dtype` properties and abstract
  `preprocess`, `postprocess`, `__call__`, `run`, `get_<results>`, `end2end`, with the same
  `@overload`s (single image vs list, `postprocess: Literal[True]/[False]`).
- `image/_schema.py`: a `<Task>OutputSchema` enum whose members are output tensor names
  (TensorRT 8.6 reorders outputs, so dispatch is by name, never by index, `isinstance` or
  flags). A `("*",)` member is a catch-all tried last.
- `image/types.py`: the result `TypeAlias` and its `__all__` entry.

## 2. Image model (`src/trtutils/image/_<task>.py`)

`class <Task>(ImageModel, <Task>Interface)`, a peer of the other tasks (never subclass
another task model or add a private base):

- Constructor follows `Detector.__init__`: `engine_path`, `warmup_iterations`,
  `input_range`, `preprocessor`, `resize_method`, thresholds, `mean`, `std`, `dla_core`,
  `device`, `backend`, then keyword-only `warmup`, `pagelocked_mem`, `unified_mem`,
  `cuda_graph`, `postprocessor`, `cpu_resize`, `no_warn`, `verbose`. The benchmark passes
  `postprocessor=` and reads `_CUDA_POSTPROCESSOR` on every task class.
- Public API: `preprocess`, `postprocess`, `run`, `get_<results>`, `end2end`, `__call__`;
  route `run`/`end2end` through `ImageModel._run_core` / `_end2end_core` via a `_post_fn`.
- Per-call `verbose`, `no_copy` and thresholds fall back with
  `x if x is not None else self._x` (never `or`).
- `end2end` skips the output copies: anything it returns must be a new array, not a view of
  engine memory the next inference overwrites.
- NVTX: add tags to `self._nvtx_tags` and wrap each stage in
  `if FLAGS.NVTX_ENABLED: nvtx.push_range(...)` / `pop_range()`; set `self._post_profile`.
- Log only inside `if verbose:`. Export it from `image/__init__.py`.
- A model with several engines (detector + recognizer, encoder + decoder) is a plain
  composite of its parts, like `OCR` in `image/_ocr.py`; an engine without an image input
  cannot be an `ImageModel`.

## 3. Postprocessor (`src/trtutils/image/postprocessors/_cpu_<task>.py`)

Take the engine output list (plus `ratios`, `padding`, thresholds, `no_copy`, `verbose`) and
return a per-image list; a `get_<results>` function turns that into results. Export both
from `postprocessors/__init__.py`. Put helpers shared by several families in
`postprocessors/_common.py` instead of copying. Keep numpy vectorized; `@register_jit` cores
run as plain Python unless numba is installed. A CUDA twin (`_cuda_<task>.py`, a kernel in
`_kernels/`) is optional: `DepthEstimator` has none.

## 4. Model wrappers (`src/trtutils/models/<family>/`)

Follow `models/detectors/_yolo.py`: `class X(<Task>, Model)` with `_model_type` (the config
file stem and the `data/` directory name), `_friendly_name`, `_default_imgsz`,
`_input_tensors`, optional `_build_hooks`, `_valid_imgszs`, `_imgsz_divisor`,
`_model_imgszs`, `_fp16`; an `__init__` with family defaults that calls `<Task>.__init__`.
Wrappers hold defaults only, no inference logic. Engines with in-graph NMS (data-dependent
shapes) default `cuda_graph=False`. Export from the family package and `models/__init__.py`.

## 5. Download / export (`src/trtutils/download/`)

- `configs/<family>.json`: a `"_defaults"` object shared by the file's entries, then one
  entry per model. Every entry needs `exporter` (a key of `EXPORTERS`) and `license`
  (entry keys override `_defaults`). Model names are unique across all configs.
- License: check the upstream LICENSE and model cards for the WEIGHTS, not only the code.
  Sizes of one family can differ, upstream can relicense after the commit an exporter pins,
  and some tiers need an account. Use the SPDX id, the upstream name for custom licenses
  (with `non-commercial` in it when that is what it says, so the notice is the loud one),
  or `"unknown"` when upstream states none. Never guess.
- `_models/_<family>.py`: `export_<family>(directory, config, python_path, bin_path, model,
  opset, imgsz, *, no_cache, no_uv_cache, no_warn, verbose)`; register it in `EXPORTERS`
  in `_models/__init__.py` with its capability flags. There is no dispatch code to edit.
- `model_deps/<family>.txt`: pinned export requirements. Pin the upstream repo commit too.
- Export with the legacy exporter (`dynamo=False`) at opset 17 under
  `torch.set_grad_enabled(False)`: newer opsets do not build on TensorRT 8.6.
- Verify with `trtutils download --model <name> --output data/<family>/<name>_<imgsz>.onnx`.
  For pip TensorRT pitfalls (plugins, scripted NMS, baked `Split`), read the comments in
  `_models/_hands23.py` and `_models/_hoi_detr.py`.

## 6. Accuracy

Validate against the reference implementation on the same ONNX
(`accuracy-validation` skill) before writing tests that pin numbers, FP32 first, then the
default build.

- FP16 can fail silently or with NaN: transformer attention logits, residual streams and
  hand-written layer norms (`Pow` of a large value) exceed the FP16 range. Find the tensor
  with onnxruntime, then prefer an export-side fix (e.g. export the norm as
  `LayerNormalization`) over pinning layers; precision hooks live in `models/_hooks.py`.
- A hook that matches layers by name returns no overrides when nothing matches. Test the
  number of layers it pins on the exported ONNX.
- The `linear` and `letterbox` resizes do not antialias; expect a gap against upstream
  pipelines that do when images are downscaled a lot, and report it separately from the
  TensorRT-vs-onnxruntime numbers.

## 7. Tests (`test-style` skill)

- Postprocessor unit tests with synthetic outputs (`@pytest.mark.cpu` where no GPU is used).
  CI runs the FIXED file list in `ci/run_cpu_tests.sh` and holds `trtutils/image` branch
  coverage with zero tolerance: add a new CPU test file to that list (files under
  `tests/image/preprocessors/` are picked up by directory) and run `make test-cov-cpu`.
- GPU tests in `tests/models/` via `build_model_engine`, which skips when the ONNX is
  missing and reads the size from the `<model>_<imgsz>.onnx` filename. Engines are cached in
  `data/engines` by ONNX filename, not content: after re-exporting a model under the same
  name, delete its stale engines.
- `tests/models/test_model.py` runs over every `Model` subclass automatically.
- Hand-kept lists to extend: `tests/download/test_config.py` (`_EXPECTED_FAMILIES`,
  `_EXPECTED_MODELS`), `tests/download/conftest.py` (`ROUTING_CASES`),
  `tests/download/test_exporters_network.py` (`EXPORT_CASES`, opt-in with `-m download`),
  and for a new task `tests/image/test_schema.py` / `test_types.py`.
- Mark CUDA graph tests `@pytest.mark.cuda_graph`; output checks `@pytest.mark.correctness`.

## 8. Docs and registration

`CHANGELOG.rst` (`Unreleased`), `docs/models.rst` (the task section, the download list and
the license table), `README.md`, an example in `examples/image/` with its
`docs/examples/<name>.rst` and a line in `docs/examples.rst`. A new task type also goes in
`docs/index.rst` and in `TASKS` of `benchmark/utils/tasks.py`. Non-`.py` files under
`src/trtutils` need `MANIFEST.in` and `pyproject.toml` package-data entries. Then
`make fix && make typecheck` and the full suite.
