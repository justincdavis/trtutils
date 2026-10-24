---
name: trtutils-new-model-family
description: Use when adding a new model family or task type to trtutils - "add support for <model>", "new ImageModel subclass", "add a segmentation/pose/OCR model", "add download/export for <model>", "add a models/ wrapper". Checklist for API parity, postprocessors, wrappers, download configs, docs, and tests.
---

# Adding a model family

Mirror the existing families; read `image/_detector.py`, `image/_classifier.py` and
`image/_hand_interaction.py` before writing anything. A family that fits an existing task
(e.g. another detector) only needs steps 4-8.

## 1. Interface (`src/trtutils/image/interfaces.py`)

Add `<Task>Interface(ABC)` next to `DetectorInterface` / `ClassifierInterface`: the
`engine`, `name`, `input_shape`, `dtype` properties and abstract `preprocess`, `postprocess`,
`__call__`, `run`, `get_<results>`, `end2end`, with the same `@overload`s
(single image vs list, `postprocess: Literal[True]/[False]`).

## 2. Image model (`src/trtutils/image/_<task>.py`)

`class <Task>(ImageModel, <Task>Interface)`:

- Constructor follows `Detector.__init__`: `engine_path`, `warmup_iterations`,
  `input_range`, `preprocessor`, `resize_method`, thresholds, `mean`, `std`, `dla_core`,
  `device`, `backend`, then keyword-only `warmup`, `pagelocked_mem`, `unified_mem`,
  `cuda_graph`, `no_warn`, `verbose`.
- Public API: `preprocess`, `postprocess`, `run`, `get_<results>`, `end2end`, `__call__`;
  route `run`/`end2end` through `ImageModel._run_core` / `_end2end_core` via a `_post_fn`.
- Per-call `verbose`, `no_copy` and thresholds fall back with
  `x if x is not None else self._x` (never `or`).
- NVTX: add tags to `self._nvtx_tags` and wrap each stage in
  `if FLAGS.NVTX_ENABLED: nvtx.push_range(...)` / `pop_range()`; set `self._post_profile`.
- Log only inside `if verbose:`. Export it from `image/__init__.py`.

## 3. Postprocessor (`src/trtutils/image/postprocessors/_<task>.py`)

Take the engine output list (plus `ratios`, `padding`, thresholds, `no_copy`, `verbose`) and
return a per-image list. Put helpers shared by several families in
`postprocessors/_common.py` (create it if missing) instead of copying. Keep numpy
vectorized; `@register_jit` cores run as plain Python unless numba is installed.

## 4. Model wrappers (`src/trtutils/models/<family>/`)

Follow `models/detectors/_yolo.py`: `class X(<Task>, Model)` with `_model_type`,
`_friendly_name`, `_default_imgsz`, `_input_tensors`, optional `_build_hooks`,
`_valid_imgszs`, `_imgsz_divisor`, `_fp16`; an `__init__` with family defaults that calls
`<Task>.__init__`. Engines with in-graph NMS (data-dependent shapes) default
`cuda_graph=False`. Export from `models/__init__.py`.

## 5. Download / export (`src/trtutils/download/`)

- `configs/<family>.json`: `{"<model-name>": {"url": ..., "name": ...}, ...}`.
- `_models/_<family>.py`: `export_<family>(directory, config, python_path, bin_path, model,
  opset, imgsz, *, no_cache, no_uv_cache, no_warn, verbose)`; import it in
  `_models/__init__.py` and dispatch it in `_download.py::download_model`.
- `model_deps/<family>.txt`: pinned export requirements.
- Verify with `trtutils download --model <name> --output data/<family>/<name>_<imgsz>.onnx`.
  For pip TensorRT pitfalls (plugins, scripted NMS, baked `Split`), read the comments in
  `_models/_hands23.py` and `_models/_hoi_detr.py`.

## 6. Accuracy

Validate against the reference implementation on the same ONNX
(`trtutils-accuracy-validation` skill) before writing tests that pin numbers.

## 7. Tests (`trtutils-test-style` skill)

- Postprocessor unit tests with synthetic outputs (`@pytest.mark.cpu` where no GPU is used).
- GPU tests in `tests/models/` via `build_model_engine`, which skips when the ONNX is missing.
- Mark CUDA graph tests `@pytest.mark.cuda_graph`; output checks `@pytest.mark.correctness`.

## 8. Docs

`CHANGELOG.rst` (`Unreleased`), `docs/models.rst`, an example in `examples/`, API docs in
`docs/`. Then `make fix && make typecheck` and the full suite.
