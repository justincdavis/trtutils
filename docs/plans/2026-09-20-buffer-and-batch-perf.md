# Unified Buffer + batch / resolution perf port from dnnkit

**Source:** `justincdavis/dnnkit` branch `perf/io-module` (24 commits, superset of
`perf/fable_plan` and `perf/nsys-cpu-batch`). dnnkit `main` is a squashed
trtutils snapshot from 2026-07-24; all work is on that branch. Commit bodies and
`patches/*.patch` there carry the measurements quoted below (GB10, integrated).

**Target:** trtutils `origin/main` @ `bf8edcf`. The `#204` run/end2end dedup means
`_image_model.py` / `_detector.py` hunks do not apply textually; they are
re-derived against `_run_core` / `_end2end_core` / `_engine_inputs`.

**Hardware for verification:** RTX 5080 (discrete, TRT 10.15.1.29, numpy 2.4.2).
`FLAGS.INTEGRATED_GPU` paths are ported but cannot be measured here.

**Not ported:** `trtutils.io` (nvImageCodec / PyNvVideoCodec readers, `tests/io`).
It is a feature, not a perf change, and adds two optional deps. Follow-up if wanted.

---

## What dnnkit changed, mapped to PRs

| dnnkit commit | Change | PR |
|---|---|---|
| `dc73faf` `834ee79` `fb1fce2` | `core.Buffer` + `MemoryLocation`; `Binding` composes two Buffers (`upload`/`download`, `from_buffers`, `_partial`); pitched 2-D + strided N-D memcpy | 1 |
| `f5fc4f7` (kernels) | `KernelArgs` ndarray subclass owns its intermediates; fixes use-after-free when args are cached per batch size | 1 |
| `4dcd32d` (stream) | `create_event` / `record_event` / `stream_wait_event` / `event_synchronize` | 1 |
| `52441ed` `856facf` `062c6d9` `559aa70` | `download --batch/--dynamic`; benchmark input compositions (single / homo / het); per-batch static engines; `patch_series.py` plot | 2 |
| `379dc37` `42f8842` (engine, builder) `f5fc4f7` (engine) | `_apply_input_shapes` / `_resolve_dynamic_batch`; execute + `direct_exec` return the valid prefix; `build_engine(shapes=[(name, (min,opt,max))])` | 3 |
| `db18ed2` `50388bf` | `preprocess(..., out=)` single-pass HWC→CHW + fused fp32 affine; CPU grow-only batch tensor above 16 MB; `ImageModel.preprocess` honors `no_copy` on CPU | 4 |
| `4dcd32d` `f17659c` `c5687d3` `775cbe2` `42f8842` (preproc) `da6cf27` (preproc) | per-shape pinned staging pool + copy stream, event-gated reuse; grow-only batch input + SST buffers; SST arg cache keyed by batch; threaded pack; integrated-GPU direct DMA | 5 |
| `85a362c` | TRT preprocess engine built with a 1..B profile and resolved to the submitted batch | 6 |
| `42f8842` `da6cf27` `f5fc4f7` (image model / detector) | graphs cached per `(batch, input ptrs)`; dims no longer locked on first call; `_validate_batch_size` on every path; direct-GPU `run()` for single-input schemas; `_resolve_dynamic_batch` before `direct_exec` | 7 |
| `d75c434` | `rescale_dets.cu` in-graph un-letterbox; `Detector(postprocessor='cuda')` | 8 |

dnnkit's own patch-series plot (14 stages, GB10) shows batch-1 latency flat
across the series; the wins are: CPU preproc batch≥8 (271→621 img/s), TRT preproc
batch 1 (355→831), CUDA homogeneous batches (772→1199), all against a dynamic
engine. On the 5080 the discrete-PCIe staging paths are the ones that matter.

---

## Improvements beyond the port

- **Resolution switching (PR 5).** `_create_resize_args` caches one
  `(h, w, method)`; alternating 720p/1080p misses every frame and re-packs
  kernel args. Make it a dict (bounded, cleared at 64). The single-image path
  reallocates `_input_binding` on every shape change (`_validate_input` →
  `_reallocate_input`); route it through the staging pool so a shape seen
  before reuses its slot.
- **Resolution switching (PR 7).** `_end2end_graph_core` currently raises
  `RuntimeError` when image dims change after the first call. With graphs keyed
  on `(batch, ptrs)` and pool slots stable per shape this becomes a cache hit.
  Add a `resolution-switch` composition to the benchmark grid to prove it.
- **Batch-1 (PR 6, PR 7).** TRT preproc at submitted batch, and the direct-GPU
  `run()` path removes a D2H+H2D round trip when `run()` (not `end2end`) is
  used with a GPU preprocessor.
- **Batch ≥32 (PR 4, PR 5, PR 8).** CPU batch tensor reuse; pinned staging with
  copy/compute overlap; optional in-graph rescale.
- Threaded pack (`ThreadPoolExecutor`, PR 5) and the CUDA rescale (PR 8) are
  **measured before merge**: dropped if inside noise on the 5080.

---

## PR series

Branches `perf/NN-<name>`, stacked where noted. Each PR: `make fix && make
typecheck`, `pytest tests/core tests/engine tests/image tests/models` on the
5080, CHANGELOG entry, and (for 3–8) benchmark numbers in the PR body.

### PR 1 — `perf/01-buffer` core: Buffer, Binding on Buffer, N-D memcpy, events, KernelArgs
Base: `origin/main`. Pure refactor; no perf claim.
- `src/trtutils/core/_buffer.py` (new, from dnnkit): `MemoryLocation`, `Buffer`
  (`empty`/`from_array`/`from_ptr(owner=)`, `__getitem__` leading-axis views,
  `reshape`, `copy_to`/`copy_from` dispatch on location, `numpy`, `free`,
  `__cuda_array_interface__`, `__array__`).
- `src/trtutils/core/_bindings.py`: `Binding(host: Buffer, device: Buffer, ...)`,
  `from_buffers`, `allocation`/`host_allocation` properties, `_partial`,
  `upload(data, stream, shape)`, `download(stream, shape)`. `create_binding` /
  `allocate_bindings` signatures unchanged.
- `src/trtutils/core/_memory.py`: `memcpy_device_to_host_async(nbytes=)`,
  `memcpy_2d[_async]`, `memcpy_nd_{host_to_device,device_to_host}[_async]`.
- `src/trtutils/core/_stream.py`: event helpers.
- `src/trtutils/core/_kernels.py`: `KernelArgs` keepalive.
- `src/trtutils/core/__init__.py` exports.
- Adopt in call sites that only swap `create_binding` for `Buffer.from_array`
  (`_image_preproc.py` mean/std/orig_size/scale_factor, `_trt.py` scale/offset,
  `_cuda.py` `.ptr`), and `output_binding.download(stream)` in
  `GPUImagePreprocessor.preprocess`. No behavior change.
- Tests: `tests/core/test_buffer.py` (new: constructors, views, copies in all
  four directions, free/owning semantics, unified alias), port
  `tests/core/test_buffer_nd.py`, `test_bindings.py` additions, `test_kernels.py`
  regression tests. Existing `tests/engine` must pass unchanged.

### PR 2 — `perf/02-bench-batch` download `--batch/--dynamic` + benchmark compositions
Base: `origin/main` (parallel with PR 1).
- `download/_tools.py` `handle_batch` / `handle_dynamic`; `_download.py`,
  `__main__.py` flags; per-family exporters (`_yolo.py`, `_ultralytics.py`,
  `_torchvision.py` honor; others raise `NotImplementedError`).
- `benchmark/utils/runners.py`: drop the `yolo` CLI shell-out for the download
  tool; `ensure_dynamic_engine`; `benchmark_optimizations` gets an `inputs` axis
  `single / homogeneous8 / heterogeneous8 / resolution-switch` (alternating
  720p/1080p singles) and a static engine per batch; `benchmark_batch` gains
  `--batch-sizes 1 2 4 8 16 32`.
- `benchmark/plotting/patch_series.py` (from dnnkit) for the per-PR plot.
- Tests: `tests/download` for `handle_batch` / `handle_dynamic`.
- **Baseline run on `origin/main`** recorded as `benchmark/data/perf-series/stage-00.json`
  before PR 3 starts; every later PR appends a stage.

### PR 3 — `perf/03-dynamic-engine` engine runs dynamic-batch engines at the submitted shape
Base: PR 1.
- `core/_interface.py`: `_engine_input_shapes`, `_dynamic_input_names`;
  `batch_size` / `is_dynamic_batch` read engine shapes, not max-profile
  bindings.
- `_engine.py`: `_apply_input_shapes`, `_refresh_output_sizes`,
  `_resolve_dynamic_batch`; `execute` uses `Binding.upload/download` with the
  active shape, skips the CUDA graph at partial shapes, returns prefix views;
  `direct_exec` returns the resolved prefix; `_set_input_bindings` resets tracking.
- `builder/_build.py`: `(min, opt, max)` triple in `shapes`.
- Tests: `tests/engine/conftest.py` `_build_dynamic_test_engine` (symbolic batch
  on `simple.onnx`), `TestDynamicBatchDirectExec`, execute at 1/2/4 returns
  correctly shaped outputs and matches a static engine bit-for-bit,
  `tests/builder` triple-profile build.

### PR 4 — `perf/04-cpu-preproc` CPU preprocess: single-pass pack, fused fp32 affine, batch tensor reuse
Base: PR 1.
- `preprocessors/_process.py`: rewrite `preprocess` (no `_preprocess_single`,
  `out=` param, explicit empty-list `ValueError`, method validated up front).
- `preprocessors/_cpu.py`: `_resolve_batch_buffer` grow-only host `Buffer`
  above `_REUSE_MIN_BYTES`; `no_copy` semantics documented and honored.
- `_image_model.py::preprocess`: pass `no_copy` to every preprocessor.
- Tests: `tests/image/test_preproc.py` — output bit-identical to the pre-PR
  implementation for scale-only and mean/std, uint8 identity range, `no_copy`
  returns a view that the next call overwrites, empty list raises.
- Numbers: CPU preproc `end2end` at batch 8/16/32.

### PR 5 — `perf/05-gpu-staging` GPU preprocess: staging pool, copy stream, grow-only buffers, resolution cache
Base: PR 1 (rebased on 3+4 before merge only if needed).
- `_image_preproc.py`: `_StagingSlot` pool + `_copy_stream` + event gating;
  `_reallocate_batch_input` high-water-mark; `_batch_input_view`; pinned
  staging in `_resize_single_image_to_batch`; heterogeneous path two-phase
  (upload all on copy stream, then kernels); `_create_resize_args` dict cache;
  single-image path takes a pool slot instead of reallocating `_input_binding`.
- `_cuda.py`: `_allocated_sst_batch` grow-only, `_cached_sst_args: dict[int, ...]`,
  `stream_synchronize` before growth.
- `core/_device.py::is_integrated`, `FLAGS.INTEGRATED_GPU`, direct-DMA branch
  (guarded; unmeasurable on 5080, keep for Jetson/GB10).
- Threaded pack: implement, measure homo8/homo32 on the 5080; keep only if it
  clears run-to-run noise, otherwise delete before opening the PR.
- Tests: `tests/image/test_preproc.py` — heterogeneous batch equals per-image
  singles; alternating 480p/720p/1080p singles equal a fresh preprocessor's
  output each time (no stale slot); pool bounded at `_MAX_STAGING_SLOTS`;
  `test_kernels` regression already covers the arg-cache lifetime.
- Numbers: `optimize` grid, all compositions, cuda + trt preprocessors.

### PR 6 — `perf/06-trt-preproc-batch` TRT preprocessor at the submitted batch
Base: PR 3.
- `image/onnx_models.py`: `(1, B, B)` profile, `dyn` cache suffix.
- `_trt.py`: `self._engine._resolve_dynamic_batch(batch_size)` before `raw_exec`.
- Tests: TRT preproc batch 1 vs batch B output identical to CUDA preproc;
  `tests/image/onnx/test_onnx_models.py` profile assertion.
- Numbers: batch 1 with `preprocessor='trt'` (the default) — the headline
  batch-1 win.

### PR 7 — `perf/07-e2e-variable-batch` variable batch + resolution in end2end, direct-GPU run
Base: PRs 3, 5, 6.
- `_image_model.py`: `_e2e_graphs: dict[(batch, ptrs), CUDAGraph]` bounded at
  32; delete `_e2e_input_dims` / `_e2e_batch_size` locking; `_validate_batch_size`;
  `_copy_engine_outputs(output_shapes)` via `Binding.download`;
  `_stage_graphed_postprocess` / `_capture_graphed_postprocess` no-op hooks;
  `_run_core` direct-GPU branch when `self._single_input_schema` (base `True`,
  Detector: not RT-DETR-V3 / image-size / scale-factor) and GPU preprocessor;
  `_end2end_core` GPU branch calls `_validate_batch_size` +
  `_resolve_dynamic_batch` before `direct_exec`.
- `_detector.py`: `_single_input_schema` override only.
- Tests: `tests/image/test_image_model.py` / `test_detector.py` — end2end with
  alternating image sizes returns the same detections as fresh models; static
  engine rejects batch≠B with `RuntimeError` on both graph and non-graph paths;
  dynamic yolov10n engine (built via PR 2 tooling, skipped if the ONNX is
  missing) over `8,2,8,4,1,8`; `run()` direct path equals host path.
- Numbers: `resolution-switch` composition; dynamic-engine batch sweep.

### PR 8 — `perf/08-cuda-rescale` in-graph CUDA rescale, `Detector(postprocessor='cuda')`
Base: PR 7.
- `image/_kernels/rescale_dets.cu` (new); delete the empty `scale_eff_nms.cu` /
  `scale_v10.cu` stubs; `kernels.py::RESCALE_DETECTIONS`.
- `_detector.py`: `postprocessor` arg, `_pp_*` state, transforms binding,
  `_stage_graphed_postprocess` / `_capture_graphed_postprocess`.
- Tests: `postprocessor='cuda'` detections equal `'cpu'` for YOLO_V10 and
  EfficientNMS engines; unsupported schema warns and falls back.
- Numbers: batch 32 cuda vs cpu postprocessor. Ship only if it wins.

---

## Verification protocol (5080)

Model: yolov10n @ 640 (existing `data/yolov10/yolov10n_640.onnx`), plus yolov8n
(EfficientNMS) for PR 8. Engines: static b1/b8/b32 and dynamic 1–32, built once
under `data/`, fp16, opt level 1, shared timing cache.

Per stage (`origin/main`, then each PR tip): `benchmark/run.py optimize --device 5080`
grid (preprocessor × cuda_graph × composition) and `benchmark/run.py batch
--batch-sizes 1 2 4 8 16 32`, 100 warmup / 500 iters, written to
`benchmark/data/perf-series/stage-NN.json`, plotted with `patch_series.py`. Raw
`tensorrt` / `tensorrt(graph)` modes are the noise control and must stay flat.
Correctness gate per stage: detections on `horse.jpg` / `people.jpeg` identical
to stage 00 for every preprocessor and composition.

## Execution

Sonnet subagents implement one PR each from this file plus the dnnkit diff
(`git -C /home/jcdavis/dnnkit diff main...origin/perf/io-module -- <paths>`);
the orchestrator reviews each diff, runs the benchmark stage, and opens the PR.
PR 1 ∥ PR 2 first; PR 3 ∥ PR 4 ∥ PR 5 next; PR 6, 7, 8 sequential.
