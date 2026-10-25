---
name: memcheck
description: Use when checking a trtutils change for CUDA memory errors - "run compute-sanitizer", "memcheck this PR", "use-after-free", "invalid device pointer", "check for leaks", after touching Buffer, memory copies, streams or CUDA graphs.
---

# compute-sanitizer memcheck vs main

`main` is not clean: tests that run `Detector` with `cuda_graph=True` already report
use-after-free and `cudaGraphDestroy` / `cudaGraphExecDestroy` invalid-argument errors.
So never read a nonzero count as a regression; compare counts against `main` on the same
tests.

## Steps

1. Have a `main` tree next to the branch (`worktree` skill, or
   `git worktree add <main-wt> origin/main`) sharing the same `data/`.

2. Pick the tests the change touches (memory code: `tests/core/`; image paths:
   `tests/image/test_detector.py`, `tests/image/test_preproc.py`). Build engines once
   without the sanitizer (a normal pytest run) so build time is not sanitized.

3. Run the same command in both trees:

   ```bash
   PYTHONPATH=src compute-sanitizer --tool memcheck --print-limit 0 \
     --log-file memcheck.log \
     <python> -m pytest -q -p no:cacheprovider <tests> > pytest.log 2>&1
   ```

   `compute-sanitizer` is under `/usr/local/cuda/bin/` if it is not on PATH. Use the same
   `<python>` for both trees. Expect 10-100x slowdown; run long sets detached.

4. Count and classify:

   ```bash
   grep -c '^========= [A-Z]' memcheck.log
   grep '^========= [A-Z]' memcheck.log | sed 's/ at .*//' | sort | uniq -c | sort -rn
   ```

   To attribute errors to tests, run per test file or add `-v` and match timestamps.

5. Compare branch vs main per error kind and per test. Only new kinds, or counts that grow
   on tests the change touches, are findings. Check that graph-off tests stay at 0.

6. Optional extra tools: `--tool racecheck` for kernels, `--leak-check full` with
   `--tool memcheck` for leaks at exit.

7. Report: command, test set, error totals by kind for main and branch, and any new error
   with its backtrace.
