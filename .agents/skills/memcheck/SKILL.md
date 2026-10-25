---
name: memcheck
description: Use when checking a trtutils change for CUDA memory errors - "run compute-sanitizer", "memcheck this PR", "use-after-free", "invalid device pointer", "check for leaks", after touching Buffer, memory copies, streams or CUDA graphs.
---

# compute-sanitizer memcheck

`make memcheck` (`ci/run_memcheck.sh`) runs pytest under `compute-sanitizer --tool memcheck`,
prints the error total and a count per error kind, and exits 1 on any error.

## Steps

1. Build engines once without the sanitizer (a normal pytest run on the same tests), so engine
   builds are not sanitized.

2. Pick the tests the change touches: memory code `tests/core/` (the default), image paths
   `tests/image/test_detector.py` / `tests/image/test_preproc.py`. Then:

   ```bash
   PYTHONPATH=src make memcheck MEMCHECK_ARGS="tests/core/test_graph.py tests/image/test_preproc.py"
   # or directly: PYTHONPATH=src ci/run_memcheck.sh <pytest args>
   ```

   `PYTHON=<interpreter>` picks the python, `MEMCHECK_LOG=<path>` the log file (default
   `memcheck.log`). Expect a 10-100x slowdown; run long sets detached.

3. If the branch reports errors, run the same command on a `main` tree with the same `data/`
   (`worktree` skill) before calling them a regression. Only new error kinds, or counts that
   grow on tests the change touches, are findings.

4. Attribute errors to tests by running per test file, or read the backtraces in the log.

5. Optional extra tools, run by hand: `--tool racecheck` for kernels, `--leak-check full`
   with `--tool memcheck` for leaks at exit.

6. Report: command, test set, error totals by kind (branch and, if nonzero, main), and any new
   error with its backtrace.

## Baseline

Since #244, `tests/core/` and the cuda_graph engine tests report 0 errors on `main`. Before it,
every engine deleted with `cuda_graph=True` logged `cudaGraphDestroy` /
`cudaGraphExecDestroy` invalid-argument errors.
