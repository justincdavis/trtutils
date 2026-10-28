---
name: benchmarking
description: Use when measuring or comparing trtutils performance - "is this faster", "benchmark branch vs main", "perf regression", "measure latency", "benchmark on Jetson", "lock clocks", before claiming a perf win or loss in a PR.
---

# Fair perf comparisons

A perf claim needs branch and `main` measured under the same conditions, with warm-up
discarded and noise reported.

## Steps

1. Get both sources side by side without switching branches mid-run:
   `git -C <main> worktree add <bench-main> origin/main` (or `git archive origin/main | tar -x`).
   Import each via its own `PYTHONPATH=<tree>/src` and log `trtutils.__file__`.

2. Same conditions: same engine file (or engines built with the same settings and TRT
   version), same images, same batch size and preprocessor, nothing else on the GPU.

3. Measure:
   - warm up (e.g. 20 calls) and discard
   - run interleaved rounds: A, B, A, B, ... (e.g. 10 rounds x 100 calls) so drift hits both
   - report the median ms per call (plus p90 or spread), not the mean or the best
   - time with `time.perf_counter()` around synchronous calls, or read the model's
     `_pre_profile` / `infer_profile` / `_post_profile` for per-stage times

4. On Jetson, lock clocks first. GPU DVFS swings results 1.5-2.5x even in MAXN:

   ```bash
   sudo -n jetson_clocks --store /tmp/clocks.conf
   sudo -n jetson_clocks
   # ... benchmarks ...
   sudo -n jetson_clocks --restore /tmp/clocks.conf
   ```

   Always restore. Report locked numbers (DVFS numbers optional, labeled).

5. Decide: a difference within run-to-run spread (typically ±3%) is noise. Repeat a
   surprising result once before acting on it.

6. Report a table: config, main median, branch median, delta %, spread, machine, TRT version.

## Repo tooling

- `benchmark/run.py` subcommands: `models`, `batch`, `optimize`, `sahi`, `plot`, `table`,
  `tasks`, `compare`, `bootstrap`. Results go under `benchmark/data/`; do not commit ad-hoc result files.
- Profiling: `trtutils.enable_nvtx()` then `nsys profile ...` to see stage ranges.
- A real dynamic-batch yolov10n needs `trtutils download --model yolov10n --dynamic`;
  making a static export's batch dim symbolic collapses back to static in TensorRT.

- `tasks` times one model per task on a 1080p frame (end2end plus preprocess/infer/
  postprocess/extract) and prints medians. Without `--stage` it caches by key in
  `data/tasks/<device>.json`; `--stage NAME` always measures fresh into
  `data/perf-series/<device>/stage-NAME.json` with `trtutils_sha`, `dirty` and `timestamp`.
- `compare A [A ...] --vs B [B ...] [--labels LA LB] [--csv PATH]` matches keys across stage
  or tasks files and prints median A, median B and delta % per key and stage (median of
  per-file medians when a side has several files).

## Rule: GPU commands go through the lock

Every command that touches the GPU is wrapped in `flock ~/.gpu.lock`.

## A/B recipe (two worktrees)

```bash
# 3 alternating rounds per side; A and B are worktrees, run from <tree>/benchmark
for i in 1 2 3; do
  for side in A B; do
    flock ~/.gpu.lock env PYTHONPATH=<tree-$side>/src python run.py tasks --device <dev> \
      --model yolov11n --stage $side-$i
  done
done
python run.py compare data/perf-series/<dev>/stage-A-{1,2,3}.json \
  --vs data/perf-series/<dev>/stage-B-{1,2,3}.json --labels A B --csv ab.csv
```

Check the `dirty` and `trtutils_sha` fields in each file before trusting a side.

## Profiling a tasks run

```bash
flock ~/.gpu.lock env PYTHONPATH=<tree>/src nsys profile -o prof python run.py tasks --device <dev> --nvtx --stage scratch
flock ~/.gpu.lock env PYTHONPATH=<tree>/src python -m cProfile -s tottime run.py tasks --device <dev> --stage scratch
```
