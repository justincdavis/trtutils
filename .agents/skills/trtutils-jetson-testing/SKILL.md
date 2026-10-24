---
name: trtutils-jetson-testing
description: Use when a trtutils change must be tested or benchmarked on a remote Jetson (AGX Orin, JetPack 6, TensorRT 8.6) - "run the suite on the Orin", "test on Jetson", "check TRT 8.6", "benchmark on the Orin", "remote Jetson run".
---

# Remote Jetson Orin testing

The Orin runs JetPack 6 with TensorRT 8.6 and an integrated GPU, so it catches TRT 8.6 API
differences, output reordering and unified-memory paths the desktop GPU does not.

## Rules

- Credentials come from the environment only: `$ORIN_HOST`, `$ORIN_USER`, `$ORIN_PW`.
  Never hardcode, log, or commit them, or the host name/IP.
- Never touch the user's own checkouts on the Orin. Make your own worktree.

## Steps

1. Write a small paramiko helper in your scratch directory (not the repo):

   ```python
   import os, sys, paramiko
   c = paramiko.SSHClient()
   c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
   c.connect(os.environ["ORIN_HOST"], username=os.environ["ORIN_USER"],
             password=os.environ["ORIN_PW"])
   _, out, err = c.exec_command(sys.argv[1], timeout=600)
   print(out.read().decode(), err.read().decode())
   ```

   Use `c.open_sftp().put(local, remote)` for file copies.

2. Create a worktree on the Orin from its trtutils clone (`git fetch origin` there first),
   the same way as the `trtutils-worktree` skill: `PYTHONPATH=src` with the Orin's existing
   python env, and data entries symlinked from that clone's `data/`.

3. Copy the ONNX files the tests need via SFTP into a real `data/<family>/` directory of your
   worktree. The Orin's own `data/` may hold older exports; use the desktop's files so both
   machines test the same models.

4. Start long runs detached, because the SSH channel hangs on long commands:

   ```bash
   cd <orin-wt> && setsid nohup env PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests \
     > pytest.log 2>&1 < /dev/null &
   ```

   Poll with `tail -n 5 <orin-wt>/pytest.log` until the summary line appears. First runs
   build engines for TRT 8.6 and take much longer.

5. Before calling a failure real: rerun it alone; known flakes include the jetson tegrastats
   benchmark test (`tests/jetson/test_benchmark.py`). Compare against `main` on the Orin
   with the same data.

6. Benchmarks: lock clocks first (`trtutils-benchmarking` skill).

7. Clean up when done: remove the Orin worktree and any copied files outside it.

## TRT 8.6 notes

- `FLAGS.TRT_10` is True on 8.6 (it marks the I/O tensor API); gate on `FLAGS.TRT_VERSION`.
- Engine outputs may be reordered vs the ONNX; map by name.
- Some fp16 engines lose detections on 8.6 (RT-DETR v1/v2); those build at fp32 there.
