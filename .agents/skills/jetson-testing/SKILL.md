---
name: jetson-testing
description: Use when a trtutils change must be tested or benchmarked on a remote Jetson device (any JetPack / TensorRT version) - "run the suite on the Jetson", "test on device", "check an older TRT", "benchmark on the Jetson", "remote device run".
---

# Remote Jetson testing

A Jetson has an integrated GPU and usually an older TensorRT than the desktop (it is pinned by
its JetPack release), so it catches TensorRT version differences, output reordering and
unified-memory paths a discrete GPU does not.

## Rules

- Connection details come from the environment only: `$JETSON_HOST`, `$JETSON_USER`,
  `$JETSON_PW`. Never hardcode, log, or commit them, or the host name/IP.
- Never touch the user's own checkouts on the device. Make your own worktree.

## Steps

1. Write a small paramiko helper in your scratch directory (not the repo):

   ```python
   import os, sys, paramiko
   c = paramiko.SSHClient()
   c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
   c.connect(os.environ["JETSON_HOST"], username=os.environ["JETSON_USER"],
             password=os.environ["JETSON_PW"])
   _, out, err = c.exec_command(sys.argv[1], timeout=600)
   print(out.read().decode(), err.read().decode())
   ```

   Use `c.open_sftp().put(local, remote)` for file copies. Check the device's versions first:
   `cat /etc/nv_tegra_release` and `python3 -c "import tensorrt; print(tensorrt.__version__)"`.

2. Create a worktree on the device from its trtutils clone (`git fetch origin` there first),
   the same way as the `worktree` skill: `PYTHONPATH=src` with the device's existing python
   env, and data entries symlinked from that clone's `data/`.

3. Copy the ONNX files the tests need via SFTP into a real `data/<family>/` directory of your
   worktree. The device's own `data/` may hold older exports; use the desktop's files so both
   machines test the same models.

4. Start long runs detached, because the SSH channel hangs on long commands:

   ```bash
   cd <device-wt> && setsid nohup env PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests \
     > pytest.log 2>&1 < /dev/null &
   ```

   Poll with `tail -n 5 <device-wt>/pytest.log` until the summary line appears. First runs
   build engines for the device's TensorRT and take much longer.

5. Before calling a failure real: rerun it alone; known flakes include the jetson tegrastats
   benchmark test (`tests/jetson/test_benchmark.py`). Compare against `main` on the device
   with the same data, and bisect if `main` fails too.

6. Benchmarks: lock clocks first (`benchmarking` skill).

7. Clean up when done: remove the device worktree and any copied files outside it.

## Older TensorRT notes

- `FLAGS.TRT_10` is also True on 8.6 (it marks the I/O tensor API); gate on
  `FLAGS.TRT_VERSION` for real version checks.
- Engine outputs may be reordered vs the ONNX on 8.x; map by name.
- Some fp16 engines lose detections on 8.6 (RT-DETR v1/v2); those build at fp32 there.
- Near-tied scores can swap order between engines; compare detections as sets, not by index.
