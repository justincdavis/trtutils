# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
"""
File showcasing the trtexec wrapper utilities.

Demonstrates :func:`trtutils.find_trtexec` for locating the binary,
:func:`trtutils.run_trtexec` for running raw commands, and
:func:`trtutils.trtexec.build_engine` for building an engine through the
external tool. The resulting engine is then loaded with
:class:`trtutils.TRTEngine` to confirm round-trip compatibility.

Exits cleanly when ``trtexec`` is not installed on the system.
"""

from __future__ import annotations

from pathlib import Path

from trtutils import TRTEngine, find_trtexec, run_trtexec, set_log_level
from trtutils import trtexec as trtexec_mod
from trtutils.download import download


def main() -> None:
    try:
        trtexec_path = find_trtexec()
    except FileNotFoundError as exc:
        print(f"Skipping: {exc}")
        return

    print(f"Found trtexec at: {trtexec_path}")

    # run a trivial command — the version banner — and print the first few lines
    success, stdout, _stderr = run_trtexec("--help")
    if not success:
        print("trtexec --help did not exit cleanly; skipping rest.")
        return
    head = "\n".join(stdout.splitlines()[:3])
    print(f"Banner:\n{head}")

    onnx_path = Path("/tmp/yolov8n.onnx")  # noqa: S108
    engine_path = Path("/tmp/yolov8n_trtexec.engine")  # noqa: S108

    if not onnx_path.exists():
        print("Downloading yolov8n ONNX model...")
        download("yolov8n", onnx_path, imgsz=640, simplify=True)

    if engine_path.exists():
        engine_path.unlink()

    print("\nBuilding engine via trtexec.build_engine(fp16=True)...")
    ok = trtexec_mod.build_engine(
        onnx_path,
        engine_path,
        fp16=True,
        shapes=[("images", (1, 3, 640, 640))],
    )
    if not ok:
        print("trtexec.build_engine reported failure.")
        return

    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"trtexec built engine: {engine_path} ({size_mb:.2f} MB)")

    # round-trip: load with TRTEngine
    engine = TRTEngine(engine_path, warmup=True)
    engine.mock_execute()
    print(f"Loaded {engine.name} with TRTEngine, mock_execute OK")
    del engine


if __name__ == "__main__":
    set_log_level("ERROR")
    main()
