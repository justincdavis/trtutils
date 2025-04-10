# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

try:
    from common import yolo_results
except ModuleNotFoundError:
    from .common import yolo_results


def test_yolo_dla_7_preproc_cpu_results():
    yolo_results(7, preprocessor="cpu", use_dla=True)


def test_yolo_dla_7_preproc_cuda_results():
    yolo_results(7, preprocessor="cuda", use_dla=True)


# cannot build yolox engine with trtexec
# def test_yolo_dla_x_results():
#     yolo_dla_results(0)


if __name__ == "__main__":
    yolo_dla_results(7)
