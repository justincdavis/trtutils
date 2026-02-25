# Copyright (c) 2025 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from trtutils.download import _download as dl

MODEL_CONFIGS = dl.load_model_configs()
TEST_MODELS = [
    "deim_dfine_n",
    "deim_rtdetrv2_r18",
    "deimv2_atto",
    "dfine_n",
    "rfdetr_n",
    "rtdetrv1_r18",
    "rtdetrv2_r18",
    "rtdetrv3_r18",
    "yolov3tu",
    "yolov5nu",
    "yolov7t",
    "yolov8n",
    "yolov9t",
    "yolov9tu",
    "yolov10n",
    "yolov10nu",
    "yolov11n",
    "yolov12n",
    "yolov12nu",
    "yolov13n",
    "yoloxn",
]
