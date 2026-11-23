# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple

from ._deim import export_deim, export_deimv2
from ._depth_anything import (
    export_depth_anything_v1,
    export_depth_anything_v2,
    export_depth_anything_v3,
)
from ._dfine import export_dfine
from ._hands23 import export_hands23
from ._hoi_detr import export_hoi_detr
from ._ppocr import export_ppocr
from ._rfdetr import export_rfdetr
from ._rtdetr import export_rtdetrv1, export_rtdetrv2, export_rtdetrv3
from ._timm import export_timm_classifier
from ._torchvision import export_torchvision_classifier
from ._ultralytics import export_ultralytics
from ._yolo import (
    export_yolov7,
    export_yolov9,
    export_yolov10,
    export_yolov12,
    export_yolov13,
)
from ._yolox import export_yolox

if TYPE_CHECKING:
    from pathlib import Path


class Exporter(NamedTuple):
    """
    An export function and what it supports.

    Attributes
    ----------
    fn : Callable[..., Path]
        The export function, called as ``fn(directory, config, python_path, bin_path,
        model, opset, imgsz, **kwargs)``.
    batch : bool
        Whether ``fn`` accepts ``batch`` and ``dynamic``. When False, a batch above 1
        or a dynamic request is rejected.
    dynamic_default : bool
        Whether the export is dynamic when neither ``batch`` nor ``dynamic`` is given.

    """

    fn: Callable[..., Path]
    batch: bool = False
    dynamic_default: bool = False


# config entries name their exporter with the "exporter" key
EXPORTERS: dict[str, Exporter] = {
    "deim": Exporter(export_deim),
    "deimv2": Exporter(export_deimv2),
    "depth_anything_v1": Exporter(export_depth_anything_v1),
    "depth_anything_v2": Exporter(export_depth_anything_v2),
    "depth_anything_v3": Exporter(export_depth_anything_v3),
    "dfine": Exporter(export_dfine),
    "hands23": Exporter(export_hands23),
    "hoi_detr": Exporter(export_hoi_detr),
    "ppocr": Exporter(export_ppocr),
    "rfdetr": Exporter(export_rfdetr),
    "rtdetrv1": Exporter(export_rtdetrv1),
    "rtdetrv2": Exporter(export_rtdetrv2),
    "rtdetrv3": Exporter(export_rtdetrv3),
    "timm": Exporter(export_timm_classifier, batch=True),
    "torchvision": Exporter(export_torchvision_classifier, batch=True, dynamic_default=True),
    "ultralytics": Exporter(export_ultralytics, batch=True),
    "yolov7": Exporter(export_yolov7),
    "yolov9": Exporter(export_yolov9),
    "yolov10": Exporter(export_yolov10, batch=True),
    "yolov12": Exporter(export_yolov12, batch=True),
    "yolov13": Exporter(export_yolov13, batch=True),
    "yolox": Exporter(export_yolox),
}

__all__ = [
    "EXPORTERS",
    "Exporter",
    "export_deim",
    "export_deimv2",
    "export_depth_anything_v1",
    "export_depth_anything_v2",
    "export_depth_anything_v3",
    "export_dfine",
    "export_hands23",
    "export_hoi_detr",
    "export_ppocr",
    "export_rfdetr",
    "export_rtdetrv1",
    "export_rtdetrv2",
    "export_rtdetrv3",
    "export_timm_classifier",
    "export_torchvision_classifier",
    "export_ultralytics",
    "export_yolov7",
    "export_yolov9",
    "export_yolov10",
    "export_yolov12",
    "export_yolov13",
    "export_yolox",
]
