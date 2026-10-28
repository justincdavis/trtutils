# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: T201
"""Per-stage latency of one model per task on a 1080p frame."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from tqdm import tqdm

from trtutils import models
from trtutils.compat._libs import cudart
from trtutils.core._cuda import cuda_call

from .config import REPO_DIR
from .data import get_data, write_data
from .timing import compute_results

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

TASK_DIR = REPO_DIR / "data" / "tasks"
FRAME_SIZE = (1920, 1080)
STAGES = ("preprocess", "infer", "postprocess", "extract")


@dataclass
class TaskModel:
    """One benchmarked model: its wrapper class(es) and how to build and call it."""

    task: str
    name: str
    cls: Any
    image: str = "street.jpg"
    imgsz: int | None = None
    # the downloaded model name per ONNX, when it differs from ``name``
    parts: dict[str, Any] = field(default_factory=dict)


TASK_MODELS = [
    TaskModel("detection", "yolov11n", models.YOLOv11),
    TaskModel("detection", "yolov26n", models.YOLOv26),
    TaskModel("classification", "yolov11n-cls", models.YOLOv11Cls),
    TaskModel("classification", "yolov26n-cls", models.YOLOv26Cls),
    TaskModel("segmentation", "yolov11n-seg", models.YOLOv11Seg),
    TaskModel("segmentation", "yolov26n-seg", models.YOLOv26Seg),
    TaskModel("pose", "yolov11n-pose", models.YOLOv11Pose),
    TaskModel("pose", "yolov26n-pose", models.YOLOv26Pose),
    TaskModel("obb", "yolov11n-obb", models.YOLOv11OBB, image="aerial.jpg"),
    TaskModel("obb", "yolov26n-obb", models.YOLOv26OBB, image="aerial.jpg"),
    TaskModel("depth", "depth_anything_v2_small", models.DepthAnythingV2),
    TaskModel("hand_interaction", "hands23_x101", models.Hands23),
    TaskModel(
        "ocr",
        "ppocrv4",
        models.PPOCR,
        parts={"det": models.PPOCRDetector, "rec": models.PPOCRRecognizer},
    ),
]


def load_frame(name: str) -> np.ndarray:
    """Read a data/ image and resize it to a 1080p frame."""
    image = cv2.imread(str(REPO_DIR / "data" / name))
    if image is None:
        err_msg = f"Could not read data/{name}"
        raise FileNotFoundError(err_msg)
    if image.shape[1::-1] != FRAME_SIZE:
        image = cv2.resize(image, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
    return image


def _engine(cls: Any, name: str, imgsz: int | None) -> Path:  # noqa: ANN401
    """Download the ONNX and build the engine for one model, once, at its default precision."""
    onnx = TASK_DIR / f"{name}.onnx"
    engine = TASK_DIR / "engines" / f"{name}.engine"
    if not onnx.exists():
        print(f"\tDownloading {name}")
        cls.download(name, onnx, imgsz=imgsz)
    if not engine.exists():
        print(f"\tBuilding {engine.name}")
        engine.parent.mkdir(parents=True, exist_ok=True)
        cls.build(onnx, engine, imgsz=imgsz)
    return engine


def load_model(spec: TaskModel, preprocessor: str) -> Any:  # noqa: ANN401
    """Build (if needed) and load the model of ``spec`` with ``preprocessor``."""
    if spec.parts:
        det = _engine(spec.parts["det"], f"{spec.name}_det", spec.imgsz)
        rec = _engine(spec.parts["rec"], f"{spec.name}_rec", None)
        return spec.cls(det, rec, warmup=True)
    engine = _engine(spec.cls, spec.name, spec.imgsz)
    return spec.cls(engine, preprocessor=preprocessor, warmup=True)


def _sync() -> None:
    cuda_call(cudart.cudaDeviceSynchronize())


class StageTimer:
    """
    Wrap a model's stage methods to attribute end2end time to stages.

    ``prepare_inputs`` (preprocessing) syncs the device so GPU preprocessing is
    not billed to the engine; ``infer`` covers preprocessing plus the engine;
    ``postprocess`` is timed as is; whatever is left of the end2end call
    (result extraction, OCR crops) is ``extract``.
    """

    def __init__(self, image_models: list[Any]) -> None:
        self.times: dict[str, float] = dict.fromkeys(STAGES, 0.0)
        for m in image_models:
            m.prepare_inputs = self._wrap(m.prepare_inputs, "preprocess", sync=True)
            m.infer = self._wrap(m.infer, "infer")
            m.postprocess = self._wrap(m.postprocess, "postprocess")

    def _wrap(self, fn: Callable[..., Any], stage: str, *, sync: bool = False) -> Callable[..., Any]:
        def timed(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            if sync:
                _sync()
            self.times[stage] += time.perf_counter() - t0
            return out

        return timed

    def reset(self) -> None:
        self.times = dict.fromkeys(STAGES, 0.0)

    def split(self, total: float) -> dict[str, float]:
        """Per-stage seconds for one end2end call that took ``total`` seconds."""
        t = self.times
        # infer wraps prepare_inputs, so take preprocessing back out of it
        infer = t["infer"] - t["preprocess"]
        return {
            "preprocess": t["preprocess"],
            "infer": infer,
            "postprocess": t["postprocess"],
            "extract": total - t["infer"] - t["postprocess"],
        }


def _image_models(model: Any) -> list[Any]:  # noqa: ANN401
    """The ImageModels inside ``model`` (OCR holds a detector and a recognizer)."""
    if hasattr(model, "prepare_inputs"):
        return [model]
    return [model._detector, model._recognizer]  # noqa: SLF001


def bench_task(
    spec: TaskModel,
    preprocessor: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Time ``iterations`` end2end calls, both plain and split into stages."""
    image = load_frame(spec.image)
    model = load_model(spec, preprocessor)
    for _ in range(warmup):
        model.end2end(image)

    # plain end2end latency, no extra syncs
    totals = []
    for _ in tqdm(range(iterations), desc=f"{spec.name} {preprocessor} e2e"):
        t0 = time.perf_counter()
        model.end2end(image)
        totals.append(time.perf_counter() - t0)

    timer = StageTimer(_image_models(model))
    stages: dict[str, list[float]] = {s: [] for s in STAGES}
    for _ in tqdm(range(iterations), desc=f"{spec.name} {preprocessor} stages"):
        timer.reset()
        t0 = time.perf_counter()
        model.end2end(image)
        for stage, sec in timer.split(time.perf_counter() - t0).items():
            stages[stage].append(sec)

    del model
    return {
        "task": spec.task,
        "model": spec.name,
        "preprocessor": preprocessor,
        "frame": list(FRAME_SIZE),
        "end2end": compute_results(totals),
        "stages": {s: compute_results(v) for s, v in stages.items()},
    }


def run_tasks(
    device: str,
    names: list[str] | None,
    preprocessors: list[str],
    warmup: int,
    iterations: int,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Benchmark every task model (or those in ``names``) and save to data/tasks/<device>.json."""
    data: dict[str, Any] = get_data(device, "tasks", [])
    for spec in TASK_MODELS:
        if names is not None and spec.name not in names:
            continue
        # OCR preprocesses on the CPU only
        for prep in ["cpu"] if spec.parts else preprocessors:
            key = f"{spec.name}/{prep}"
            if key in data and not overwrite:
                print(f"Skipping {key} (already exists)")
                continue
            print(f"Benchmarking {key}")
            data[key] = bench_task(spec, prep, warmup, iterations)
            res = data[key]
            parts = ", ".join(f"{s} {res['stages'][s]['mean']:.2f}" for s in STAGES)
            print(f"\t{key}: end2end {res['end2end']['mean']:.2f}ms ({parts})")
            write_data(device, "tasks", data)
    return data
