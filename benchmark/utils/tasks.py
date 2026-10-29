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
from .runners import trtutils_provenance, write_stage_snapshot
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
    TaskModel("detection", "yolov8n", models.YOLOv8),
    TaskModel("detection", "yolov10n", models.YOLOv10),
    TaskModel("detection", "yoloxn", models.YOLOX),
    TaskModel("detection", "rtdetrv2_r18", models.RTDETRv2),
    TaskModel("detection", "dfine_n", models.DFINE),
    TaskModel("classification", "yolov11n-cls", models.YOLOv11Cls),
    TaskModel("classification", "yolov26n-cls", models.YOLOv26Cls),
    TaskModel("classification", "yolov8n-cls", models.YOLOv8Cls),
    TaskModel("classification", "resnet18", models.ResNet),
    TaskModel("classification", "efficientnet_b0", models.EfficientNet),
    TaskModel("segmentation", "yolov11n-seg", models.YOLOv11Seg),
    TaskModel("segmentation", "yolov26n-seg", models.YOLOv26Seg),
    TaskModel("segmentation", "yolov8n-seg", models.YOLOv8Seg),
    TaskModel("pose", "yolov11n-pose", models.YOLOv11Pose),
    TaskModel("pose", "yolov26n-pose", models.YOLOv26Pose),
    TaskModel("pose", "yolov8n-pose", models.YOLOv8Pose),
    TaskModel("obb", "yolov11n-obb", models.YOLOv11OBB, image="aerial.jpg"),
    TaskModel("obb", "yolov26n-obb", models.YOLOv26OBB, image="aerial.jpg"),
    TaskModel("obb", "yolov8n-obb", models.YOLOv8OBB, image="aerial.jpg"),
    TaskModel("depth", "depth_anything_v2_small", models.DepthAnythingV2),
    TaskModel("depth", "depth_anything_v1_small", models.DepthAnythingV1),
    TaskModel("hand_interaction", "hands23_x101", models.Hands23),
    TaskModel(
        "ocr",
        "ppocrv4",
        models.PPOCR,
        image="text",
        parts={"det": models.PPOCRDetector, "rec": models.PPOCRRecognizer},
    ),
]


def _text_frame() -> np.ndarray:
    """A 1080p page of 14 printed text lines, so OCR runs its recognizer (photos hold no text)."""
    image = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 255, dtype=np.uint8)
    for i in range(14):
        line = f"Line {i + 1}: the quick brown fox jumps over the lazy dog {1000 + 37 * i}"
        cv2.putText(image, line, (60, 70 + 72 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    return image


def load_frame(name: str) -> np.ndarray:
    """Read a data/ image and resize it to a 1080p frame ("text" renders a page of text)."""
    if name == "text":
        return _text_frame()
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


def load_model(spec: TaskModel, preprocessor: str, postprocessor: str = "cpu") -> Any:  # noqa: ANN401
    """Build (if needed) and load the model of ``spec`` with ``preprocessor``."""
    if spec.parts:
        det = _engine(spec.parts["det"], f"{spec.name}_det", spec.imgsz)
        rec = _engine(spec.parts["rec"], f"{spec.name}_rec", None)
        return spec.cls(det, rec, postprocessor=postprocessor, warmup=True)
    engine = _engine(spec.cls, spec.name, spec.imgsz)
    return spec.cls(engine, preprocessor=preprocessor, postprocessor=postprocessor, warmup=True)


def _sync() -> None:
    cuda_call(cudart.cudaDeviceSynchronize())


class StageTimer:
    """
    Wrap a model's stage methods to attribute end2end time to stages.

    ``prepare_inputs`` and ``prepare_regions`` (preprocessing) sync the device so
    GPU preprocessing is not billed to the engine; ``infer`` / ``_infer_post`` /
    ``_infer_regions`` cover the engine and sync too; ``postprocess`` is timed as
    is. A stage nested in another is taken out of its parent, and whatever is left
    of the end2end call (result extraction, OCR crops) is ``extract``.
    """

    _WRAPPED = (
        ("prepare_inputs", "preprocess", True),
        ("prepare_regions", "preprocess", True),
        ("infer", "infer", True),
        ("_infer_post", "infer", True),
        ("_infer_regions", "infer", True),
        ("postprocess", "postprocess", False),
    )

    def __init__(self, image_models: list[Any]) -> None:
        self.times: dict[str, float] = dict.fromkeys(STAGES, 0.0)
        self._active: set[str] = set()
        self._children: list[float] = []
        for m in image_models:
            for name, stage, sync in self._WRAPPED:
                if hasattr(m, name):
                    setattr(m, name, self._wrap(getattr(m, name), stage, sync=sync))

    def _wrap(self, fn: Callable[..., Any], stage: str, *, sync: bool = False) -> Callable[..., Any]:
        def timed(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            # _infer_post calls infer on the CPU path, count the stage once
            if stage in self._active:
                return fn(*args, **kwargs)
            self._active.add(stage)
            self._children.append(0.0)
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
                if sync:
                    _sync()
            finally:
                self._active.discard(stage)
                elapsed = time.perf_counter() - t0
                self.times[stage] += elapsed - self._children.pop()
                if self._children:
                    self._children[-1] += elapsed
            return out

        return timed

    def reset(self) -> None:
        self.times = dict.fromkeys(STAGES, 0.0)

    def split(self, total: float) -> dict[str, float]:
        """Per-stage seconds for one end2end call that took ``total`` seconds."""
        t = self.times
        return {
            "preprocess": t["preprocess"],
            "infer": t["infer"],
            "postprocess": t["postprocess"],
            "extract": total - sum(t.values()),
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
    postprocessor: str = "cpu",
) -> dict[str, Any]:
    """Time ``iterations`` end2end calls, both plain and split into stages."""
    image = load_frame(spec.image)
    model = load_model(spec, preprocessor, postprocessor)
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
        "postprocessor": postprocessor,
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
    postprocessors: tuple[str, ...] = ("cpu",),
    overwrite: bool = False,
    stage: str | None = None,
) -> dict[str, Any]:
    """
    Benchmark every task model (or those in ``names``).

    Results go to data/tasks/<device>.json (cached by key unless ``overwrite``), or, with
    ``stage``, are always measured fresh into data/perf-series/<device>/stage-<stage>.json.
    """
    data: dict[str, Any] = {} if stage is not None else get_data(device, "tasks", [])
    cached = 0
    for spec in TASK_MODELS:
        if names is not None and spec.name not in names:
            continue
        # OCR preprocesses on the CPU only
        runs = [(pre, post) for post in postprocessors for pre in (["cpu"] if spec.parts else preprocessors)]
        for prep, post in runs:
            key = f"{spec.name}/{prep}" + ("" if post == "cpu" else f"/{post}-post")
            if key in data and not overwrite:
                print(f"Skipping {key} (already exists)")
                cached += 1
                continue
            if post == "cuda" and not (spec.parts or spec.cls._cuda_postprocess):
                print(f"Skipping {key}: no CUDA postprocessor")
                continue
            print(f"Benchmarking {key}")
            data[key] = bench_task(spec, prep, warmup, iterations, post)
            data[key].update(trtutils_provenance())
            res = data[key]
            parts = ", ".join(f"{s} {res['stages'][s]['median']:.2f}" for s in STAGES)
            e2e = res["end2end"]
            print(f"\t{key}: end2end median {e2e['median']:.2f}ms max {e2e['max']:.2f}ms ({parts})")
            if stage is None:
                write_data(device, "tasks", data)
    if stage is not None:
        write_stage_snapshot(
            device, stage, "tasks", {"warmup": warmup, "iterations": iterations, "data": data}
        )
    elif cached:
        print(f"{cached} cached entries skipped; pass --overwrite or --stage to re-measure")
    return data
