# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: T201
"""Per-stage latency of one model per task on a 1080p frame."""

from __future__ import annotations

import functools
import inspect
import itertools
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from tqdm import tqdm

from trtutils import image, models
from trtutils.compat._libs import cudart
from trtutils.core._cuda import cuda_call
from trtutils.download._download import load_model_configs
from trtutils.models._model import Model

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
    # wrapper class per ONNX for multi-engine models (OCR det + rec)
    parts: dict[str, Any] = field(default_factory=dict)


# image-model base class -> task
TASKS = {
    image.Detector: "detection",
    image.Classifier: "classification",
    image.Segmenter: "segmentation",
    image.PoseEstimator: "pose",
    image.OBBDetector: "obb",
    image.DepthEstimator: "depth",
    image.HandInteractionDetector: "hand_interaction",
}
# per-task overrides of TaskModel fields
TASK_OVERRIDES: dict[str, dict[str, Any]] = {"obb": {"image": "aerial.jpg"}}


def _norm(name: str) -> str:
    return re.sub("[^a-z0-9]", "", name.lower()).replace("transformer", "")


def _wrapper(name: str, classes: list[Any]) -> Any:  # noqa: ANN401
    """The wrapper class of a model: the family's only class, else the longest name prefix."""
    if len(classes) == 1:
        return classes[0]
    # torchvision nets share one config, so match e.g. wide_resnet50_2 to WideResNet
    prefixed = [c for c in classes if _norm(name).startswith(_norm(c.__name__))]
    return max(prefixed, key=lambda c: len(c.__name__)) if prefixed else None


def build_task_models() -> list[TaskModel]:
    """One TaskModel per downloadable model, from the wrapper classes and download configs."""
    by_type: dict[str, list[Any]] = {}
    for cls in vars(models).values():
        is_model = inspect.isclass(cls) and issubclass(cls, Model)
        if is_model and any(issubclass(cls, base) for base in TASKS):
            by_type.setdefault(cls._model_type, []).append(cls)  # noqa: SLF001
    specs = []
    for model_type, names in load_model_configs().items():
        for name in names:
            cls = _wrapper(name, by_type.get(model_type, []))
            if cls is None:
                continue
            task = next(t for base, t in TASKS.items() if issubclass(cls, base))
            specs.append(TaskModel(task, name, cls, **TASK_OVERRIDES.get(task, {})))
    ocr = {"det": models.PPOCRDetector, "rec": models.PPOCRRecognizer}
    specs.append(TaskModel("ocr", "ppocrv4", models.PPOCR, image="text", parts=ocr))
    return specs


TASK_MODELS = build_task_models()


def select_task_models(
    names: list[str] | None = None,
    tasks: list[str] | None = None,
    *,
    all_models: bool = False,
) -> list[TaskModel]:
    """
    Pick task models: ``names`` if given, else the first (smallest) model per wrapper class
    (every model with ``all_models``), optionally limited to ``tasks``.
    """
    known = {m.name for m in TASK_MODELS}
    unknown = sorted(set(names or []) - known)
    if unknown:
        err_msg = f"Unknown task models: {', '.join(unknown)}"
        raise ValueError(err_msg)
    if names is not None:
        return [m for m in TASK_MODELS if m.name in names and (tasks is None or m.task in tasks)]
    seen: set[Any] = set()
    picked = []
    for m in TASK_MODELS:
        if tasks is not None and m.task not in tasks:
            continue
        if all_models or m.cls not in seen:
            picked.append(m)
        seen.add(m.cls)
    return picked


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


@functools.lru_cache(maxsize=None)
def _cache_dir() -> Path:
    """One cache per trtutils under test, so A/B runs never share ONNX files or engines."""
    prov = trtutils_provenance()
    return TASK_DIR / f"{str(prov['trtutils_sha'])[:10]}{'-dirty' if prov['dirty'] else ''}"


def _engine(cls: Any, name: str) -> Path:  # noqa: ANN401
    """Download the ONNX and build the engine for one model, once, at its default precision."""
    cache = _cache_dir()
    onnx = cache / f"{name}.onnx"
    engine = cache / "engines" / f"{name}.engine"
    if not onnx.exists():
        onnx.parent.mkdir(parents=True, exist_ok=True)
        print(f"\tDownloading {name}")
        cls.download(name, onnx)
    if not engine.exists():
        print(f"\tBuilding {engine.name}")
        engine.parent.mkdir(parents=True, exist_ok=True)
        cls.build(onnx, engine)
    return engine


def load_model(spec: TaskModel, preprocessor: str, postprocessor: str = "cpu") -> Any:  # noqa: ANN401
    """Build (if needed) and load the model of ``spec`` with the given pre/postprocessor."""
    if spec.parts:
        det = _engine(spec.parts["det"], f"{spec.name}_det")
        rec = _engine(spec.parts["rec"], f"{spec.name}_rec")
        return spec.cls(det, rec, postprocessor=postprocessor, warmup=True)
    engine = _engine(spec.cls, spec.name)
    return spec.cls(engine, preprocessor=preprocessor, postprocessor=postprocessor, warmup=True)


def _has_cuda_post(spec: TaskModel) -> bool:
    """Whether the task has a CUDA postprocessor (OCR: both of its parts)."""
    classes = spec.parts.values() if spec.parts else [spec.cls]
    return all(c._CUDA_POSTPROCESSOR is not None for c in classes)  # noqa: SLF001


def _sync() -> None:
    cuda_call(cudart.cudaDeviceSynchronize())


class StageTimer:
    """
    Wrap a model's stage methods to attribute end2end time to stages.

    Every stage syncs the device before its clock stops, so GPU work is billed to
    the stage that queued it: ``prepare_inputs`` (preprocessing) to preprocess, the
    engine to infer, and a CUDA postprocessor's kernels to postprocess. ``infer``
    and ``_infer_post`` (end2end's path, which skips ``infer`` with
    ``postprocessor="cuda"``) cover preprocessing plus the engine; whatever is left
    of the end2end call (result extraction, OCR crops) is ``extract``.
    """

    def __init__(self, image_models: list[Any]) -> None:
        self.times: dict[str, float] = dict.fromkeys(STAGES, 0.0)
        self._active: set[str] = set()
        for m in image_models:
            m.prepare_inputs = self._wrap(m.prepare_inputs, "preprocess")
            m.infer = self._wrap(m.infer, "infer")
            m._infer_post = self._wrap(m._infer_post, "infer")  # noqa: SLF001
            m.postprocess = self._wrap(m.postprocess, "postprocess")

    def _wrap(self, fn: Callable[..., Any], stage: str) -> Callable[..., Any]:
        def timed(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            # _infer_post calls infer on the CPU postprocessor path, count it once
            if stage in self._active:
                return fn(*args, **kwargs)
            self._active.add(stage)
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
                _sync()
            finally:
                self._active.discard(stage)
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


def _resolved(model: Any) -> str:  # noqa: ANN401
    """The preprocessor a model actually runs (OCR: its detector's)."""
    return (model if hasattr(model, "prepare_inputs") else model._detector)._preprocessor_method  # noqa: SLF001


def _key(name: str, pre: str, post: str) -> str:
    return f"{name}/{pre}" + ("" if post == "cpu" else f"/{post}-post")


def bench_task(
    spec: TaskModel,
    preprocessor: str,
    warmup: int,
    iterations: int,
    postprocessor: str = "cpu",
) -> dict[str, Any] | None:
    """Time ``iterations`` end2end calls, both plain and split into stages (None if skipped)."""
    image = load_frame(spec.image)
    key = _key(spec.name, preprocessor, postprocessor)
    try:
        model = load_model(spec, preprocessor, postprocessor)
    except ValueError as e:
        # the engine's outputs do not fit the CUDA postprocessor (e.g. a YOLOv8 head)
        if postprocessor == "cpu":
            raise
        print(f"\tSkipping {key}: {e}")
        return None
    if _resolved(model) != preprocessor:
        print(f"\tSkipping {key}: ran as {_resolved(model)}")
        return None
    for _ in range(warmup):
        model.end2end(image)

    # plain end2end latency, no extra syncs
    totals = []
    for _ in tqdm(range(iterations), desc=f"{key} e2e"):
        t0 = time.perf_counter()
        model.end2end(image)
        totals.append(time.perf_counter() - t0)

    timer = StageTimer(_image_models(model))
    stages: dict[str, list[float]] = {s: [] for s in STAGES}
    for _ in tqdm(range(iterations), desc=f"{key} stages"):
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


def _write_stage(device: str, stage: str, warmup: int, iterations: int, data: dict) -> None:
    write_stage_snapshot(
        device, stage, "tasks", {"warmup": warmup, "iterations": iterations, "data": data}
    )


def run_tasks(
    device: str,
    names: list[str] | None,
    preprocessors: list[str],
    warmup: int,
    iterations: int,
    *,
    postprocessors: tuple[str, ...] = ("cpu", "cuda"),
    overwrite: bool = False,
    stage: str | None = None,
    tasks: list[str] | None = None,
    all_models: bool = False,
) -> dict[str, Any]:
    """
    Benchmark the task models picked by ``select_task_models``; a model that fails is skipped.

    Each model runs with every preprocessor and every postprocessor in ``postprocessors``;
    "cuda" is skipped for tasks and engines without a CUDA postprocessor.

    Results go to data/tasks/<device>.json (cached by key unless ``overwrite``), or, with
    ``stage``, are always measured fresh into data/perf-series/<device>/stage-<stage>.json.
    """
    data: dict[str, Any] = {} if stage is not None else get_data(device, "tasks", [])
    cached = 0
    for spec in select_task_models(names, tasks, all_models=all_models):
        # OCR preprocesses on the CPU only
        pres = ["cpu"] if spec.parts else preprocessors
        for prep, post in itertools.product(pres, postprocessors):
            key = _key(spec.name, prep, post)
            if key in data and not overwrite:
                print(f"Skipping {key} (already exists)")
                cached += 1
                continue
            if post == "cuda" and not _has_cuda_post(spec):
                print(f"Skipping {key}: no CUDA postprocessor")
                continue
            print(f"Benchmarking {key}")
            try:
                res = bench_task(spec, prep, warmup, iterations, post)
            except Exception as e:  # noqa: BLE001
                print(f"\tFAILED {key}: {type(e).__name__}: {e}")
                continue
            if res is None:
                continue
            data[key] = res
            res.update(trtutils_provenance())
            parts = ", ".join(f"{s} {res['stages'][s]['median']:.2f}" for s in STAGES)
            e2e = res["end2end"]
            print(f"\t{key}: end2end median {e2e['median']:.2f}ms max {e2e['max']:.2f}ms ({parts})")
            if stage is None:
                write_data(device, "tasks", data)
            else:
                _write_stage(device, stage, warmup, iterations, data)
    if stage is None and cached:
        print(f"{cached} cached entries skipped; pass --overwrite or --stage to re-measure")
    return data
